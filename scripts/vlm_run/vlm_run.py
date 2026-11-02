"""Run a vision-language model over a chat JSONL and write its raw replies (W4-B).

``python vlm_run.py --backend openai|vllm|transformers|gold --chat <chat.jsonl>
--out <pred.jsonl> [--model <id>] [--image-root <dir>] [--max-side N]
[--max-tokens 512] [--repeat k] [--limit n] [--ids <file>] [--dry-run] ...``

Input: the chat JSONL of ``vlm_to_chat.py`` in any of its three formats
(``openai`` / ``qwen`` messages, ``internvl`` conversations); the images it
names are read from disk (``--image-root`` resolves relative paths) or from
``data:`` URLs. Output: one line per request, ``{"id", "raw_text", "meta"}``,
the file ``vlm_score.py --pred`` reads. ``meta`` carries the model id and the
snapshot the server reported, the image sizes actually sent, latency, token
counts when the backend gives them, and the error when a request failed
(``raw_text`` is then ``null``).

Backends, all behind :class:`Backend` (``generate(requests) -> replies``):

* ``openai`` -- any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI,
  Gemini's and Anthropic's compatibility layers, ``vllm serve``). The key is
  read from the environment variable named by ``--api-key-env``; it is never
  printed, never written, and redacted from error messages. Temperature 0,
  fixed ``--max-tokens``, retries with exponential backoff, ``--workers``
  parallel requests.
* ``vllm`` -- offline ``vllm.LLM`` over a local or Hub model id, batched;
  ``--tensor-parallel``, ``--dtype``, ``--max-model-len``, ``--gpu-mem`` and
  ``--engine-kwargs`` go straight through.
* ``transformers`` -- the fallback for Qwen3-VL / InternVL (HF-format
  checkpoints): ``AutoProcessor`` + ``AutoModelForImageTextToText.generate``,
  one request at a time.
* ``gold`` -- no model: the reply a perfect model would give, from the labels
  of ``--truth`` through ``vlm_check.gold_reply``. The end-to-end check of the
  chain (``vlm_to_chat -> vlm_run -> vlm_score`` must score 1.0).

``vllm`` / ``transformers`` / ``torch`` are imported only when their backend is
built, so ``--dry-run`` (validate the inputs, print the first prompt, load
nothing) and the ``gold`` and ``openai`` backends run on a machine without them.

Resume: ids already answered in ``--out`` are skipped (rows whose
``raw_text`` is ``null`` are re-run unless ``--keep-errors``). ``--repeat k``
asks every request ``k`` times and writes ``raw_text_1 .. raw_text_k`` beside
``raw_text`` (= ``raw_text_1``), one ``meta.repeats`` entry each; split them
with ``vlm_collect.py split`` to score each repeat.

Exit codes: 0 all answered, 2 usage / input error, 3 some requests failed
(their rows are written with ``meta.error`` so a re-run picks them up).

Self-contained: no ``tda`` import; only the ``gold`` backend needs
``vlm_io.py`` / ``vlm_check.py`` (from ``scripts/`` or the pack's ``tools/``).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

HERE = Path(__file__).resolve().parent
#: Where ``vlm_io`` / ``vlm_check`` may live: beside this file, one directory up
#: (``scripts/`` or the pack's ``tools/``), and the repo root for ``tda``.
_SEARCH = [HERE, HERE.parent, HERE.parent.parent]

BACKENDS = ("openai", "vllm", "transformers", "gold")
DEFAULT_MAX_TOKENS = 512
RETRY_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)
REDACTED = "***"


# --------------------------------------------------------------------------- #
# small IO (duplicated from vlm_io on purpose: this file must run alone)
# --------------------------------------------------------------------------- #
def read_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path, rows: Iterable[dict]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
        fh.flush()
    return n


class Log:
    """stderr + optional file; every message passes the redactor first."""

    def __init__(self, path: Optional[str] = None, quiet: bool = False):
        self.secrets: list[str] = []
        self.quiet = quiet
        self.fh = open(path, "a", encoding="utf-8") if path else None

    def hide(self, secret: Optional[str]) -> None:
        if secret:
            self.secrets.append(secret)

    def redact(self, text: Any) -> str:
        text = str(text)
        for s in self.secrets:
            text = text.replace(s, REDACTED)
        return text

    def __call__(self, msg: str) -> None:
        msg = self.redact(msg)
        if not self.quiet:
            print(f"[vlm-run] {msg}", file=sys.stderr, flush=True)
        if self.fh:
            self.fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
            self.fh.flush()

    def close(self) -> None:
        if self.fh:
            self.fh.close()


# --------------------------------------------------------------------------- #
# requests: the three chat formats, normalised
# --------------------------------------------------------------------------- #
@dataclass
class Request:
    """One question: ordered text / image parts, plus the chat file's ``meta``."""

    id: str
    system: Optional[str]
    parts: list[tuple[str, str]]  # ("text", str) | ("image", ref)
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(p for kind, p in self.parts if kind == "text")

    @property
    def images(self) -> list[str]:
        return [p for kind, p in self.parts if kind == "image"]


def _openai_parts(content) -> list[tuple[str, str]]:
    if isinstance(content, str):
        return [("text", content)]
    parts = []
    for part in content:
        kind = part.get("type")
        if kind == "text":
            parts.append(("text", part["text"]))
        elif kind == "image_url":
            url = part["image_url"]
            parts.append(("image", url["url"] if isinstance(url, dict) else url))
        elif kind == "image":  # qwen
            parts.append(("image", part["image"]))
        else:
            raise ValueError(f"unknown content part type {kind!r}")
    return parts


def to_request(row: dict) -> Request:
    """A chat row of any of the three formats as a :class:`Request`."""
    rid = str(row["id"])
    meta = row.get("meta") or {}
    if "messages" in row:  # openai / qwen
        system = None
        parts: list[tuple[str, str]] = []
        for msg in row["messages"]:
            if msg["role"] == "system":
                system = msg["content"] if isinstance(msg["content"], str) else \
                    "\n".join(p.get("text", "") for p in msg["content"])
            elif msg["role"] == "user":
                parts += _openai_parts(msg["content"])
            else:
                raise ValueError(f"{rid}: a {msg['role']!r} turn in the input; only "
                                 "system + user are run")
        return Request(rid, system, parts, meta)
    if "conversations" in row:  # internvl
        human = next(c for c in row["conversations"] if c.get("from") == "human")
        text = human["value"]
        images = row.get("image") or []
        if isinstance(images, str):
            images = [images]
        # drop the placeholders; the images go first, as the template puts them
        for i in range(len(images), 0, -1):
            text = text.replace(f"Image-{i}: <image>\n", "")
        text = text.replace("<image>\n", "").replace("<image>", "").strip()
        parts = [("image", ref) for ref in images] + [("text", text)]
        return Request(rid, meta.get("system"), parts, meta)
    raise ValueError(f"{rid}: neither 'messages' nor 'conversations' -- not a chat row")


def load_requests(path) -> list[Request]:
    reqs = [to_request(row) for row in read_jsonl(path)]
    seen: set[str] = set()
    for r in reqs:
        if r.id in seen:
            raise ValueError(f"duplicate id {r.id!r} in {path}")
        seen.add(r.id)
    return reqs


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
@dataclass
class Picture:
    data: bytes
    mime: str
    size: tuple[int, int]          # what is sent
    source_size: tuple[int, int]   # what was on disk

    @property
    def data_url(self) -> str:
        return f"data:{self.mime};base64," + base64.b64encode(self.data).decode("ascii")


def resolve_path(ref: str, root: Optional[Path]) -> Path:
    """A chat image reference (path or ``file://`` URL) as a path; relative ones under ``root``."""
    if ref.startswith("file://"):
        ref = ref[len("file://"):]
        if len(ref) > 2 and ref[0] == "/" and ref[2] == ":":  # /D:/x -> D:/x
            ref = ref[1:]
    p = Path(ref)
    if not p.is_absolute() and root is not None:
        p = root / p
    return p


def _sniff_mime(data: bytes, suffix: str) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp"}.get(suffix.lower().lstrip("."), "image/png")


class ImagePrep:
    """Reads, optionally downsizes (``max_side``) and encodes the pictures sent.

    Frames repeat across records, so the last ``cache_size`` encodings are kept.
    """

    def __init__(self, root: Optional[str] = None, max_side: Optional[int] = None,
                 jpeg_quality: int = 92, cache_size: int = 64):
        self.root = Path(root) if root else None
        self.max_side = max_side
        self.jpeg_quality = jpeg_quality
        self.cache: dict[str, Picture] = {}
        self.cache_size = cache_size

    def _raw(self, ref: str) -> tuple[bytes, str]:
        if ref.startswith("data:"):
            head, _, payload = ref.partition(",")
            mime = head[5:].split(";")[0] or "image/png"
            return base64.b64decode(payload), mime
        path = resolve_path(ref, self.root)
        data = path.read_bytes()
        return data, _sniff_mime(data, path.suffix)

    def load(self, ref: str) -> Picture:
        hit = self.cache.get(ref)
        if hit is not None:
            return hit
        data, mime = self._raw(ref)
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img.load()
            source = img.size
            if self.max_side and max(img.size) > self.max_side:
                scale = self.max_side / max(img.size)
                img = img.resize((max(1, round(img.size[0] * scale)),
                                  max(1, round(img.size[1] * scale))))
                buf = io.BytesIO()
                if mime == "image/jpeg":
                    img.convert("RGB").save(buf, "JPEG", quality=self.jpeg_quality)
                else:
                    img.save(buf, "PNG")
                    mime = "image/png"
                data = buf.getvalue()
            pic = Picture(data, mime, img.size, source)
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[ref] = pic
        return pic

    def pil(self, ref: str):
        from PIL import Image

        pic = self.load(ref)
        img = Image.open(io.BytesIO(pic.data))
        img.load()
        return img.convert("RGB")

    def check(self, ref: str) -> Optional[str]:
        """``None`` when the reference can be read, else the reason."""
        if ref.startswith("data:"):
            return None
        p = resolve_path(ref, self.root)
        return None if p.is_file() else f"missing image {p}"


def openai_messages(req: Request, prep: ImagePrep) -> tuple[list[dict], list[Picture]]:
    """The request as OpenAI chat messages with ``data:`` images (every server takes those)."""
    pics: list[Picture] = []
    content: list[dict] = []
    for kind, value in req.parts:
        if kind == "text":
            content.append({"type": "text", "text": value})
        else:
            pic = prep.load(value)
            pics.append(pic)
            content.append({"type": "image_url", "image_url": {"url": pic.data_url}})
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": content})
    return messages, pics


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
@dataclass
class Reply:
    text: Optional[str] = None
    error: Optional[str] = None
    latency_s: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model_snapshot: Optional[str] = None
    finish_reason: Optional[str] = None
    image_sizes: list = field(default_factory=list)
    attempts: int = 1
    extra: dict = field(default_factory=dict)

    def meta(self) -> dict:
        d = {"latency_s": None if self.latency_s is None else round(self.latency_s, 3),
             "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
             "model_snapshot": self.model_snapshot, "finish_reason": self.finish_reason,
             "image_sizes_sent": self.image_sizes, "attempts": self.attempts,
             "error": self.error}
        d.update(self.extra)
        return d


class Backend:
    """``generate`` answers a batch in order; ``describe`` is what ``meta`` records."""

    name = "?"
    batch_size = 1

    def __init__(self, model: str, prep: ImagePrep, max_tokens: int, temperature: float,
                 log: Log):
        self.model, self.prep, self.max_tokens = model, prep, max_tokens
        self.temperature, self.log = temperature, log

    def describe(self) -> dict:
        return {"backend": self.name, "model": self.model, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "max_side": self.prep.max_side}

    def generate(self, requests: list[Request]) -> list[Reply]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _redact_exc(exc: BaseException, log: Log) -> str:
    return log.redact(f"{type(exc).__name__}: {exc}")[:2000]


class OpenAIBackend(Backend):
    """Any ``/chat/completions`` server. The key leaves this process only as a header."""

    name = "openai"

    def __init__(self, model, prep, max_tokens, temperature, log, *, base_url: str,
                 api_key: Optional[str], timeout: float = 120.0, max_retries: int = 5,
                 backoff: float = 2.0, workers: int = 1, seed: Optional[int] = None,
                 extra_body: Optional[dict] = None, extra_headers: Optional[dict] = None):
        super().__init__(model, prep, max_tokens, temperature, log)
        base = base_url.rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self._key = api_key
        self.timeout, self.max_retries, self.backoff = timeout, max_retries, backoff
        self.workers = max(1, workers)
        self.batch_size = self.workers
        self.seed, self.extra_body = seed, extra_body or {}
        self.extra_headers = extra_headers or {}
        log.hide(api_key)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"base_url": self.url, "api_key": "set" if self._key else "none",
                  "seed": self.seed, "workers": self.workers})
        return d

    def _body(self, req: Request) -> tuple[bytes, list[Picture]]:
        messages, pics = openai_messages(req, self.prep)
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "temperature": self.temperature,
                                "max_tokens": self.max_tokens}
        if self.seed is not None:
            body["seed"] = self.seed
        body.update(self.extra_body)
        return json.dumps(body).encode("utf-8"), pics

    def _post(self, data: bytes) -> tuple[int, dict, dict]:
        """``(status, headers, json)``; raises ``urllib.error`` on transport / HTTP errors."""
        import urllib.request

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        headers.update(self.extra_headers)
        request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            payload = resp.read().decode("utf-8", "replace")
            return resp.status, dict(resp.headers), json.loads(payload)

    def _one(self, req: Request) -> Reply:
        import urllib.error

        data, pics = self._body(req)
        sizes = [list(p.size) for p in pics]
        last_error = "no attempt"
        for attempt in range(1, self.max_retries + 2):
            t0 = time.perf_counter()
            try:
                _, _, out = self._post(data)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:500]
                except Exception:  # noqa: BLE001
                    pass
                last_error = self.log.redact(f"HTTP {exc.code}: {detail or exc.reason}")
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if exc.code not in RETRY_STATUS or attempt > self.max_retries:
                    return Reply(error=last_error, attempts=attempt, image_sizes=sizes,
                                 latency_s=time.perf_counter() - t0)
                self._sleep(attempt, retry_after)
                continue
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = _redact_exc(exc, self.log)
                if attempt > self.max_retries:
                    return Reply(error=last_error, attempts=attempt, image_sizes=sizes)
                self._sleep(attempt, None)
                continue
            latency = time.perf_counter() - t0
            try:
                choice = out["choices"][0]
                content = choice.get("message", {}).get("content")
                if isinstance(content, list):  # some servers return parts
                    content = "".join(p.get("text", "") for p in content
                                      if isinstance(p, dict))
                usage = out.get("usage") or {}
                return Reply(text=content, latency_s=latency, attempts=attempt,
                             prompt_tokens=usage.get("prompt_tokens"),
                             completion_tokens=usage.get("completion_tokens"),
                             model_snapshot=out.get("model"),
                             finish_reason=choice.get("finish_reason"), image_sizes=sizes,
                             extra={"system_fingerprint": out.get("system_fingerprint")})
            except (KeyError, IndexError, TypeError) as exc:
                return Reply(error=f"malformed response: {_redact_exc(exc, self.log)}",
                             latency_s=latency, attempts=attempt, image_sizes=sizes)
        return Reply(error=last_error, attempts=self.max_retries + 1, image_sizes=sizes)

    def _sleep(self, attempt: int, retry_after: Optional[str]) -> None:
        delay = self.backoff * (2 ** (attempt - 1)) * (1 + 0.25 * random.random())
        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), 120.0))
            except ValueError:
                pass
        if delay > 0:
            self.log(f"retry {attempt} in {delay:.1f}s")
            time.sleep(delay)

    def generate(self, requests: list[Request]) -> list[Reply]:
        if self.workers == 1 or len(requests) == 1:
            return [self._one(r) for r in requests]
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._one, requests))


class VLLMBackend(Backend):
    """Offline ``vllm.LLM`` -- built on first use, imports ``vllm`` then."""

    name = "vllm"

    def __init__(self, model, prep, max_tokens, temperature, log, *, batch_size: int = 16,
                 tensor_parallel: int = 1, dtype: str = "auto",
                 max_model_len: Optional[int] = None, gpu_mem: float = 0.90,
                 max_images: int = 2, seed: Optional[int] = 0, trust_remote_code: bool = True,
                 engine_kwargs: Optional[dict] = None, mm_processor_kwargs: Optional[dict] = None):
        super().__init__(model, prep, max_tokens, temperature, log)
        self.batch_size = max(1, batch_size)
        self.opts = {"tensor_parallel_size": tensor_parallel, "dtype": dtype,
                     "gpu_memory_utilization": gpu_mem,
                     "limit_mm_per_prompt": {"image": max_images},
                     "trust_remote_code": trust_remote_code}
        if max_model_len:
            self.opts["max_model_len"] = max_model_len
        if mm_processor_kwargs:
            self.opts["mm_processor_kwargs"] = mm_processor_kwargs
        self.opts.update(engine_kwargs or {})
        self.seed = seed
        self._llm = None
        self._params = None

    def describe(self) -> dict:
        d = super().describe()
        d.update({"engine": self.opts, "seed": self.seed, "batch_size": self.batch_size})
        return d

    def _ensure(self):
        if self._llm is None:
            from vllm import LLM, SamplingParams  # lazy: never at import time

            self.log(f"loading {self.model} with vllm {self.opts}")
            self._llm = LLM(model=self.model, **self.opts)
            self._params = SamplingParams(temperature=self.temperature,
                                          max_tokens=self.max_tokens, seed=self.seed)
        return self._llm, self._params

    def generate(self, requests: list[Request]) -> list[Reply]:
        llm, params = self._ensure()
        convs, sizes = [], []
        for r in requests:
            messages, pics = openai_messages(r, self.prep)
            convs.append(messages)
            sizes.append([list(p.size) for p in pics])
        t0 = time.perf_counter()
        try:
            outs = llm.chat(convs, sampling_params=params, use_tqdm=False)
        except Exception as exc:  # noqa: BLE001 -- one bad batch must not end the run
            err = _redact_exc(exc, self.log)
            self.log(f"batch of {len(requests)} failed: {err}")
            return [Reply(error=err, image_sizes=s) for s in sizes]
        share = (time.perf_counter() - t0) / max(len(requests), 1)
        replies = []
        for o, s in zip(outs, sizes):
            gen = o.outputs[0]
            replies.append(Reply(text=gen.text, latency_s=share, image_sizes=s,
                                 prompt_tokens=len(o.prompt_token_ids or []),
                                 completion_tokens=len(gen.token_ids or []),
                                 model_snapshot=self.model, finish_reason=gen.finish_reason,
                                 extra={"batch_latency_s": round(share * len(requests), 3)}))
        return replies


class TransformersBackend(Backend):
    """HF ``AutoModelForImageTextToText`` + ``AutoProcessor`` (Qwen3-VL, InternVL ``*-hf``)."""

    name = "transformers"

    def __init__(self, model, prep, max_tokens, temperature, log, *, dtype: str = "auto",
                 device_map: str = "auto", trust_remote_code: bool = True,
                 attn_implementation: Optional[str] = None):
        super().__init__(model, prep, max_tokens, temperature, log)
        self.dtype, self.device_map = dtype, device_map
        self.trust_remote_code, self.attn = trust_remote_code, attn_implementation
        self._model = self._processor = None

    def describe(self) -> dict:
        d = super().describe()
        d.update({"dtype": self.dtype, "device_map": self.device_map, "attn": self.attn})
        return d

    def _ensure(self):
        if self._model is None:
            import torch  # lazy
            from transformers import AutoModelForImageTextToText, AutoProcessor  # lazy

            dtype = {"auto": "auto", "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                     "fp16": torch.float16, "float16": torch.float16,
                     "fp32": torch.float32, "float32": torch.float32}[self.dtype]
            self.log(f"loading {self.model} with transformers (dtype {self.dtype})")
            self._processor = AutoProcessor.from_pretrained(
                self.model, trust_remote_code=self.trust_remote_code)
            kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": self.device_map,
                                      "trust_remote_code": self.trust_remote_code}
            if self.attn:
                kwargs["attn_implementation"] = self.attn
            self._model = AutoModelForImageTextToText.from_pretrained(self.model, **kwargs)
            self._model.eval()
        return self._model, self._processor

    def _one(self, req: Request) -> Reply:
        import torch

        model, processor = self._ensure()
        pils, content, sizes = [], [], []
        for kind, value in req.parts:
            if kind == "text":
                content.append({"type": "text", "text": value})
            else:
                img = self.prep.pil(value)
                pils.append(img)
                sizes.append(list(img.size))
                content.append({"type": "image", "image": img})
        messages = []
        if req.system:
            messages.append({"role": "system", "content": [{"type": "text", "text": req.system}]})
        messages.append({"role": "user", "content": content})
        t0 = time.perf_counter()
        try:
            try:
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt")
            except Exception:  # noqa: BLE001 -- older processors: template, then tensors
                text = processor.apply_chat_template(messages, add_generation_prompt=True,
                                                     tokenize=False)
                inputs = processor(text=[text], images=pils or None, return_tensors="pt")
            inputs = inputs.to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
            n_in = inputs["input_ids"].shape[1]
            new = out[:, n_in:]
            text = processor.batch_decode(new, skip_special_tokens=True)[0]
            return Reply(text=text, latency_s=time.perf_counter() - t0, image_sizes=sizes,
                         prompt_tokens=int(n_in), completion_tokens=int(new.shape[1]),
                         model_snapshot=self.model,
                         finish_reason="length" if new.shape[1] >= self.max_tokens else "stop")
        except Exception as exc:  # noqa: BLE001
            return Reply(error=_redact_exc(exc, self.log), image_sizes=sizes,
                         latency_s=time.perf_counter() - t0)

    def generate(self, requests: list[Request]) -> list[Reply]:
        return [self._one(r) for r in requests]


def _import_tools():
    """``(vlm_io, vlm_check)`` from the pack's ``tools/``, ``scripts/`` or ``tda``."""
    for p in _SEARCH:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import vlm_io  # type: ignore[import-not-found]

    try:
        from tda.core.export import vlm_check  # type: ignore[import-not-found]
    except ImportError:
        try:
            import vlm_check  # type: ignore[import-not-found,no-redef]
        except ImportError:
            from vlmtools import vlm_check  # type: ignore[import-not-found,no-redef]
    return vlm_io, vlm_check


class GoldBackend(Backend):
    """The labels' own reply, through ``vlm_check.gold_reply`` -- the chain's self-check."""

    name = "gold"
    batch_size = 256

    def __init__(self, model, prep, max_tokens, temperature, log, *, truth: str):
        super().__init__(model or "gold", prep, max_tokens, temperature, log)
        io_mod, check = _import_tools()
        self.records = {str(r["id"]): r for r in io_mod.read_jsonl(truth)}
        self.worlds, _ = io_mod.load_oracle(truth)
        self.gold_reply = check.gold_reply
        self.truth = str(truth)

    def describe(self) -> dict:
        d = super().describe()
        d["truth"] = self.truth
        return d

    def generate(self, requests: list[Request]) -> list[Reply]:
        out = []
        for r in requests:
            rec = self.records.get(r.id)
            if rec is None:
                out.append(Reply(error=f"id {r.id} not in --truth"))
                continue
            sizes = []
            for ref in r.images:
                try:
                    sizes.append(list(self.prep.load(ref).size))
                except Exception as exc:  # noqa: BLE001
                    out.append(Reply(error=_redact_exc(exc, self.log)))
                    break
            else:
                t0 = time.perf_counter()
                world = self.worlds.get(int(rec["label"]["desktop"]))
                text = self.gold_reply(rec, world)
                out.append(Reply(text=text, latency_s=time.perf_counter() - t0,
                                 image_sizes=sizes, model_snapshot="gold"))
        return out


def _json_arg(text: Optional[str], name: str) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name}: not JSON ({exc})")
    if not isinstance(value, dict):
        raise SystemExit(f"{name}: must be a JSON object")
    return value


def build_backend(args, prep: ImagePrep, log: Log) -> Backend:
    common = dict(prep=prep, max_tokens=args.max_tokens, temperature=args.temperature, log=log)
    if args.backend == "openai":
        key = None
        if not args.no_api_key:
            if not args.api_key_env:
                raise SystemExit("--api-key-env <VAR> is required for the openai backend "
                                 "(or --no-api-key for a keyless local server)")
            key = os.environ.get(args.api_key_env)
            if not key:
                raise SystemExit(f"environment variable {args.api_key_env} is not set")
        if not args.model:
            raise SystemExit("--model is required")
        return OpenAIBackend(args.model, base_url=args.base_url, api_key=key,
                             timeout=args.timeout, max_retries=args.max_retries,
                             backoff=args.backoff, workers=args.workers, seed=args.seed,
                             extra_body=_json_arg(args.extra_body, "--extra-body"),
                             extra_headers=_json_arg(args.extra_headers, "--extra-headers"),
                             **common)
    if args.backend == "vllm":
        if not args.model:
            raise SystemExit("--model is required")
        return VLLMBackend(args.model, batch_size=args.batch_size,
                           tensor_parallel=args.tensor_parallel, dtype=args.dtype,
                           max_model_len=args.max_model_len, gpu_mem=args.gpu_mem,
                           max_images=args.max_images, seed=args.seed,
                           trust_remote_code=not args.no_trust_remote_code,
                           engine_kwargs=_json_arg(args.engine_kwargs, "--engine-kwargs"),
                           mm_processor_kwargs=_json_arg(args.mm_processor_kwargs,
                                                         "--mm-processor-kwargs"), **common)
    if args.backend == "transformers":
        if not args.model:
            raise SystemExit("--model is required")
        return TransformersBackend(args.model, dtype=args.dtype, device_map=args.device_map,
                                   trust_remote_code=not args.no_trust_remote_code,
                                   attn_implementation=args.attn, **common)
    if args.backend == "gold":
        if not args.truth:
            raise SystemExit("--truth <export.jsonl> is required for the gold backend")
        return GoldBackend(args.model, truth=args.truth, **common)
    raise SystemExit(f"unknown backend {args.backend!r}")


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def done_ids(out_path, keep_errors: bool) -> tuple[set[str], int]:
    """Ids already answered in ``out_path`` (``raw_text`` present), and how many error rows."""
    done: set[str] = set()
    errors = 0
    if not Path(out_path).exists():
        return done, errors
    for row in read_jsonl(out_path):
        if row.get("raw_text") is not None or keep_errors:
            done.add(str(row["id"]))
        else:
            errors += 1
    return done, errors


def select(requests: list[Request], ids_file: Optional[str], limit: Optional[int],
           skip: set[str]) -> list[Request]:
    keep = None
    if ids_file:
        keep = {ln.strip() for ln in Path(ids_file).read_text(encoding="utf-8").splitlines()
                if ln.strip()}
    out = [r for r in requests if (keep is None or r.id in keep) and r.id not in skip]
    return out[:limit] if limit else out


def row_for(req: Request, replies: list[Reply], info: dict, run_id: str) -> dict:
    """The output line: ``raw_text`` (= the first repeat), ``raw_text_i`` when repeated."""
    first = replies[0]
    meta = dict(info)
    meta.update({"run_id": run_id, "task": req.meta.get("task"),
                 "prompt_template_id": req.meta.get("prompt_template_id"),
                 "reference": req.meta.get("reference"), "images": len(req.images)})
    meta.update(first.meta())
    row: dict[str, Any] = {"id": req.id, "raw_text": first.text, "meta": meta}
    if len(replies) > 1:
        for i, rep in enumerate(replies, start=1):
            row[f"raw_text_{i}"] = rep.text
        meta["repeats"] = [rep.meta() for rep in replies]
        meta["error"] = "; ".join(f"{i}: {r.error}" for i, r in enumerate(replies, 1)
                                  if r.error) or None
    return row


def batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), max(1, size)):
        yield items[i:i + size]


def dry_run(requests: list[Request], todo: list[Request], prep: ImagePrep, args,
            log: Log) -> int:
    problems = []
    n_images = 0
    for r in todo:
        for ref in r.images:
            n_images += 1
            bad = prep.check(ref)
            if bad:
                problems.append(f"{r.id}: {bad}")
    print(f"[dry-run] {len(requests)} requests in {args.chat}, {len(todo)} to run, "
          f"{n_images} images; backend {args.backend}, model {args.model or '-'}, "
          f"max_tokens {args.max_tokens}, temperature {args.temperature}, "
          f"max_side {prep.max_side or 'as is'}, repeat {args.repeat}")
    if args.backend == "openai":
        status = ("not needed" if args.no_api_key else
                  "set" if args.api_key_env and os.environ.get(args.api_key_env) else
                  f"NOT SET ({args.api_key_env or '--api-key-env missing'})")
        print(f"[dry-run] base_url {args.base_url}; api key: {status}")
    for p in problems[:20]:
        print(f"[dry-run] problem: {p}")
    if len(problems) > 20:
        print(f"[dry-run] ... {len(problems) - 20} more problems")
    if todo:
        first = todo[0]
        print(f"\n--- first request {first.id} ({first.meta.get('task')}, "
              f"{first.meta.get('prompt_template_id')}) ---")
        if first.system:
            print(f"[system] {first.system}")
        for kind, value in first.parts:
            if kind == "text":
                print(f"[text]\n{value}")
            elif value.startswith("data:"):
                print(f"[image] <{value.split(';')[0][5:]} data URL, {len(value)} chars>")
            else:
                path = resolve_path(value, prep.root)
                size = ""
                if path.is_file():
                    try:
                        size = f" {prep.load(value).source_size[0]}x{prep.load(value).source_size[1]}"
                        if prep.max_side:
                            size += f" -> sent {prep.load(value).size[0]}x{prep.load(value).size[1]}"
                    except Exception as exc:  # noqa: BLE001
                        size = f" (unreadable: {exc})"
                print(f"[image] {path}{size}")
    if problems:
        log(f"dry-run: {len(problems)} problems")
        return 2
    return 0


def run(args, log: Log) -> int:
    chat = Path(args.chat)
    if not chat.is_file():
        raise SystemExit(f"--chat {chat}: no such file")
    requests = load_requests(chat)
    skip, n_errors = done_ids(args.out, args.keep_errors)
    todo = select(requests, args.ids, args.limit, skip)
    prep = ImagePrep(args.image_root, args.max_side, args.jpeg_quality)
    if skip:
        log(f"resume: {len(skip)} ids already in {args.out}"
            + (f", {n_errors} error rows will be re-run" if n_errors else ""))
    if args.dry_run:
        return dry_run(requests, todo, prep, args, log)
    if not todo:
        log("nothing to do")
        return 0
    backend = build_backend(args, prep, log)
    info = backend.describe()
    info["chat"] = str(chat)
    run_id = time.strftime("%Y%m%dT%H%M%S")
    log(f"{len(todo)} requests -> {args.out} ({json.dumps(info, default=str)})")
    failed = written = 0
    t_start = time.perf_counter()
    try:
        size = backend.batch_size if args.backend == "openai" else args.batch_size
        for batch in batched(todo, size):
            per_request: list[list[Reply]] = [[] for _ in batch]
            for _ in range(args.repeat):
                replies = backend.generate(batch)
                if len(replies) != len(batch):
                    raise RuntimeError(f"{backend.name} returned {len(replies)} replies "
                                       f"for {len(batch)} requests")
                for lst, rep in zip(per_request, replies):
                    lst.append(rep)
            rows = [row_for(req, reps, info, run_id) for req, reps in zip(batch, per_request)]
            written += append_jsonl(args.out, rows)
            failed += sum(1 for r in rows if r["meta"].get("error"))
            elapsed = time.perf_counter() - t_start
            log(f"{written}/{len(todo)} written, {failed} failed, {elapsed:.0f}s")
    finally:
        backend.close()
    log(f"done: {written} rows, {failed} with errors -> {args.out}")
    return 3 if failed else 0


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--chat", required=True, help="chat JSONL from vlm_to_chat.py (any format)")
    ap.add_argument("--out", required=True, help="prediction JSONL (appended; resume by id)")
    ap.add_argument("--model", default=None, help="model id (API name, or HF id / local path)")
    ap.add_argument("--image-root", default=None,
                    help="directory relative image paths in the chat file are under")
    ap.add_argument("--max-side", type=int, default=None,
                    help="downsize images so the longer side is at most this (pixels)")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed where supported")
    ap.add_argument("--repeat", type=int, default=1,
                    help="ask every request k times (raw_text_1..k; the repeat subset)")
    ap.add_argument("--limit", type=int, default=None, help="run at most n requests")
    ap.add_argument("--ids", default=None, help="a file of ids to run (one per line)")
    ap.add_argument("--keep-errors", action="store_true",
                    help="on resume, do not re-run rows whose raw_text is null")
    ap.add_argument("--batch-size", type=int, default=16, help="vllm / gold batch")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate inputs and print the first prompt; load no model")
    ap.add_argument("--log", default=None, help="append progress lines to this file")
    ap.add_argument("--quiet", action="store_true")
    o = ap.add_argument_group("openai")
    o.add_argument("--base-url", default="https://api.openai.com/v1")
    o.add_argument("--api-key-env", default=None,
                   help="NAME of the environment variable holding the key (never logged)")
    o.add_argument("--no-api-key", action="store_true", help="keyless server (vllm serve)")
    o.add_argument("--timeout", type=float, default=120.0)
    o.add_argument("--max-retries", type=int, default=5)
    o.add_argument("--backoff", type=float, default=2.0, help="first retry delay, seconds")
    o.add_argument("--workers", type=int, default=1, help="parallel requests")
    o.add_argument("--extra-body", default=None, help="JSON merged into the request body")
    o.add_argument("--extra-headers", default=None, help="JSON of extra HTTP headers")
    v = ap.add_argument_group("vllm / transformers")
    v.add_argument("--tensor-parallel", type=int, default=1)
    v.add_argument("--dtype", default="auto", help="auto | bf16 | fp16 | fp32")
    v.add_argument("--max-model-len", type=int, default=None)
    v.add_argument("--gpu-mem", type=float, default=0.90, help="gpu_memory_utilization")
    v.add_argument("--max-images", type=int, default=2, help="limit_mm_per_prompt images")
    v.add_argument("--engine-kwargs", default=None, help="JSON of extra vllm.LLM kwargs")
    v.add_argument("--mm-processor-kwargs", default=None,
                   help='JSON, e.g. {"max_pixels": 1003520} for Qwen-VL')
    v.add_argument("--no-trust-remote-code", action="store_true")
    v.add_argument("--device-map", default="auto")
    v.add_argument("--attn", default=None, help="attn_implementation (transformers)")
    g = ap.add_argument_group("gold")
    g.add_argument("--truth", default=None, help="the export-vlm JSONL the chat was made from")
    return ap


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    log = Log(args.log, args.quiet)
    try:
        return run(args, log)
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
