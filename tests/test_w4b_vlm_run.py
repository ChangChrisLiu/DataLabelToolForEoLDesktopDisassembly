"""The W4-B inference driver: ``scripts/vlm_run/vlm_run.py`` and ``vlm_collect.py``.

All on the :mod:`tests.w3b_scene` export with the ``gold`` pseudo backend and a
fake OpenAI-compatible server on the loopback -- no model, no network, no
``vllm`` / ``transformers`` import.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import w3b_scene as W
from tda.core.db import Db
from tda.core.export.vlm import export_vlm

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RUN_DIR = SCRIPTS / "vlm_run"
for p in (SCRIPTS, RUN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import vlm_collect  # noqa: E402
import vlm_io as IO  # noqa: E402
import vlm_run  # noqa: E402
import vlm_score  # noqa: E402
import vlm_to_chat  # noqa: E402

FORMATS = ("openai", "qwen", "internvl")


@pytest.fixture(scope="module")
def pack(tmp_path_factory) -> dict:
    """The scanner export of the scene, its tiny image cache, and a chat file per format."""
    tmp = tmp_path_factory.mktemp("w4b")
    db = Db(str(tmp / "s.sqlite"))
    tax = W.build(db)
    export = tmp / "vlm_scan.jsonl"
    export_vlm(db, tax, [W.DESKTOP], W.VIEW, str(export))
    db.close()
    from PIL import Image

    root = tmp / "images"
    for entry in IO.load_manifest(export).values():
        path = root / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (entry["step"] * 20 % 255, 80, 120)).save(path)
    chats = {}
    for fmt in FORMATS:
        out = tmp / f"chat_{fmt}.jsonl"
        assert vlm_to_chat.main(["--export", str(export), "--format", fmt,
                                 "--image-root", str(root), "--out", str(out)]) == 0
        chats[fmt] = out
    # a relative-path chat file, the way a pack converted on another machine is
    rel = tmp / "chat_relative.jsonl"
    rows = []
    for row in IO.read_jsonl(chats["openai"]):
        for part in row["messages"][1]["content"]:
            if part["type"] == "image_url":
                url = Path(part["image_url"]["url"])
                part["image_url"]["url"] = url.relative_to(root).as_posix()
        rows.append(row)
    IO.write_jsonl(rel, rows)
    chats["relative"] = rel
    return {"export": export, "root": root, "chats": chats,
            "records": list(IO.read_jsonl(export))}


def _rows(path) -> list[dict]:
    return list(IO.read_jsonl(path))


def _gold(pack, chat, out, *extra) -> int:
    return vlm_run.main(["--backend", "gold", "--truth", str(pack["export"]),
                         "--chat", str(chat), "--out", str(out), "--quiet", *extra])


# --------------------------------------------------------------------------- #
# gold end to end
# --------------------------------------------------------------------------- #
def test_the_gold_backend_scores_one_end_to_end(pack, tmp_path):
    out = tmp_path / "pred.jsonl"
    assert _gold(pack, pack["chats"]["openai"], out) == 0
    rows = _rows(out)
    assert [r["id"] for r in rows] == [r["id"] for r in pack["records"]]
    for row in rows:
        meta = row["meta"]
        assert row["raw_text"] and meta["error"] is None
        assert meta["backend"] == "gold" and meta["model_snapshot"] == "gold"
        assert meta["latency_s"] is not None and meta["task"] and meta["prompt_template_id"]
        assert meta["image_sizes_sent"] == [[64, 64]] * meta["images"]
    result = vlm_score.score(pack["export"], out, pack["chats"]["openai"], resamples=50)
    assert result["predicted"] == result["records"] == len(rows)
    for mode in ("strict", "lenient"):
        assert result[mode]["parse_failure_rate_all"] == 0.0
        for task, cell in result[mode]["by_task"].items():
            assert cell["mean"] == 1.0, (mode, task, cell)


@pytest.mark.parametrize("fmt", ("qwen", "internvl", "relative"))
def test_every_chat_format_normalises_to_the_same_requests(pack, fmt, tmp_path):
    ref = {r.id: r for r in vlm_run.load_requests(pack["chats"]["openai"])}
    other = vlm_run.load_requests(pack["chats"][fmt])
    assert [r.id for r in other] == list(ref)
    for r in other:
        assert r.text.strip() == ref[r.id].text.strip()
        assert len(r.images) == len(ref[r.id].images)
        assert "<image>" not in r.text
        if fmt != "internvl":
            assert r.system == ref[r.id].system
    # and the gold run over it is the same reply set
    out = tmp_path / f"pred_{fmt}.jsonl"
    root = [] if fmt != "relative" else ["--image-root", str(pack["root"])]
    assert _gold(pack, pack["chats"][fmt], out, *root) == 0
    first = {r["id"]: r["raw_text"] for r in _rows(out)}
    gold = tmp_path / "pred_openai.jsonl"
    assert _gold(pack, pack["chats"]["openai"], gold) == 0
    assert first == {r["id"]: r["raw_text"] for r in _rows(gold)}


def test_a_relative_chat_file_needs_the_image_root(pack, tmp_path):
    out = tmp_path / "pred.jsonl"
    assert _gold(pack, pack["chats"]["relative"], out, "--limit", "2") == 3
    rows = _rows(out)
    assert len(rows) == 2
    assert all(r["raw_text"] is None and "FileNotFoundError" in r["meta"]["error"]
               for r in rows)


# --------------------------------------------------------------------------- #
# resume, repeat
# --------------------------------------------------------------------------- #
def test_resume_skips_done_ids_and_reruns_errors(pack, tmp_path, capsys):
    out = tmp_path / "pred.jsonl"
    assert _gold(pack, pack["chats"]["openai"], out, "--limit", "5") == 0
    assert len(_rows(out)) == 5
    # a failed row for the sixth id: the resume re-runs it, keeps the five
    sixth = pack["records"][5]["id"]
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": sixth, "raw_text": None, "meta": {"error": "boom"}}) + "\n")
    assert vlm_run.main(["--backend", "gold", "--truth", str(pack["export"]),
                         "--chat", str(pack["chats"]["openai"]), "--out", str(out)]) == 0
    err = capsys.readouterr().err
    assert "resume: 5 ids already" in err and "1 error rows will be re-run" in err
    rows = _rows(out)
    ids = [r["id"] for r in rows]
    assert ids.count(sixth) == 2  # the failed row stays, the answer is appended
    assert len(set(ids)) == len(pack["records"])
    answered = {r["id"]: r["raw_text"] for r in rows if r["raw_text"] is not None}
    assert len(answered) == len(pack["records"])
    # the scorer keeps the last row of an id: the sixth is answered now
    result = vlm_score.score(pack["export"], out, resamples=20)
    assert result["predicted"] == result["records"]
    # nothing left to do, and --keep-errors would also have kept the failed row
    assert vlm_run.main(["--backend", "gold", "--truth", str(pack["export"]), "--chat",
                         str(pack["chats"]["openai"]), "--out", str(out), "--quiet"]) == 0
    assert len(_rows(out)) == len(rows)
    done, errors = vlm_run.done_ids(out, keep_errors=True)
    assert errors == 0 and len(done) == len(pack["records"])


def test_repeat_writes_raw_text_1_to_k_and_split_scores_each(pack, tmp_path):
    out = tmp_path / "pred.jsonl"
    assert _gold(pack, pack["chats"]["openai"], out, "--repeat", "3", "--limit", "4") == 0
    rows = _rows(out)
    assert len(rows) == 4
    for row in rows:
        assert row["raw_text"] == row["raw_text_1"] == row["raw_text_2"] == row["raw_text_3"]
        assert "raw_text_4" not in row
        assert len(row["meta"]["repeats"]) == 3
        assert all(m["error"] is None for m in row["meta"]["repeats"])
    parts = vlm_collect.split(out)
    assert [p.name for p in parts] == ["pred.r1.jsonl", "pred.r2.jsonl", "pred.r3.jsonl"]
    for i, p in enumerate(parts, start=1):
        for row in _rows(p):
            assert row["raw_text"] and row["meta"]["repeat"] == i
    result = vlm_score.score(pack["export"], parts[2], resamples=20)
    assert result["predicted"] == 4
    with pytest.raises(SystemExit):
        vlm_collect.split(parts[0])  # not a repeat file


# --------------------------------------------------------------------------- #
# dry run: nothing heavy imported
# --------------------------------------------------------------------------- #
_PROBE = """
import runpy, sys
sys.argv = [{script!r}] + {argv!r}
code = 0
try:
    runpy.run_path({script!r}, run_name="__main__")
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else 1
print("HEAVY", [m for m in ("vllm", "transformers", "torch") if m in sys.modules])
print("EXIT", code)
"""


def _probe(argv: list[str]) -> subprocess.CompletedProcess:
    code = _PROBE.format(script=str(RUN_DIR / "vlm_run.py"), argv=argv)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          encoding="utf-8", timeout=120)


@pytest.mark.parametrize("backend", ("vllm", "transformers", "openai"))
def test_dry_run_prints_the_first_prompt_without_importing_a_model_stack(pack, tmp_path, backend):
    extra = ["--api-key-env", "W4B_NO_SUCH_KEY"] if backend == "openai" else []
    proc = _probe(["--backend", backend, "--model", "some/model", "--chat",
                   str(pack["chats"]["qwen"]), "--out", str(tmp_path / "p.jsonl"),
                   "--dry-run", "--max-side", "32", *extra])
    assert proc.returncode == 0, proc.stderr
    assert "HEAVY []" in proc.stdout and "EXIT 0" in proc.stdout
    first = pack["records"][0]
    assert first["prompt"]["question"] in proc.stdout
    assert f"{len(pack['records'])} requests" in proc.stdout
    assert "64x64 -> sent 32x32" in proc.stdout
    if backend == "openai":
        assert "NOT SET" in proc.stdout
    assert not (tmp_path / "p.jsonl").exists()


def test_dry_run_reports_missing_images(pack, tmp_path):
    proc = _probe(["--backend", "vllm", "--model", "m", "--chat", str(pack["chats"]["relative"]),
                   "--out", str(tmp_path / "p.jsonl"), "--dry-run"])
    assert "EXIT 2" in proc.stdout and "problem:" in proc.stdout and "HEAVY []" in proc.stdout


def test_importing_the_driver_imports_nothing_heavy():
    proc = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(RUN_DIR)!r}); import vlm_run; "
         "print([m for m in ('vllm', 'transformers', 'torch', 'tda') if m in sys.modules])"],
        capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


# --------------------------------------------------------------------------- #
# the openai backend against a fake server
# --------------------------------------------------------------------------- #
class _Fake(BaseHTTPRequestHandler):
    good_key = "good-key"
    fail_first = 0
    received: list[dict] = []
    lock = threading.Lock()

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        auth = self.headers.get("Authorization")
        with _Fake.lock:
            _Fake.received.append({"path": self.path, "auth": auth, "body": body})
            fail = _Fake.fail_first > 0
            if fail:
                _Fake.fail_first -= 1
        if auth != f"Bearer {_Fake.good_key}":
            key = (auth or "").replace("Bearer ", "")
            # a careless server echoes the key back: the driver must not repeat it
            self._send(401, {"error": {"message": f"invalid api key: {key}"}})
            return
        if fail:
            self.send_response(503)
            self.send_header("Retry-After", "0")
            self.end_headers()
            return
        self._send(200, {"id": "cmpl-1", "model": "fake-vlm-2026-01-01",
                         "system_fingerprint": "fp_test",
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant",
                                                  "content": '{"changed": true}'}}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    def _send(self, status: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    _Fake.received = []
    _Fake.fail_first = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()
    httpd.server_close()


def _openai(pack, server, out, key_env, *extra) -> int:
    return vlm_run.main(["--backend", "openai", "--base-url", server, "--model", "fake-vlm",
                         "--api-key-env", key_env, "--chat", str(pack["chats"]["openai"]),
                         "--out", str(out), "--backoff", "0", *extra])


def test_the_openai_backend_sends_temperature_zero_and_inline_images(pack, server, tmp_path,
                                                                     monkeypatch):
    monkeypatch.setenv("W4B_KEY", _Fake.good_key)
    out = tmp_path / "pred.jsonl"
    assert _openai(pack, server, out, "W4B_KEY", "--limit", "3", "--max-tokens", "77",
                   "--workers", "2", "--quiet") == 0
    rows = _rows(out)
    assert [r["raw_text"] for r in rows] == ['{"changed": true}'] * 3
    for row in rows:
        meta = row["meta"]
        assert meta["model"] == "fake-vlm" and meta["model_snapshot"] == "fake-vlm-2026-01-01"
        assert meta["prompt_tokens"] == 10 and meta["completion_tokens"] == 5
        assert meta["finish_reason"] == "stop" and meta["system_fingerprint"] == "fp_test"
        assert meta["attempts"] == 1 and meta["error"] is None
        assert meta["api_key"] == "set" and meta["temperature"] == 0.0
    assert len(_Fake.received) == 3
    for req in _Fake.received:
        assert req["path"] == "/v1/chat/completions"
        assert req["auth"] == f"Bearer {_Fake.good_key}"
        body = req["body"]
        assert body["model"] == "fake-vlm" and body["temperature"] == 0 and body["max_tokens"] == 77
        assert body["messages"][0]["role"] == "system"
        parts = body["messages"][1]["content"]
        assert parts[0]["type"] == "text" and "Answer with JSON only" in parts[0]["text"]
        for p in parts[1:]:
            assert p["type"] == "image_url"
            assert p["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_bad_key_never_appears_in_output_log_or_rows(pack, server, tmp_path, monkeypatch,
                                                       capsys):
    secret = "sk-very-secret-9f8e7d6c"
    monkeypatch.setenv("W4B_BAD", secret)
    out = tmp_path / "pred.jsonl"
    log = tmp_path / "run.log"
    rc = _openai(pack, server, out, "W4B_BAD", "--limit", "2", "--log", str(log))
    assert rc == 3  # the requests failed, their rows are written with the error
    rows = _rows(out)
    assert len(rows) == 2 and all(r["raw_text"] is None for r in rows)
    assert all("HTTP 401" in r["meta"]["error"] for r in rows)
    captured = capsys.readouterr()
    for text in (captured.out, captured.err, log.read_text(encoding="utf-8"),
                 out.read_text(encoding="utf-8")):
        assert secret not in text
    assert "***" in rows[0]["meta"]["error"]  # the server's echo was redacted
    assert len(_Fake.received) == 2  # 401 is not retried


def test_retries_with_backoff_until_the_server_answers(pack, server, tmp_path, monkeypatch):
    monkeypatch.setenv("W4B_KEY", _Fake.good_key)
    _Fake.fail_first = 2
    out = tmp_path / "pred.jsonl"
    assert _openai(pack, server, out, "W4B_KEY", "--limit", "1", "--max-retries", "3",
                   "--quiet") == 0
    row = _rows(out)[0]
    assert row["raw_text"] == '{"changed": true}' and row["meta"]["attempts"] == 3
    assert len(_Fake.received) == 3
    # and gives up after --max-retries
    _Fake.fail_first = 5
    out2 = tmp_path / "pred2.jsonl"
    assert _openai(pack, server, out2, "W4B_KEY", "--limit", "1", "--max-retries", "1",
                   "--quiet") == 3
    row = _rows(out2)[0]
    assert row["raw_text"] is None and "HTTP 503" in row["meta"]["error"]
    assert row["meta"]["attempts"] == 2


def test_a_missing_key_variable_is_a_usage_error_naming_the_variable(pack, tmp_path, monkeypatch):
    monkeypatch.delenv("W4B_ABSENT", raising=False)
    with pytest.raises(SystemExit, match="W4B_ABSENT is not set"):
        _openai(pack, "http://127.0.0.1:9/v1", tmp_path / "p.jsonl", "W4B_ABSENT", "--quiet")
    with pytest.raises(SystemExit, match="--api-key-env"):
        vlm_run.main(["--backend", "openai", "--base-url", "http://127.0.0.1:9/v1", "--model",
                      "m", "--chat", str(pack["chats"]["openai"]),
                      "--out", str(tmp_path / "p.jsonl"), "--quiet"])
    assert not (tmp_path / "p.jsonl").exists()


def test_repeat_asks_the_server_k_times(pack, server, tmp_path, monkeypatch):
    monkeypatch.setenv("W4B_KEY", _Fake.good_key)
    out = tmp_path / "pred.jsonl"
    assert _openai(pack, server, out, "W4B_KEY", "--limit", "2", "--repeat", "3",
                   "--quiet") == 0
    assert len(_Fake.received) == 6
    for row in _rows(out):
        assert row["raw_text_3"] == row["raw_text"]
        assert [m["prompt_tokens"] for m in row["meta"]["repeats"]] == [10, 10, 10]


# --------------------------------------------------------------------------- #
# vlm_collect: subset, blind, table
# --------------------------------------------------------------------------- #
def test_subset_carries_the_sidecars_and_scores_as_a_truth_file(pack, tmp_path):
    ids = [r["id"] for r in pack["records"][:7]]
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
    sub = tmp_path / "truth" / "scan.jsonl"
    assert vlm_collect.main(["subset", "--export", str(pack["export"]), "--out", str(sub),
                             "--ids", str(ids_file)]) == 0
    assert [r["id"] for r in _rows(sub)] == ids
    for suffix in vlm_collect.SIDECARS:
        assert (tmp_path / "truth" / f"scan.{suffix}").exists()
    chat = tmp_path / "chat.jsonl"
    assert vlm_to_chat.main(["--export", str(sub), "--format", "openai", "--image-root",
                             str(pack["root"]), "--out", str(chat)]) == 0
    out = tmp_path / "pred.jsonl"
    assert vlm_run.main(["--backend", "gold", "--truth", str(sub), "--chat", str(chat),
                         "--out", str(out), "--quiet"]) == 0
    result = vlm_score.score(sub, out, chat, resamples=20)
    assert result["records"] == result["predicted"] == 7
    assert all(c["mean"] == 1.0 for c in result["strict"]["by_task"].values())
    bad = tmp_path / "bad.txt"
    bad.write_text("nope\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not in"):
        vlm_collect.subset(pack["export"], tmp_path / "x.jsonl", ids_file=str(bad))


def test_the_blind_variant_has_no_images_and_the_gold_still_scores_one(pack, tmp_path):
    blind = tmp_path / "blind.jsonl"
    assert vlm_collect.main(["blind", "--records", str(pack["export"]), "--out", str(blind)]) == 0
    rows = _rows(blind)
    assert len(rows) == len(pack["records"])
    assert all(r["prompt"]["images"] == [] and r["label"]["reference"] == "blind" for r in rows)
    chat = tmp_path / "chat.jsonl"
    assert vlm_to_chat.main(["--export", str(pack["export"]), "--records", str(blind),
                             "--format", "openai", "--out", str(chat)]) == 0
    reqs = vlm_run.load_requests(chat)
    assert all(not r.images for r in reqs)
    out = tmp_path / "pred.jsonl"
    assert _gold(pack, chat, out) == 0
    assert all(r["meta"]["images"] == 0 for r in _rows(out))
    result = vlm_score.score(pack["export"], out, resamples=20)
    assert all(c["mean"] == 1.0 for c in result["strict"]["by_task"].values())


def _score_json(path: Path, means: dict, parse_fail: float = 0.0, records: int = 100):
    by_task = {t: {"n": 10, "mean": m, "ci95": [m - 0.05, min(1.0, m + 0.05)], "desktops": 3}
               for t, m in means.items()}
    data = {"records": records, "predicted": records,
            "strict": {"parse_failure_rate_all": parse_fail, "by_task": by_task},
            "lenient": {"parse_failure_rate_all": parse_fail / 2, "by_task": by_task}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_the_table_folds_repeats_into_mean_and_std(tmp_path):
    runs = tmp_path / "runs"
    _score_json(runs / "m1" / "image" / "scan" / "score" / "m1.r1.score.json",
                {"V12": 0.8, "V14": 0.6}, 0.1)
    _score_json(runs / "m1" / "image" / "scan" / "score" / "m1.r2.score.json",
                {"V12": 0.9, "V14": 0.6}, 0.0)
    _score_json(runs / "m1" / "image" / "scan" / "score" / "m1.score.json",
                {"V12": 0.8, "V14": 0.6}, 0.1)
    _score_json(runs / "m2" / "blind" / "oak1" / "score" / "m2.score.json", {"V12": 0.5}, 0.2)
    md = tmp_path / "score.md"
    js = tmp_path / "score.json"
    assert vlm_collect.main(["table", "--scores", str(runs / "*" / "*" / "*" / "score" /
                                                      "*.score.json"),
                             "--out", str(md), "--json", str(js)]) == 0
    table = json.loads(js.read_text(encoding="utf-8"))
    assert table["tasks"] == ["V12", "V14"]
    rows = {(r["run"], r["name"]): r for r in table["rows"]}
    rep = rows[("m1/image/scan", "m1")]
    assert rep["repeats"] == 2  # r1 + r2 fold into one row ...
    assert rep["by_task"]["V12"] == {"n": 10, "mean": 0.85, "repeats": 2,
                                     "std": round(((0.05 ** 2) * 2) ** 0.5, 4)}
    assert rep["by_task"]["V14"]["std"] == 0.0
    assert rep["parse_failure_strict"] == 0.05
    single = rows[("m2/blind/oak1", "m2")]
    assert single["by_task"]["V12"]["ci95"] == [0.45, 0.55] and "std" not in single["by_task"]["V12"]
    text = md.read_text(encoding="utf-8")
    assert "0.850 ± 0.071 (k=2)" in text and "0.500 [0.450, 0.550]" in text
    assert "| m1/image/scan | m1 |" in text and "20.0% / 10.0%" in text


def _bash() -> str | None:
    """A working bash: Git's on Windows (the WSL stub in System32 fails without a distro)."""
    candidates = []
    git = shutil.which("git")
    if git:
        candidates.append(str(Path(git).resolve().parents[1] / "bin" / "bash.exe"))
        candidates.append(str(Path(git).resolve().parents[1] / "usr" / "bin" / "bash.exe"))
    if shutil.which("bash"):
        candidates.append(shutil.which("bash"))
    for cand in candidates:
        try:
            proc = subprocess.run([cand, "-c", "echo ok"], capture_output=True, text=True,
                                  timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout.strip() == "ok":
            return cand
    return None


def test_the_shell_templates_are_lf_and_parse():
    scripts = list(RUN_DIR.glob("*.sh")) + list((RUN_DIR / "slurm").glob("*"))
    assert len(scripts) >= 6
    for script in scripts:
        assert b"\r\n" not in script.read_bytes(), f"{script.name} has CRLF line endings"
    bash = _bash()
    if bash is None:
        pytest.skip("no working bash on this machine")
    for script in scripts:
        proc = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{script.name}: {proc.stderr}"
