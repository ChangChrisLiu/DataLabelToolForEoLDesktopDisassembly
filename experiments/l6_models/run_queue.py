"""Run one of L6's GPU queues: every job a subprocess, interrupted jobs resumed after the app closes.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.run_queue det
    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l6_models.run_queue sam

A job that exits with code 3 stopped because the annotator's app was open (or
STOP exists): the queue waits until no ``*tda.cli*app*`` python process is
left (``env.wait_for_gpu``, polling every 5 minutes) and starts the job again;
detector training resumes from ``last.ckpt`` (``--resume``), SAM training and
the evaluations from their own saved state.  STOP ends the queue.  Every start,
end and wait goes to ``logs/queue_<name>.jsonl``.
"""
from __future__ import annotations

from experiments.l6_models import env  # noqa: F401  (side effects first)

import json
import subprocess
import sys
import time

PY = str(env.TRAIN_ENV_PY)
APP_PY = str(env.APP_ENV_PY)


def _m(module: str, *args: str, py: str = PY) -> list[str]:
    return [py, "-m", f"experiments.l6_models.{module}", *args]


QUEUES = {
    "det": [
        ("train v1h", _m("train_det", "--set", "v1h"), "det"),
        ("train v2h", _m("train_det", "--set", "v2h"), "det"),
        ("train F", _m("train_det", "--set", "F"), "det"),
        ("infer v2", _m("infer_det", "--model", "v2"), None),
        ("infer v2h", _m("infer_det", "--model", "v2h"), None),
        ("infer F", _m("infer_det", "--model", "F"), None),
        ("eval det", _m("eval_det"), None),
        ("train v3", _m("train_det", "--set", "v3"), "det"),
    ],
    "sam": [
        ("SAM train F", _m("train_sam", "--fold", "F"), None),
        ("SAM eval F", _m("eval_sam", "--out", "heldout"), None),
        ("SAM gate", _m("gate_sam"), None),
        ("SAM train v3", _m("train_sam", "--fold", "v3"), None),
    ],
}


def log(name: str, rec: dict) -> None:
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (env.ROOT / "logs" / f"queue_{name}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[queue {name}] {json.dumps(rec)}", flush=True)


def run(name: str, only: list[str] | None = None) -> int:
    for label, cmd, kind in QUEUES[name]:
        if only and label not in only:
            continue
        resume = False
        attempt = 0
        while True:
            if env.stop_requested():
                log(name, {"event": "STOP", "job": label})
                return 0
            env.wait_for_gpu(f"{name}: {label}")
            attempt += 1
            full = cmd + (["--resume"] if resume and kind == "det" else [])
            safe = label.replace(" ", "_")
            out = env.ROOT / "logs" / f"{safe}_{attempt}.log"
            log(name, {"event": "start", "job": label, "attempt": attempt, "cmd": full[2:]})
            t0 = time.time()
            with open(out, "w", encoding="utf-8") as fh:
                rc = subprocess.call(full, cwd=str(env.REPO), stdout=fh, stderr=subprocess.STDOUT)
            log(name, {"event": "end", "job": label, "attempt": attempt, "rc": rc,
                       "seconds": round(time.time() - t0)})
            if rc == 0:
                break
            if rc == 3:
                resume = True
                continue
            log(name, {"event": "failed", "job": label, "rc": rc, "log": str(out)})
            return rc
    log(name, {"event": "done"})
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(run(args[0], args[1:] or None))
