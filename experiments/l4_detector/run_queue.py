"""Run L4 jobs one after another: one GPU job at a time, BELOW_NORMAL, the stop file honoured.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l4_detector.run_queue \\
        "infer --model production" "evaluate --model production" ...

A job is ``"<module> <args...>"`` (a module of this package).  Its output goes
to ``logs/<module>_<args>.log``.  Before each job the controller's stop file
(``D:/DataSet/.cache/tmp/l4/STOP``) is checked: when it exists the queue ends
there (a training job checks it itself between epochs).  A failed job is
reported and the queue goes on.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l4_detector import env  # noqa: E402

PY = r"D:\Anaconda\envs\tda_l3\python.exe"
LOGS = env.OUT / "logs"
BELOW_NORMAL = 0x00004000


def main(argv=None) -> int:
    jobs = list(argv if argv is not None else sys.argv[1:])
    LOGS.mkdir(parents=True, exist_ok=True)
    status = []
    for job in jobs:
        if env.stop_requested():
            print(f"[queue] stop file present: not starting {job!r} or anything after it", flush=True)
            status.append((job, "skipped", 0))
            continue
        parts = job.split()
        cmd = [PY, "-m", f"experiments.l4_detector.{parts[0]}", *parts[1:]]
        tag = re.sub(r"[^A-Za-z0-9_.,=-]+", "_", "_".join(parts)).strip("_")
        t0 = time.perf_counter()
        with (LOGS / f"{tag}.log").open("w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(env.REPO),
                                 creationflags=BELOW_NORMAL if sys.platform == "win32" else 0)
        dt = time.perf_counter() - t0
        status.append((job, rc, dt))
        print(f"[queue] {time.strftime('%H:%M:%S')} {job}: rc={rc} {dt:.0f} s", flush=True)
    bad = [j for j, rc, _ in status if rc not in (0,)]
    print(f"[queue] done; not ok: {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
