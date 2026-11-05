"""Prove a VLM pack stands alone: its self-check in a fresh virtual environment (W4-A).

``python scripts/vlmpack_cleanvenv_check.py --pack <vlmpack dir>
[--venv D:/DataSet/.cache/tmp/w4a/venv] [--python <interpreter to make it from>]
[--wheelhouse <dir of wheels> | --yaml-from <site-packages>] [--out <dir>] [--fresh]``

1. ``python -m venv`` at ``--venv`` (``--fresh`` recreates it). With
   ``--wheelhouse`` the venv gets pip and ``pip install --no-index
   --find-links <wheelhouse> -r tools/requirements.txt``; **without one --
   the offline case -- the venv is made ``--without-pip`` and the pure-Python
   ``yaml`` package is copied from ``--yaml-from`` (default: the ``yaml`` the
   interpreter that runs this script imports) into its ``site-packages``**. That
   is the offline stand-in for ``pip install PyYAML``; the C accelerator
   (``_yaml``) is left out on purpose and PyYAML runs without it. Nothing is
   downloaded either way.
2. In that venv, from a neutral working directory, ``import tda`` must **fail**
   (the venv has no annotation tool), then ``tools/vlm_selfcheck.py`` runs with
   ``python -I`` on every ``eval_v3/<unit>.jsonl`` and must exit 0, and the
   ``tda`` it imported must be the pack's ``tools/tda``.

Exit 0 when all of it holds; 1 otherwise, saying what failed. ``<out>``
(default ``<venv>/../venv_check``) receives the self-check's files and
``cleanvenv_check.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

DEFAULT_VENV = "D:/DataSet/.cache/tmp/w4a/venv"


class CheckFailed(RuntimeError):
    pass


def _env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX")}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def venv_python(venv: Path) -> Path:
    win = venv / "Scripts" / "python.exe"
    return win if win.exists() or os.name == "nt" else venv / "bin" / "python"


def site_packages(python: Path) -> Path:
    out = subprocess.run([str(python), "-I", "-c",
                          "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                         capture_output=True, text=True, env=_env(), check=True)
    return Path(out.stdout.strip())


def make_venv(venv: Path, python: str, *, with_pip: bool, fresh: bool, log=print) -> Path:
    if fresh and venv.exists():
        shutil.rmtree(venv)
    if not venv_python(venv).exists():
        cmd = [python, "-m", "venv"] + ([] if with_pip else ["--without-pip"]) + [str(venv)]
        log(f"[cleanvenv] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=_env())
    return venv_python(venv)


def install_yaml(py: Path, yaml_from: Optional[str], log=print) -> str:
    """Copy the pure-Python ``yaml`` package into the venv; returns where from."""
    if yaml_from:
        src = Path(yaml_from) / "yaml"
    else:
        import yaml as host_yaml

        src = Path(host_yaml.__file__).resolve().parent
    if not (src / "__init__.py").is_file():
        raise CheckFailed(f"no yaml package at {src}")
    dest = site_packages(py) / "yaml"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyd", "*.so"))
    log(f"[cleanvenv] offline: copied the pure-Python yaml package from {src} "
        f"(the stand-in for `pip install PyYAML`; no _yaml accelerator)")
    return str(src)


def install_requirements(py: Path, wheelhouse: str, requirements: Path, log=print) -> None:
    cmd = [str(py), "-m", "pip", "install", "--no-index", "--find-links", wheelhouse,
           "-r", str(requirements)]
    log(f"[cleanvenv] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=_env())


def run_check(pack: Path, py: Path, out: Path, log=print) -> dict:
    tools = pack / "tools"
    neutral = out
    neutral.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run([str(py), "-I", "-c", "import tda"], capture_output=True, text=True,
                           cwd=str(neutral), env=_env())
    if probe.returncode == 0:
        raise CheckFailed("the virtual environment can import tda on its own: not clean")
    log("[cleanvenv] `import tda` fails in the venv, as it must")
    truth = sorted((pack / "eval_v3").glob("*.jsonl"))
    truth = [p for p in truth if not p.name.endswith((".images_manifest.jsonl", ".oracle.jsonl", ".masks.jsonl"))]
    if not truth:
        raise CheckFailed(f"no eval_v3/<unit>.jsonl under {pack}")
    cmd = [str(py), "-I", "-B", str(tools / "vlm_selfcheck.py"), "--truth", *map(str, truth),
           "--out", str(out / "selfcheck")]
    log(f"[cleanvenv] {' '.join(cmd[:4])} ... ({len(truth)} truth files)")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(neutral), env=_env())
    for line in proc.stdout.splitlines():
        log("  " + line)
    if proc.returncode != 0:
        raise CheckFailed(f"the self-check exited {proc.returncode}:\n{proc.stderr}")
    report = json.loads((out / "selfcheck" / "selfcheck.json").read_text(encoding="utf-8"))
    where = Path(report["tda_module"]).resolve()
    if where != (tools / "tda").resolve():
        raise CheckFailed(f"tda was imported from {where}, not from the pack")
    return {"truth_files": [p.name for p in truth], "python": report.get("python"),
            "tda_module": str(where), "files": report["files"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--venv", default=DEFAULT_VENV)
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter the venv is created from")
    ap.add_argument("--wheelhouse", default=None,
                    help="a directory of wheels for pip install --no-index (then pip is used)")
    ap.add_argument("--yaml-from", default=None,
                    help="site-packages holding the yaml package to copy (offline default: "
                         "this interpreter's)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fresh", action="store_true", help="recreate the venv")
    args = ap.parse_args(argv)
    pack = Path(args.pack)
    venv = Path(args.venv)
    out = Path(args.out) if args.out else venv.parent / "venv_check"
    try:
        py = make_venv(venv, args.python, with_pip=bool(args.wheelhouse), fresh=args.fresh)
        if args.wheelhouse:
            install_requirements(py, args.wheelhouse, pack / "tools" / "requirements.txt")
            yaml_source = f"wheelhouse {args.wheelhouse}"
        else:
            yaml_source = install_yaml(py, args.yaml_from)
        result = run_check(pack, py, out)
    except (CheckFailed, subprocess.CalledProcessError, OSError) as exc:
        print(f"[cleanvenv] FAILED: {exc}")
        return 1
    report = {"ok": True, "pack": str(pack), "venv": str(venv), "yaml": yaml_source, **result}
    out.mkdir(parents=True, exist_ok=True)
    (out / "cleanvenv_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"[cleanvenv] OK: gold = 1.0 on every task of {len(result['truth_files'])} files "
          f"with the pack's tools alone (python {result['python']}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
