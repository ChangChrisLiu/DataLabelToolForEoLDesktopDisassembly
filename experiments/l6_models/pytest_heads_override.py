"""A pytest plugin: run ``tests/test_u19_weights.py`` against another head, without editing it.

::

    set L6_SAM_YAML=D:/DataSet/.cache/tmp/l6/configs/sam_v3.yaml
    set L6_SAM_FULL=D:/DataSet/models/weights/sam2.1_hiera_large_tda_l6_v3.pt
    D:\\Anaconda\\envs\\tda\\python.exe -m pytest -p experiments.l6_models.pytest_heads_override \\
        tests/test_u19_weights.py

The test module reads the head configuration from its ``DEFAULT_FILE``
(``configs/sam.yaml``) and the whole fine-tuned model from ``FULL_F5_V2``;
after collection this points both at the L6 files named by the environment, so
the same three tests check that the v3 head file is the whole v3 model, that
the stock head is today's service, and that the routing sends each prompt to
the model it stands for.  Nothing in ``tests/`` or ``configs/`` changes.
"""
from __future__ import annotations

import os
from pathlib import Path


def pytest_collection_modifyitems(session, config, items):
    yaml_path = os.environ.get("L6_SAM_YAML")
    full = os.environ.get("L6_SAM_FULL")
    if not yaml_path or not full:
        return
    seen = set()
    for item in items:
        mod = getattr(item, "module", None)
        if mod is None or not mod.__name__.endswith("test_u19_weights") or id(mod) in seen:
            continue
        seen.add(id(mod))
        mod.DEFAULT_FILE = Path(yaml_path)
        mod.FULL_F5_V2 = Path(full)
        print(f"\n[l6] test_u19_weights: head config {yaml_path}, whole model {full}")
