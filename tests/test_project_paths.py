"""Model downloads and caches must stay inside the project folder."""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def test_ultralytics_torch_and_hf_caches_live_in_project() -> None:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"YOLO_CONFIG_DIR", "TORCH_HOME", "HF_HOME"}
    }
    code = (
        "import json, os, src\n"
        "from ultralytics.utils import SETTINGS, USER_CONFIG_DIR\n"
        "print(json.dumps({'config': str(USER_CONFIG_DIR),"
        " 'torch': os.environ['TORCH_HOME'], 'hf': os.environ['HF_HOME'],"
        " 'weights': SETTINGS['weights_dir'], 'runs': SETTINGS['runs_dir'],"
        " 'datasets': SETTINGS['datasets_dir']}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT, env=env, capture_output=True, text=True, check=True,
    )
    paths = json.loads(out.stdout.strip().splitlines()[-1])
    for name, value in paths.items():
        assert Path(value).resolve().is_relative_to(PROJECT), (name, value)
