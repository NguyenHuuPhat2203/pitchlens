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


def test_moved_project_repoints_stale_ultralytics_dirs(tmp_path) -> None:
    """A settings file written before the project moved must not keep the old folders."""
    settings = tmp_path / "Ultralytics" / "settings.json"
    settings.parent.mkdir()
    old = {"settings_version": "0.0.8", "uuid": "keep-me", "weights_dir": "D:/old/place/weights",
           "runs_dir": "D:/old/place/runs", "datasets_dir": "D:/old/place/data/datasets"}
    settings.write_text(json.dumps(old), encoding="utf-8")
    env = {**os.environ, "YOLO_CONFIG_DIR": str(tmp_path)}
    code = "import json, src\nfrom ultralytics.utils import SETTINGS\nprint(json.dumps(dict(SETTINGS)))"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT, env=env, capture_output=True, text=True, check=True,
    )
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    for key in ("weights_dir", "runs_dir", "datasets_dir"):
        assert Path(loaded[key]).resolve().is_relative_to(PROJECT), (key, loaded[key])
    assert loaded["uuid"] == "keep-me"
