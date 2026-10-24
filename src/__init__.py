# Football Tracker — real-time football match tracking with computer vision.
"""Keep every model download, checkpoint, and cache inside the project.

Runs before any ``src`` module imports Ultralytics, torch, or Hugging Face,
so their default per-user folders on drive C are never used. Values already
set in the environment win.
"""

import json
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / ".cache"

os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR))  # settings.json, fonts
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))

# Ultralytics defaults datasets_dir to the parent of the git root, outside the
# project. Point its folders here on every import, so a moved project does not
# keep writing to the old location; Ultralytics fills in the other keys.
_settings = Path(os.environ["YOLO_CONFIG_DIR"]) / "Ultralytics" / "settings.json"
_dirs = {
    "datasets_dir": str(PROJECT_DIR / "data" / "datasets"),
    "weights_dir": str(PROJECT_DIR / "weights"),
    "runs_dir": str(PROJECT_DIR / "runs"),
}
_current = json.loads(_settings.read_text(encoding="utf-8")) if _settings.exists() else {}
if any(_current.get(k) != v for k, v in _dirs.items()):
    _settings.parent.mkdir(parents=True, exist_ok=True)
    _settings.write_text(json.dumps({**_current, **_dirs}, indent=2), encoding="utf-8")
