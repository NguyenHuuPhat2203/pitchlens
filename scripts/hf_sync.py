"""Sync E2 training data and runs with one private Hugging Face dataset repo (Colab training).

Config in <repo>/.env (see .env.example):   HF_TOKEN=hf_...   HF_REPO=<user>/football-tracker

    python scripts/hf_sync.py data            # zip + upload data/calibration/yolo-sn32 (once, ~1.9 GB)
    python scripts/hf_sync.py pull runs       # fetch runs/ (Colab outputs: weights, results.csv, plots)

Repo layout:  data/calibration/yolo-sn32/{train.zip,val.zip,data.yaml}   runs/pose/<run>/...
Re-running `data` is cheap: zips already in the repo are skipped.
Notebook: notebooks/training/04_train_pitch_keypoints_sn32.ipynb
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DATASET = "data/calibration/yolo-sn32"  # same path locally, in the repo and on Colab


def load_env(path: Path = ROOT / ".env") -> dict[str, str]:
    env = dict(os.environ)
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.split("#", 1)[0].strip().strip("'\""))
    missing = [k for k in ("HF_TOKEN", "HF_REPO") if not env.get(k)]
    if missing:
        raise SystemExit(f"Missing {missing}: put them in {path} (see .env.example)")
    return env


def zip_split(dataset: Path, split: str, out: Path) -> Path:
    """images/<split>/* and labels/<split>/* stored (JPEGs do not compress) in one zip."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for kind in ("images", "labels"):
            for f in sorted((dataset / kind / split).iterdir()):
                zf.write(f, f"{kind}/{split}/{f.name}")
    tmp.replace(out)
    return out


def push_data(api: HfApi, repo: str) -> None:
    dataset = ROOT / DATASET
    for split in ("train", "val"):
        target = f"{DATASET}/{split}.zip"
        if api.file_exists(repo, target, repo_type="dataset"):
            print("skip (already uploaded):", target)
            continue
        archive = zip_split(dataset, split, ROOT / ".cache" / "hf" / f"yolo-sn32-{split}.zip")
        print(f"uploading {target} ({archive.stat().st_size / 1e9:.2f} GB)")
        api.upload_file(path_or_fileobj=archive, path_in_repo=target, repo_id=repo, repo_type="dataset",
                        commit_message=f"data: yolo-sn32 {split}")
        archive.unlink()
    api.upload_file(path_or_fileobj=dataset / "data.yaml", path_in_repo=f"{DATASET}/data.yaml", repo_id=repo,
                    repo_type="dataset", commit_message="data: yolo-sn32 data.yaml")


def pull(api: HfApi, repo: str, pattern: str, dest: Path) -> None:
    api.snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, allow_patterns=[f"{pattern}/*"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["data", "pull"])
    p.add_argument("pattern", nargs="?", default="runs", help="for pull: top-level folder to fetch")
    p.add_argument("--dest", type=Path, default=ROOT)
    args = p.parse_args()

    env = load_env()
    api = HfApi(token=env["HF_TOKEN"])
    repo = env["HF_REPO"]
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    {"data": lambda: push_data(api, repo),
     "pull": lambda: pull(api, repo, args.pattern, args.dest)}[args.what]()
    print("done:", args.what, repo)


if __name__ == "__main__":
    main()
