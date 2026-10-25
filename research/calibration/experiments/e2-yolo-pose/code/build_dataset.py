"""E2 data: SoccerNet-Calibration-2023 line annotations -> fitted homographies -> YOLO-pose labels.

Usage: build_dataset.py  (fits train + valid in parallel, writes data/calibration/yolo-sn32/)
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from fieldreg import DATA, Pitch, apply  # noqa: E402
from sn_labels import fit  # noqa: E402

SN = DATA / "sn-calib-2023"
OUT = DATA / "yolo-sn32"
FITS = DATA / "sn-fits"
W, H = 960, 540
MAX_RMS, MAX_ERR = 3.0, 10.0  # accept a fit only if annotations agree with one homography
PITCH = Pitch()
# Roboflow's left/right mirror (unused while fliplr = 0, kept so the yaml is complete).
FLIP = [24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 8, 6, 7, 0, 1, 2, 3, 4, 5, 31, 30]


def _fit_one(path: str) -> dict | None:
    res = fit(json.loads(Path(path).read_text()), W, H)
    if res is None:
        return None
    return {"image": Path(path).with_suffix(".jpg").name, "H": res["H"].tolist(), "rms_px": res["rms_px"],
            "max_px": res["max_px"], "n": res["n"], "lines": res["lines"]}


def fit_split(split: str) -> list[dict]:
    files = [str(f) for f in sorted((SN / split / split).glob("*.json")) if not f.name.startswith("match_info")]
    # ponytail: 6 workers fit in the RAM left beside the GPU jobs; raise on a bigger box.
    with ProcessPoolExecutor(max_workers=6) as pool:
        fits = [r for r in pool.map(_fit_one, files, chunksize=32) if r is not None]
    FITS.mkdir(parents=True, exist_ok=True)
    (FITS / f"{split}.json").write_text(json.dumps(fits))
    print(split, "images", len(files), "fitted", len(fits),
          "accepted", sum(f["rms_px"] <= MAX_RMS and f["max_px"] <= MAX_ERR for f in fits), flush=True)
    return fits


def label(Hm: np.ndarray) -> str | None:
    """YOLO-pose line: pitch bbox + 32 keypoints (x, y, visibility), normalised."""
    Hinv = np.linalg.inv(Hm)
    front = np.sign((Hinv @ np.r_[apply(Hm, np.array([[W / 2, H - 1.0]]))[0], 1.0])[2])

    def to_image(pts):
        p = np.c_[pts, np.ones(len(pts))] @ Hinv.T
        ok = p[:, 2] * front > 1e-9
        xy = p[:, :2] / np.where(ok, p[:, 2], 1)[:, None]
        return xy, ok & (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)

    gx, gy = np.meshgrid(np.linspace(0, PITCH.length, 106), np.linspace(0, PITCH.width, 69))
    grid, ok = to_image(np.c_[gx.ravel(), gy.ravel()])
    if ok.sum() < 20:
        return None
    x0, y0 = grid[ok].min(axis=0)
    x1, y1 = grid[ok].max(axis=0)
    kp, vis = to_image(PITCH.keypoints32())
    if vis.sum() < 4:
        return None  # nothing a detector could anchor a homography on
    vals = [0, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
    for (x, y), v in zip(kp, vis):
        vals += [x / W, y / H, 2] if v else [0, 0, 0]
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals)


def write_yolo(split: str, fits: list[dict], name: str) -> int:
    (OUT / "images" / name).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / name).mkdir(parents=True, exist_ok=True)
    n = 0
    for f in fits:
        if f["rms_px"] > MAX_RMS or f["max_px"] > MAX_ERR:
            continue
        line = label(np.array(f["H"]))
        if line is None:
            continue
        src, dst = SN / split / split / f["image"], OUT / "images" / name / f["image"]
        if not dst.exists():
            os.link(src, dst)  # same volume: no copy of the 2.5 GB of images
        (OUT / "labels" / name / f["image"].replace(".jpg", ".txt")).write_text(line + "\n")
        n += 1
    return n


if __name__ == "__main__":
    counts = {}
    for split, name in (("valid", "val"), ("train", "train")):
        counts[name] = write_yolo(split, fit_split(split), name)
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nkpt_shape: [32, 3]\n"
        f"flip_idx: {FLIP}\nnames:\n  0: pitch\n")
    print("yolo labels", counts)
