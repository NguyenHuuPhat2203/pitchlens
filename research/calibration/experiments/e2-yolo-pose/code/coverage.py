"""E2 (exploratory): how much of SoccerNet looks along the pitch length, like sample.mp4?

View direction on the ground = projection of the image centre minus projection of the image's
bottom centre. End-on if within 37 degrees of the length axis (|cos| > 0.8). Also reports the
same numbers for the sample GT frames, so 'like sample.mp4' is measured, not eyeballed.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools"))
from fieldreg import DATA, apply  # noqa: E402

MAX_RMS, MAX_ERR = 3.0, 10.0


def view(H: np.ndarray, w: int, h: int) -> tuple[float, float]:
    """(|cos| of view direction to the length axis, ground depth in metres from bottom to centre)."""
    b, c = apply(H, np.array([[w / 2, h - 1.0], [w / 2, h / 2]]))
    d = c - b
    return abs(d[0]) / (np.linalg.norm(d) + 1e-9), float(np.linalg.norm(d))


if __name__ == "__main__":
    report = {}
    for split in ("train", "valid"):
        fits = [f for f in json.loads((DATA / "sn-fits" / f"{split}.json").read_text())
                if f["rms_px"] <= MAX_RMS and f["max_px"] <= MAX_ERR]
        v = np.array([view(np.array(f["H"]), 960, 540) for f in fits])
        end_on = v[:, 0] > 0.8
        report[split] = {"accepted": len(fits), "end_on": int(end_on.sum()),
                         "end_on_share": round(float(end_on.mean()), 4),
                         "end_on_and_deep_(>25m)": int((end_on & (v[:, 1] > 25)).sum())}
    gt = json.loads((HERE.parents[1] / "gt-sample" / "gt.json").read_text())
    report["sample_gt"] = {k: [round(x, 2) for x in view(np.array(g["H"]), 1280, 720)] for k, g in gt.items()}
    out = HERE.parent / "results"
    out.mkdir(exist_ok=True)
    (out / "coverage.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
