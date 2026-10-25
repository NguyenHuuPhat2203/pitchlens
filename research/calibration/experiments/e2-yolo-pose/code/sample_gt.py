"""E2 on the target view: single-frame registration of sample.mp4 scored against GT homographies.

Usage: sample_gt.py <checkpoint relative to the project root>

"Registered" in the protocol's sample criterion (>= 6 of the 8 end-on frames 0, 60, 120, 180, 230,
1330, 1400, 1480) is judged here by projection error < 1 m against gt-sample/gt.json instead of by eye.
Also reported: all 425 dense-GT frames (dev 0-239, test 1320-1504), each frame on its own.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path[:0] = [str(HERE.parents[2] / "tools"), str(ROOT)]
import src  # noqa: E402,F401
from fieldreg import Pitch, projection_error  # noqa: E402
from methods import YoloPitch  # noqa: E402

P = Pitch()  # sample.mp4 GT uses the product pitch, 105 x 68 m
R180 = np.array([[-1, 0, P.length], [0, -1, P.width], [0, 0, 1.0]])
GT = HERE.parents[1] / "gt-sample"


def error(H: np.ndarray | None, Hg: np.ndarray) -> float | None:
    if H is None:
        return None
    return min(projection_error(H, Hg, 1280, 720, P), projection_error(R180 @ H, Hg, 1280, 720, P))


def stats(errs: list[float | None]) -> dict:
    e = np.array([x for x in errs if x is not None])
    return {"frames": len(errs), "with_h": len(e), "acc_1m": round(float((e < 1).sum() / len(errs)), 4),
            "median_err_m": round(float(np.median(e)), 3) if len(e) else None}


if __name__ == "__main__":
    ckpt = sys.argv[1]
    model = YoloPitch(ROOT / ckpt, P, imgsz=640)
    eight = {int(k): np.array(v["H"]) for k, v in json.loads((GT / "gt.json").read_text()).items()}
    dense = {int(k): np.array(v["H"]) for k, v in json.loads((GT / "gt_dense.json").read_text()).items()}
    cap, errs = cv2.VideoCapture(str(ROOT / "data" / "sample.mp4")), {}
    for n in sorted(set(eight) | set(dense)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        _, img = cap.read()
        errs[n] = error(model(img), eight[n] if n in eight else dense[n])
    report = {
        "protocol_8": {"errors_m": {n: None if errs[n] is None else round(errs[n], 3) for n in sorted(eight)},
                       "registered_lt_1m": sum(errs[n] is not None and errs[n] < 1 for n in eight)},
        "dense_dev_0_239": stats([errs[n] for n in sorted(dense) if n < 240]),
        "dense_test_1320_1504": stats([errs[n] for n in sorted(dense) if n >= 1320]),
    }
    out = HERE.parent / "results" / f"sample_gt_{Path(ckpt).stem}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
