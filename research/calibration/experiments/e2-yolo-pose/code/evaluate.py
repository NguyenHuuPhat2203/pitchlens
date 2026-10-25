"""E2 evaluation: E1's protocol for our checkpoint, plus SoccerNet test against fitted homographies.

Usage: evaluate.py <checkpoint relative to the project root>
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[2] / "tools"
sys.path[:0] = [str(TOOLS), str(HERE.parents[1] / "e1-off-the-shelf" / "code"), str(HERE)]
import src  # noqa: E402,F401
from build_dataset import MAX_ERR, MAX_RMS, SN, _fit_one  # noqa: E402
from fieldreg import Pitch, projection_error  # noqa: E402

OUT = HERE.parent / "results"
SN_PITCH = Pitch()
R180 = np.array([[-1, 0, SN_PITCH.length], [0, -1, SN_PITCH.width], [0, 0, 1.0]])


def sn_test_gt() -> list[dict]:
    """Fitted homographies for SoccerNet test (cached), same acceptance as the training labels."""
    cache = OUT / "sn_test_fits.json"
    if cache.exists():
        return json.loads(cache.read_text())
    files = [str(f) for f in sorted((SN / "test" / "test").glob("*.json")) if not f.name.startswith("match_info")]
    with ProcessPoolExecutor(max_workers=6) as pool:
        fits = [r for r in pool.map(_fit_one, files, chunksize=32) if r and r["rms_px"] <= MAX_RMS and r["max_px"] <= MAX_ERR]
    cache.write_text(json.dumps(fits))
    return fits


def sn_test(fn, fits: list[dict]) -> dict:
    errs, missed = [], 0
    for f in fits:
        img = cv2.imread(str(SN / "test" / "test" / f["image"]))
        H = fn(img)
        if H is None:
            missed += 1
            continue
        Hg = np.array(f["H"])
        errs.append(min(projection_error(H, Hg, 960, 540, SN_PITCH), projection_error(R180 @ H, Hg, 960, 540, SN_PITCH)))
    e = np.array(errs)
    return {"images": len(fits), "completeness": round(len(e) / len(fits), 4),
            "acc_1m": round(float((e < 1).sum() / len(fits)), 4),
            "median_err_m": round(float(np.median(e)), 3), "mean_err_m": round(float(e.mean()), 3)}


if __name__ == "__main__":
    import run
    from methods import YoloPitch
    from run import evaluate, summary

    run.OUT = OUT  # overlays of this checkpoint go to E2's results, not E1's

    ckpt = sys.argv[1]
    tag = Path(ckpt).stem  # best / last: results side by side
    OUT.mkdir(parents=True, exist_ok=True)
    name = "yolo:" + ckpt
    res = evaluate(name)  # TS-WC test, WC14 test, sample overlays (E1 code and metrics)
    report = summary(res["rows"], res["latency_ms"])
    (OUT / f"rows_e2_{tag}.json").write_text(json.dumps(res["rows"]))
    fits = sn_test_gt()
    report["sn_test"] = sn_test(YoloPitch(Path(ckpt).resolve(), SN_PITCH, imgsz=640), fits)
    (OUT / f"summary_e2_{tag}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
