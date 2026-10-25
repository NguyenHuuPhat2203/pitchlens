"""E5: run the shipped AutoCalibrator over every frame of sample.mp4 and score it against the dense GT."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path[:0] = [str(ROOT), str(HERE.parents[2] / "tools")]
from fieldreg import Pitch, displacement_error, projection_error

from src.serve.pipeline.auto_calibrator import AutoCalibrator

OUT = HERE.parent / "results"
P = Pitch()
R180 = np.array([[-1, 0, P.length], [0, -1, P.width], [0, 0, 1.0]])

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dense = {int(k): np.array(v["H"]) for k, v in json.loads((HERE.parents[1] / "gt-sample" / "gt_dense.json").read_text()).items()}
    ac, cap = AutoCalibrator(), cv2.VideoCapture(str(ROOT / "data" / "sample.mp4"))
    rows, n = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        was_tracking = ac.is_calibrated
        t = time.perf_counter()
        H = ac.update(img, n)
        ms = (time.perf_counter() - t) * 1000
        row = {"frame": n, "ms": round(ms, 1), "mode": "track" if was_tracking else "search", "ok": H is not None}
        if H is not None:
            row["H"] = H.tolist()
            if n in dense:
                e0, e1 = projection_error(H, dense[n], 1280, 720, P), projection_error(R180 @ H, dense[n], 1280, 720, P)
                row["err"], row["flip"] = min(e0, e1), bool(e1 < e0)
        rows.append(row)
        n += 1
    by = {r["frame"]: r for r in rows}
    gt_rows = [r for r in rows if r["frame"] in dense]
    errs = np.array([r.get("err", np.inf) for r in gt_rows])
    disp = []
    for r in gt_rows:
        m = r["frame"] + 12
        if "err" in r and m in dense and by.get(m, {}).get("ok"):
            F = R180 if r["flip"] else np.eye(3)
            disp.append(displacement_error(F @ np.array(r["H"]), F @ np.array(by[m]["H"]), dense[r["frame"]], dense[m], 1280, 720, P))
    track_ms = [r["ms"] for r in rows if r["mode"] == "track"]
    search_ms = [r["ms"] for r in rows if r["mode"] == "search"]
    summary = {
        "frames": len(rows), "with_homography": round(float(np.mean([r["ok"] for r in rows])), 4),
        "gt_frames": len(gt_rows), "gt_acc_1m": round(float((errs < 1).mean()), 4),
        "gt_median_err_m": round(float(np.median(errs[np.isfinite(errs)])), 3) if np.isfinite(errs).any() else None,
        "gt_max_err_m": round(float(errs[np.isfinite(errs)].max()), 3) if np.isfinite(errs).any() else None,
        "disp12_median_m": round(float(np.median(disp)), 3) if disp else None,
        "disp12_p95_m": round(float(np.percentile(disp, 95)), 3) if disp else None,
        "track_ms_median": round(float(np.median(track_ms)), 1) if track_ms else None,
        "searches": len(search_ms), "search_ms_median": round(float(np.median(search_ms))) if search_ms else None,
        "lost_frames": [r["frame"] for r in rows if not r["ok"]][:50],
    }
    (OUT / "rows.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "H"} for r in rows]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
