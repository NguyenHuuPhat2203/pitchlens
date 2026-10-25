"""E1: PnLCalib (SV) and Roboflow YOLOv8x-pose on TS-WorldCup test, WC14 test and sample.mp4."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools"))
import src  # noqa: E402,F401  project-local caches
from fieldreg import (ROOT, WC_LENGTH, WC_WIDTH, Pitch, displacement_error, draw,  # noqa: E402
                      projection_error, tswc, wc14_test)

OUT = HERE.parent / "results"
TEMPLATE = Pitch(WC_LENGTH, WC_WIDTH)
SAMPLE_FRAMES = [0, 60, 120, 180, 230, 1330, 1400, 1480, 400, 800, 1100]


def make(name: str):
    from methods import PnLCalib, YoloPitch

    if name == "pnlcalib_sv":
        m = PnLCalib()
        # Its ground model is 105 x 68 m; stretch onto the GT template so landmarks coincide.
        to_template = np.diag([WC_LENGTH / 105.0, WC_WIDTH / 68.0, 1.0])
        return (lambda img: None if (H := m(img)) is None else to_template @ H), m
    if name == "roboflow_yolov8x":
        m = YoloPitch(ROOT / ".cache" / "calib" / "roboflow" / "football-pitch-detection.pt", TEMPLATE, imgsz=640)
        return m, m
    if name.startswith("yolo:"):  # any 32-keypoint pitch checkpoint, e.g. E2's
        m = YoloPitch(ROOT / name[5:], TEMPLATE, imgsz=640)
        return m, m
    raise ValueError(name)


def timed(fn, img):
    t = time.perf_counter()
    H = fn(img)
    return H, (time.perf_counter() - t) * 1000


def evaluate(name: str) -> dict:
    fn, model = make(name)
    fn(cv2.imread(str(wc14_test()[0][0])))  # warm-up (CUDA init, cudnn autotune)
    rows, lat = [], []
    for video, frames in tswc("test").items():
        prev = None
        for i, (path, Hg) in enumerate(frames):
            img = cv2.imread(str(path))
            H, ms = timed(fn, img)
            lat.append(ms)
            h, w = img.shape[:2]
            row = {"set": "tswc", "video": video, "frame": path.stem, "ok": H is not None,
                   "err": projection_error(H, Hg, w, h, TEMPLATE) if H is not None else None, "disp": None}
            if H is not None and prev is not None and prev[0] is not None:
                row["disp"] = displacement_error(prev[0], H, prev[1], Hg, w, h, TEMPLATE)
            prev = (H, Hg)
            rows.append(row)
    for path, Hg in wc14_test():
        img = cv2.imread(str(path))
        H, ms = timed(fn, img)
        lat.append(ms)
        h, w = img.shape[:2]
        rows.append({"set": "wc14", "video": "", "frame": path.stem, "ok": H is not None,
                     "err": projection_error(H, Hg, w, h, TEMPLATE) if H is not None else None, "disp": None})

    # Target domain: overlays on fixed sample.mp4 frames (product pitch, 105 x 68 m).
    cap, tiles = cv2.VideoCapture(str(ROOT / "data" / "sample.mp4")), []
    product = Pitch()
    for n in SAMPLE_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        _, img = cap.read()
        H = fn(img)
        if H is not None and name == "pnlcalib_sv":
            H = np.diag([105.0 / WC_LENGTH, 68.0 / WC_WIDTH, 1.0]) @ H  # back to the product pitch
        over = draw(img, H, product) if H is not None else img.copy()
        cv2.putText(over, f"{name} frame {n}: {'H' if H is not None else 'NO H'}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        cv2.imwrite(str(OUT / f"sample_{safe}_{n:04d}.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 88])
        tiles.append(cv2.resize(over, (640, 360)))
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
    cv2.imwrite(str(OUT / f"sample_{safe}_sheet.jpg"),
                np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    del model
    return {"rows": rows, "latency_ms": lat}


def summary(rows: list[dict], latency: list[float]) -> dict:
    out = {}
    for s in ("tswc", "wc14"):
        r = [x for x in rows if x["set"] == s]
        errs = np.array([x["err"] for x in r if x["ok"]])
        disp = np.array([x["disp"] for x in r if x["disp"] is not None])
        out[s] = {
            "frames": len(r),
            "acc_1m": round(float((errs < 1.0).sum() / len(r)), 4),
            "completeness": round(len(errs) / len(r), 4),
            "mean_err_m": round(float(errs.mean()), 3) if len(errs) else None,
            "median_err_m": round(float(np.median(errs)), 3) if len(errs) else None,
            "p90_err_m": round(float(np.percentile(errs, 90)), 3) if len(errs) else None,
            "mean_disp_m": round(float(disp.mean()), 3) if len(disp) else None,
            "median_disp_m": round(float(np.median(disp)), 3) if len(disp) else None,
        }
    out["latency_ms_median"] = round(float(np.median(latency)), 1)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in sys.argv[1:]:
        res = evaluate(name)
        report[name] = summary(res["rows"], res["latency_ms"])
        (OUT / f"rows_{name}.json").write_text(json.dumps(res["rows"]))
        print(name, json.dumps(report[name]), flush=True)
    (OUT / f"summary_{'_'.join(sys.argv[1:])}.json").write_text(json.dumps(report, indent=2))
