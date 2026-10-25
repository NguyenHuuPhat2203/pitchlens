"""E3: track from a correct first homography; variants full / orb_only / lines_only."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools"))
from fieldreg import (ROOT, WC_LENGTH, WC_WIDTH, Pitch, apply, displacement_error,  # noqa: E402
                      projection_error, tswc)
from linefit import LineAligner, camera_motion  # noqa: E402

OUT = HERE.parent / "results"
GT = HERE.parents[1] / "gt-sample"
WINDOW = 12  # frames in the product's speed window


class Variant:
    def __init__(self, kind: str, pitch: Pitch) -> None:
        self.kind, self.aligner = kind, LineAligner(pitch)

    def start(self, image, H):
        self.prev, self.H = image, H

    def step(self, image):
        pred = self.H
        if self.kind in ("full", "orb_only"):
            G = camera_motion(self.prev, image)
            if G is not None:
                pred = self.H @ np.linalg.inv(G)
                pred = pred / pred[2, 2]
        self.H = pred if self.kind == "orb_only" else self.aligner.refine(pred, image)[0]
        self.prev = image
        return self.H


def smoothed(Hs: dict[int, np.ndarray], w: int, h: int, k: int = 2) -> dict[int, np.ndarray]:
    """Moving average of homographies through their 4-corner representation (GT noise indication)."""
    corners = np.array([[w * .2, h * .3], [w * .8, h * .3], [w * .8, h * .9], [w * .2, h * .9]])
    proj = {n: apply(H, corners) for n, H in Hs.items()}
    out = {}
    for n in Hs:
        near = [proj[m] for m in range(n - k, n + k + 1) if m in proj]
        out[n] = cv2.getPerspectiveTransform(corners.astype(np.float32), np.mean(near, axis=0).astype(np.float32))
    return out


def run_sample(kind: str) -> list[dict]:
    dense = {int(k): np.array(v["H"]) for k, v in json.loads((GT / "gt_dense.json").read_text()).items()}
    cap, pitch, rows = cv2.VideoCapture(str(ROOT / "data" / "sample.mp4")), Pitch(), []
    for start, order in ((0, range(0, 240)), (1330, range(1330, 1505)), (1330, range(1330, 1319, -1))):
        v = Variant(kind, pitch)
        for n in order:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            img = cap.read()[1]
            if n == start:
                v.start(img, dense[n])
                continue
            H = v.step(img)
            rows.append({"set": "sample", "seq": f"{start}:{order.step}", "frame": n, "H": H.tolist(),
                         "err": projection_error(H, dense[n], 1280, 720, pitch)})
    return rows


def run_tswc(kind: str) -> list[dict]:
    pitch, rows = Pitch(WC_LENGTH, WC_WIDTH), []
    for video, frames in tswc("test").items():
        v = Variant(kind, pitch)
        v.start(cv2.imread(str(frames[0][0])), frames[0][1])
        for i, (path, Hg) in enumerate(frames[1:], start=1):
            img = cv2.imread(str(path))
            H = v.step(img)
            rows.append({"set": "tswc", "seq": video, "frame": i,
                         "err": projection_error(H, Hg, img.shape[1], img.shape[0], pitch)})
    return rows


def speed_metric(rows: list[dict]) -> tuple[list[float], list[float]]:
    """12-frame displacement error of the tracker, and of the GT against its own moving average."""
    dense = {int(k): np.array(v["H"]) for k, v in json.loads((GT / "gt_dense.json").read_text()).items()}
    gt_smooth = smoothed(dense, 1280, 720)
    pitch, pred = Pitch(), {r["frame"]: np.array(r["H"]) for r in rows if r["set"] == "sample"}
    pred_err, gt_err = [], []
    for n in sorted(pred):
        if n + WINDOW in pred:
            pred_err.append(displacement_error(pred[n], pred[n + WINDOW], dense[n], dense[n + WINDOW], 1280, 720, pitch))
        if n in dense and n + WINDOW in dense:
            gt_err.append(displacement_error(gt_smooth[n], gt_smooth[n + WINDOW], dense[n], dense[n + WINDOW],
                                             1280, 720, pitch))
    return pred_err, gt_err


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for kind in ("full", "orb_only", "lines_only"):
        rows = run_sample(kind) + run_tswc(kind)
        disp, gt_disp = speed_metric(rows)
        res = {}
        for s in ("sample", "tswc"):
            e = np.array([r["err"] for r in rows if r["set"] == s])
            res[s] = {"frames": len(e), "acc_1m": round(float((e < 1).mean()), 4),
                      "median_err_m": round(float(np.median(e)), 3), "mean_err_m": round(float(e.mean()), 3),
                      "p95_err_m": round(float(np.percentile(e, 95)), 3), "max_err_m": round(float(e.max()), 3)}
        res["sample"]["disp12_median_m"] = round(float(np.median(disp)), 3)
        res["sample"]["disp12_p95_m"] = round(float(np.percentile(disp, 95)), 3)
        res["gt_vs_smoothed_gt_disp12_median_m"] = round(float(np.median(gt_disp)), 3)
        report[kind] = res
        (OUT / f"rows_{kind}.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "H"} for r in rows]))
        print(kind, json.dumps(res), flush=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
