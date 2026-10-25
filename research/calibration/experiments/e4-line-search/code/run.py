"""E4: single-frame line search on the test splits (settings in protocol.md)."""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools"))
from fieldreg import ROOT, WC_LENGTH, WC_WIDTH, Pitch, projection_error, tswc, wc14_test  # noqa: E402
from linesearch import LineSearch  # noqa: E402

OUT = HERE.parent / "results"
GT = HERE.parents[1] / "gt-sample"


def err(H, Hg, P, w, h):
    R = np.array([[-1, 0, P.length], [0, -1, P.width], [0, 0, 1.0]])
    return min(projection_error(H, Hg, w, h, P), projection_error(R @ H, Hg, w, h, P))


def run(ls, items, P, tag):
    rows = []
    for name, img, Hg in items:
        t = time.perf_counter()
        H = ls(img)
        ms = (time.perf_counter() - t) * 1000
        h, w = img.shape[:2]
        rows.append({"set": tag, "frame": name, "ms": round(ms), "err": None if H is None else err(H, Hg, P, w, h),
                     **{k: v for k, v in ls.last.items() if k in ("f", "precision", "recall", "matched")}})
        print(tag, name, rows[-1]["err"], flush=True)
    return rows


def sample_items():
    dense = {int(k): np.array(v["H"]) for k, v in json.loads((GT / "gt_dense.json").read_text()).items()}
    cap = cv2.VideoCapture(str(ROOT / "data" / "sample.mp4"))
    for n in range(1320, 1505, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        yield str(n), cap.read()[1], dense[n]


def summarise(rows):
    out = {}
    for s in sorted({r["set"] for r in rows}):
        r = [x for x in rows if x["set"] == s]
        e = np.array([x["err"] for x in r if x["err"] is not None])
        out[s] = {"frames": len(r), "completeness": round(len(e) / len(r), 4),
                  "acc_1m": round(float((e < 1).sum() / len(r)), 4),
                  "wrong_rate_2m": round(float((e > 2).sum() / len(r)), 4),
                  "median_err_m": round(float(np.median(e)), 3) if len(e) else None,
                  "median_ms": round(float(np.median([x["ms"] for x in r])))}
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    S, T = Pitch(), Pitch(WC_LENGTH, WC_WIDTH)
    rows = run(LineSearch(S), sample_items(), S, "sample_test")
    wc = [(p.stem, cv2.imread(str(p)), Hg) for p, Hg in wc14_test()[::2]]
    ts = [(f"{v[-25:]}/{p.stem}", cv2.imread(str(p)), Hg) for v, fr in tswc("test").items() for p, Hg in fr[::5]]
    rows += run(LineSearch(T), wc, T, "wc14_test") + run(LineSearch(T), ts, T, "tswc_test")
    (OUT / "rows.json").write_text(json.dumps(rows))
    summary = summarise(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
