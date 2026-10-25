"""Ground-truth homographies for sample.mp4 frames (snap + SoccerNet-style fit).

rough.json maps a frame to either hand-read rough polylines, or {"propagate_from": "<frame>"}:
then the rough polylines are the pitch markings of that frame's GT, carried over by a global
ORB homography between the two frames (a broadcast camera on a fixed head rotates, so every
scene point moves by one homography). The rough position only picks which white ridge to snap
to; the GT itself is fitted to the snapped points of this frame alone. Check every overlay.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools"))
from annotate import fit_frame  # noqa: E402
from fieldreg import ROOT, Pitch, draw  # noqa: E402
from linefit import camera_motion as _orb_motion  # noqa: E402
from sn_labels import P  # noqa: E402

L, W = P.length, P.width
B0, B1 = (W - P.box_width) / 2, (W + P.box_width) / 2
G0, G1 = (W - P.goal_area_width) / 2, (W + P.goal_area_width) / 2
GOAL = (W / 2 - 5.0, W / 2 + 5.0)  # goal mouth + net: white mesh, never snap there
SEGMENTS = {
    "Side line left": [((0, 0), (0, GOAL[0])), ((0, GOAL[1]), (0, W))],
    "Side line right": [((L, 0), (L, GOAL[0])), ((L, GOAL[1]), (L, W))],
    "Side line top": [((0, 0), (L, 0))], "Side line bottom": [((0, W), (L, W))],
    "Middle line": [((L / 2, 0), (L / 2, W))],
    "Big rect. left main": [((P.box_depth, B0), (P.box_depth, B1))],
    "Big rect. left top": [((0, B0), (P.box_depth, B0))], "Big rect. left bottom": [((0, B1), (P.box_depth, B1))],
    "Small rect. left main": [((P.goal_area_depth, G0), (P.goal_area_depth, G1))],
    "Small rect. left top": [((0, G0), (P.goal_area_depth, G0))],
    "Small rect. left bottom": [((0, G1), (P.goal_area_depth, G1))],
    "Big rect. right main": [((L - P.box_depth, B0), (L - P.box_depth, B1))],
    "Big rect. right top": [((L, B0), (L - P.box_depth, B0))], "Big rect. right bottom": [((L, B1), (L - P.box_depth, B1))],
}
ARCS = {"Circle central": ((L / 2, W / 2), 0.0, 2 * np.pi),
        "Circle left": ((P.spot_distance, W / 2), -0.92, 0.92),
        "Circle right": ((L - P.spot_distance, W / 2), np.pi - 0.92, np.pi + 0.92)}


def model_polylines() -> dict[str, list[np.ndarray]]:
    out = {k: [np.linspace(a, b, max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 0.5))) for a, b in segs]
           for k, segs in SEGMENTS.items()}
    for k, (c, t0, t1) in ARCS.items():
        t = np.linspace(t0, t1, 240)
        out[k] = [np.c_[c[0] + P.circle_radius * np.cos(t), c[1] + P.circle_radius * np.sin(t)]]
    return out


def camera_motion(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Global ORB homography src -> dst with the static broadcast logos masked."""
    mask = np.full(src.shape[:2], 255, np.uint8)
    mask[590:, 1080:] = 0  # "Tactic Pro" / "RT software" logos stay put on screen
    return _orb_motion(src, dst, mask, features=6000)


def propagated_rough(H_src: np.ndarray, G: np.ndarray, w: int, h: int) -> dict[str, list]:
    """Pitch markings of the source GT, moved into the target frame, as rough polylines."""
    to_img = G @ np.linalg.inv(H_src)
    rough = {}
    for name, polys in model_polylines().items():
        out = []
        for poly in polys:
            p = np.c_[poly, np.ones(len(poly))] @ to_img.T
            xy = p[:, :2] / p[:, 2:3]
            inside = (xy[:, 0] > 3) & (xy[:, 0] < w - 3) & (xy[:, 1] > 3) & (xy[:, 1] < h - 3) & (p[:, 2] * np.sign(p[len(p) // 2, 2]) > 0)
            for q, ok in zip(xy[::4], inside[::4]):
                out.append([float(q[0]), float(q[1])] if ok else None)
            out.append(None)
        if sum(v is not None for v in out) >= 2:
            rough[name] = out
    return rough


if __name__ == "__main__":
    spec = {k: v for k, v in json.loads((HERE / "rough.json").read_text()).items() if not k.startswith("_")}
    cap, out, frames = cv2.VideoCapture(str(ROOT / "data" / "sample.mp4")), {}, {}
    (HERE / "overlays").mkdir(exist_ok=True)
    for frame, ann in spec.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        img = frames[frame] = cap.read()[1]
        h, w = img.shape[:2]
        if "propagate_from" in ann:
            src = ann["propagate_from"]
            ann = propagated_rough(np.array(out[src]["H"]), camera_motion(frames[src], img), w, h)
        res, snapped = fit_frame(img, ann)
        kept = {k: len(v) for k, v in snapped.items()}
        if res is None:
            print(frame, "NO FIT", kept)
            continue
        out[frame] = {"H": res["H"].tolist(), "rms_px": round(res["rms_px"], 3), "max_px": round(res["max_px"], 3),
                      "n": res["n"], "lines": res["lines"]}
        over = draw(img, res["H"], Pitch(), (0, 255, 255), 1)
        for pts in snapped.values():
            for x, y in pts:
                cv2.circle(over, (int(round(x)), int(round(y))), 2, (255, 0, 255), -1)
        cv2.imwrite(str(HERE / "overlays" / f"gt_{int(frame):04d}.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(frame, f"rms {res['rms_px']:.2f}px max {res['max_px']:.2f}px n {res['n']}", kept, flush=True)
    (HERE / "gt.json").write_text(json.dumps(out, indent=1))

    # Dense GT: every frame of the clean segments, each refitted to its own lines, the rough
    # position carried over from the previous frame (tiny motion between consecutive frames).
    dense = {}
    for start, lo, hi in ((0, 0, 240), (1330, 1320, 1505)):
        for order in (range(start, hi), range(start, lo - 1, -1)):
            prev = None
            for n in order:
                cap.set(cv2.CAP_PROP_POS_FRAMES, n)
                img = cap.read()[1]
                if prev is None:
                    res = {"H": np.array(out[str(n)]["H"]), "rms_px": out[str(n)]["rms_px"], "n": out[str(n)]["n"]}
                else:
                    rough = propagated_rough(prev[1], camera_motion(prev[0], img), img.shape[1], img.shape[0])
                    res = fit_frame(img, rough)[0]
                if res is None or res["rms_px"] > 3.0:
                    print("dense: lost track at frame", n, flush=True)
                    break
                dense[str(n)] = {"H": np.asarray(res["H"]).tolist(), "rms_px": round(res["rms_px"], 3), "n": res["n"]}
                prev = (img, np.asarray(res["H"]))
    (HERE / "gt_dense.json").write_text(json.dumps(dict(sorted(dense.items(), key=lambda kv: int(kv[0])))))
    rms = np.array([v["rms_px"] for v in dense.values()])
    print("dense frames", len(dense), "rms median", round(float(np.median(rms)), 2), "max", round(float(rms.max()), 2))
