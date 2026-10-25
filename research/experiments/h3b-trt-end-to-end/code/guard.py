"""H3b guard: compare per-frame outputs of a B run and a T run."""

import json
import sys
from pathlib import Path

import numpy as np

base, test = (Path(a) for a in sys.argv[1:3])


def load(d):
    return [i["result"] for i in json.loads((d / "frames.json").read_text())]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


B, T = load(base), load(test)
matched = total = 0
ious = []
for fb, ft in zip(B, T):
    boxes = [p["bbox"] for p in ft["players"]]
    used = set()
    for p in fb["players"]:
        total += 1
        best, j = max(((iou(p["bbox"], q), k) for k, q in enumerate(boxes) if k not in used),
                      default=(0.0, -1))
        if best >= 0.5:
            matched += 1
            used.add(j)
            ious.append(best)


def in_play(frames):
    return sum((f["ball"] or {}).get("state") == "in_play" for f in frames) / len(frames)


def shots(frames):
    return [e["frame_number"] for f in frames for e in f["events"]]


print(json.dumps({
    "frames": [len(B), len(T)],
    "player_recall": round(matched / total, 4), "player_mean_iou": round(float(np.mean(ious)), 4),
    "ball_in_play": [round(in_play(B), 4), round(in_play(T), 4)],
    "shots": [shots(B), shots(T)],
}))
