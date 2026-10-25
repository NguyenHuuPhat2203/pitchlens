"""Football Tracker promo video: real pipeline output -> motion graphics MP4.

Runs the product pipeline on data/sample.mp4, turns its JSON into assets for
promo.html (footage frames, player crops, build/data.js), then screenshots
promo.html frame by frame in the installed Microsoft Edge and encodes H.264.

    .venv/Scripts/python.exe promo/make.py              # promo/football-tracker-demo.mp4
    .venv/Scripts/python.exe promo/make.py still 12 30  # promo/build/still_*.png

Dev-only dependencies: pip install playwright imageio-ffmpeg (no browser download).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
RUN = BUILD / "run"
VIDEO = ROOT / "data" / "sample.mp4"
OUT = HERE / "football-tracker-demo.mp4"
FPS = 30
# Footage without the broadcaster's own analysis graphics (defensive line, arrows),
# so nothing on screen can be mistaken for this product's output.
CLEAN = [(0, 240), (1320, 1505)]
TEAMS_SEGMENT = (1322, 1505)  # team-classification scene: crops, colours, shares
PASS_SEGMENT = (140, 236)  # ball scene: the pass event shown on screen
BG_FRAME, THUMB_FRAME = 120, 200
TEAMS = ("A", "B", "referee")


def run_pipeline() -> None:
    sys.path.insert(0, str(ROOT))
    import src  # noqa: F401  project-local model and cache folders
    from src.config import load_config
    from src.serve.pipeline.pipeline import Pipeline

    cfg = load_config(ROOT / "config" / "default.yaml")
    cfg.video.source = str(VIDEO)
    cfg.pipeline.display = False
    cfg.pipeline.output_video = None
    cfg.pipeline.output_json = str(RUN / "frames.json")
    cfg.timeline.output_json = str(RUN / "timeline.json")
    Pipeline(cfg).run(cfg.video.source)


def _torso_colour(img: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    x1, y1, x2, y2 = (round(v) for v in bbox)
    h = y2 - y1
    crop = img[max(y1 + h // 4, 0) : max(y1 + h // 2, 0), max(x1, 0) : max(x2, 0)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    keep = ~((hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] > 60))  # grass
    return np.median(crop.reshape(-1, 3)[keep], axis=0) if keep.any() else None


def _isolated(player: dict, players: list[dict]) -> bool:
    x1, y1, x2, y2 = player["bbox"]
    return all(
        q is player or q["bbox"][2] < x1 - 4 or q["bbox"][0] > x2 + 4 or q["bbox"][3] < y1 - 4 or q["bbox"][1] > y2 + 4
        for q in players
    )


def _crop(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    crop = img[max(int(y1 - .12 * h), 0) : int(y2 + .12 * h), max(int(x1 - .3 * w), 0) : int(x2 + .3 * w)]
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * 120 / crop.shape[0])), 120), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((132, 72, 3), (24, 17, 11), np.uint8)  # #0b1118
    width = min(crop.shape[1], 72)
    left, src = (72 - width) // 2, (crop.shape[1] - width) // 2
    canvas[6:126, left : left + width] = crop[:, src : src + width]
    return canvas


def _r(values: list[float]) -> list[float]:
    return [round(v, 1) for v in values]


def prepare() -> None:
    frames = [item["result"] for item in json.loads((RUN / "frames.json").read_text(encoding="utf-8"))]
    timeline = json.loads((RUN / "timeline.json").read_text(encoding="utf-8"))
    for name in ("frames", "crops"):
        (BUILD / name).mkdir(parents=True, exist_ok=True)

    compact: dict[int, dict] = {}
    colours: dict[str, list] = {t: [] for t in TEAMS}
    candidates: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {t: {} for t in TEAMS}
    cap = cv2.VideoCapture(str(VIDEO))
    for start, end in CLEAN:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for n in range(start, end):
            ok, img = cap.read()
            if not ok:
                break
            cv2.imwrite(str(BUILD / "frames" / f"{n:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frame, ball = frames[n], frames[n]["ball"]
            compact[n] = {
                "p": [[p["track_id"], *_r(p["bbox"]), p["team"]] for p in frame["players"]],
                "b": _r(ball["centroid"]) if ball and ball["state"] == "in_play" else None,
            }
            if not (TEAMS_SEGMENT[0] <= n < TEAMS_SEGMENT[1] and n % 5 == 0):
                continue
            for p in frame["players"]:
                team = p["team"]
                colour = _torso_colour(img, p["bbox"]) if team in candidates else None
                if colour is None:
                    continue
                colours[team].append(colour)
                if (p["bbox"][3] - p["bbox"][1] >= 26 and p["track_id"] not in candidates[team]
                        and _isolated(p, frame["players"])):
                    candidates[team][p["track_id"]] = (colour, _crop(img, p["bbox"]))
    cap.release()

    segment = frames[TEAMS_SEGMENT[0] : TEAMS_SEGMENT[1]]
    counts = Counter(p["team"] for f in segment for p in f["players"] if p["team"])
    teams = {}
    for t in TEAMS:
        if not colours[t]:
            continue
        # Most typical members first: torso colour closest to the cluster's median colour,
        # so one box covering two overlapping players is not shown as an example.
        centre = np.median(colours[t], axis=0)
        best = sorted(candidates[t].values(), key=lambda c: float(np.linalg.norm(c[0] - centre)))
        teams[t] = {"share": counts[t] / sum(counts.values()), "crops": []}
        for i, (_, image) in enumerate(best[: 1 if t == "referee" else 3]):
            cv2.imwrite(str(BUILD / "crops" / f"{t}_{i}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            teams[t]["crops"].append(f"build/crops/{t}_{i}.jpg")

    passed = next(e for e in timeline if e["type"] == "pass" and PASS_SEGMENT[0] <= e["frame_start"] < PASS_SEGMENT[1])
    thumb = frames[THUMB_FRAME]
    player = next(p for p in thumb["players"] if p["team"] == "A")
    frame_snippet = "\n".join([
        "{",
        f'  "frame_number": {thumb["frame_number"]},',
        f'  "timestamp": {thumb["timestamp"]},',
        '  "players": [',
        f'    {{"track_id": {player["track_id"]}, "team": "A",',
        f'     "bbox": {json.dumps(_r(player["bbox"]))}}},',
        f'    … {len(thumb["players"]) - 1} người nữa',
        "  ],",
        f'  "ball": {{"centroid": {json.dumps(_r(thumb["ball"]["centroid"]))},',
        f'           "state": "{thumb["ball"]["state"]}"}}',
        "}",
    ])
    i = timeline.index(passed)
    events_snippet = ["["]
    for e in timeline[max(0, i - 1) : i + 2]:
        events_snippet += [
            f'  {{"type": "{e["type"]}", "team": "{e["team"]}",',
            f'   "frame_start": {e["frame_start"]}, "actor_track_id": {e["actor_track_id"]}}},',
        ]
    events_snippet += ["  …", "]"]

    data = {
        "frames": compact,
        "timeline": [[e["frame_start"], e["type"], e["team"], e["actor_track_id"]] for e in timeline],
        "total": len(frames),
        "teams": teams,
        "stats": {
            "people": round(sum(len(f["players"]) for f in frames) / len(frames)),
            "ball": [sum(1 for f in frames if f["ball"] and f["ball"]["state"] == "in_play"), len(frames)],
        },
        "pass": [passed["frame_start"], passed["team"], passed["actor_track_id"]],
        "snippets": {
            "frame": frame_snippet,
            "events": "\n".join(events_snippet),
            "event": json.dumps(passed, indent=2, ensure_ascii=False),
        },
        "bg": f"build/frames/{BG_FRAME:04d}.jpg",
        "thumb": THUMB_FRAME,
    }
    (BUILD / "data.js").write_text(
        "window.DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8"
    )


def render(stills: list[float]) -> None:
    from imageio_ffmpeg import get_ffmpeg_exe
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto((HERE / "promo.html").as_uri() + "?capture")
        page.wait_for_function("window.ready === true")
        for t in stills:
            page.evaluate("t => render(t)", t)
            page.screenshot(path=str(BUILD / f"still_{t:05.2f}.png"))
        if not stills:
            total = round(page.evaluate("DURATION") * FPS)
            ffmpeg = subprocess.Popen(
                [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS),
                 "-c:v", "mjpeg", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                 "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709",
                 "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)],
                stdin=subprocess.PIPE,
            )
            for i in range(total):
                page.evaluate("t => render(t)", i / FPS)
                ffmpeg.stdin.write(page.screenshot(type="jpeg", quality=95))
            ffmpeg.stdin.close()
            if ffmpeg.wait():
                raise SystemExit("ffmpeg failed")
        browser.close()


def main() -> None:
    os.chdir(ROOT)  # config and model paths are relative to the project root
    if not (RUN / "timeline.json").exists():
        run_pipeline()
    prepare()
    stills = [float(t) for t in sys.argv[2:]] if sys.argv[1:2] == ["still"] else []
    render(stills)
    print(*([BUILD / f"still_{t:05.2f}.png" for t in stills] or [OUT]), sep="\n")


if __name__ == "__main__":
    main()
