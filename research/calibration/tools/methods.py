"""Registration methods under test. Each is a callable: BGR image -> image->pitch homography or None."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from fieldreg import ROOT, Pitch

PNL = ROOT / ".cache" / "calib" / "PnLCalib"  # GPL-2.0 reference, research only


class PnLCalib:
    """PnLCalib single-view keypoint + line HRNets, ground-plane homography (reference)."""

    def __init__(self, kp_th: float = 0.3434, line_th: float = 0.7867, refine: bool = True,
                 weights: str = "SV", device: str = "cuda:0") -> None:
        sys.path.insert(0, str(PNL))
        import yaml
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from utils import utils_calib, utils_heatmap

        self._calib, self._hm = utils_calib, utils_heatmap
        self.kp_th, self.line_th, self.refine, self.device = kp_th, line_th, refine, device
        self.model = get_cls_net(yaml.safe_load(open(PNL / "config" / "hrnetv2_w48.yaml")))
        self.model.load_state_dict(torch.load(PNL / "weights" / f"{weights}_kp", map_location=device))
        self.model_l = get_cls_net_l(yaml.safe_load(open(PNL / "config" / "hrnetv2_w48_l.yaml")))
        self.model_l.load_state_dict(torch.load(PNL / "weights" / f"{weights}_lines", map_location=device))
        self.model.to(device).eval()
        self.model_l.to(device).eval()

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray | None:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        hm, hm_l = self.model(x), self.model_l(x)
        kp = self._hm.get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1])
        ln = self._hm.get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1])
        kp_d = self._hm.coords_to_dict(kp, threshold=self.kp_th, ground_plane_only=True)
        ln_d = self._hm.coords_to_dict(ln, threshold=self.line_th, ground_plane_only=True)
        kp_d, ln_d = self._hm.complete_keypoints(kp_d[0], ln_d[0], w=960, h=540, normalize=True)
        cam = self._calib.FramebyFrameCalib(w, h, denormalize=True)
        cam.update(kp_d, ln_d)
        res = cam.heuristic_voting_ground(refine_lines=self.refine)
        if res is None:
            return None
        # SoccerNet ground frame is centred on the pitch; ours starts at the far-left corner.
        H = np.array([[1, 0, 52.5], [0, 1, 34.0], [0, 0, 1]]) @ res["homography"]
        return H / H[2, 2]


class YoloPitch:
    """Ultralytics pose model with one 'pitch' object and 32 Roboflow-layout keypoints."""

    def __init__(self, weights: str | Path, pitch: Pitch, imgsz: int = 640, kp_conf: float = 0.5,
                 ransac_px: float = 4.0, device: int | str = 0) -> None:
        from ultralytics import YOLO

        self.model, self.pitch, self.imgsz = YOLO(str(weights)), pitch, imgsz
        self.kp_conf, self.ransac_px, self.device = kp_conf, ransac_px, device
        self.last_points = 0

    def keypoints(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        r = self.model.predict(image, imgsz=self.imgsz, conf=0.25, device=self.device, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return None
        i = int(r.boxes.conf.argmax())
        return r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()

    def __call__(self, image: np.ndarray) -> np.ndarray | None:
        out = self.keypoints(image)
        self.last_points = 0
        if out is None:
            return None
        xy, conf = out
        sel = conf >= self.kp_conf
        self.last_points = int(sel.sum())
        if sel.sum() < 4:
            return None
        # Fit pitch -> image so the RANSAC threshold is in pixels, where keypoint noise lives.
        thr = self.ransac_px * image.shape[1] / 1280
        G, _ = cv2.findHomography(self.pitch.keypoints32()[sel], xy[sel].astype(np.float64), cv2.RANSAC, thr)
        if G is None or abs(np.linalg.det(G)) < 1e-12:
            return None
        H = np.linalg.inv(G)
        return H / H[2, 2]
