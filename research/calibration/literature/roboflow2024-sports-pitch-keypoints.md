# Roboflow `sports`: pitch keypoint detection with YOLOv8-pose
- Authors: Roboflow (Piotr Skalski et al.)
- Year: 2024
- URLs: https://github.com/roboflow/sports (MIT) · https://blog.roboflow.com/camera-calibration-sports-computer-vision · dataset https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi (mirrored on Hugging Face as `martinjolif/football-pitch-detection`)

## Key findings
- One "pitch" object with 32 named keypoints (corners, box corners, centre-circle points, penalty spots…), YOLOv8x-pose at 1280 px.
- Homography from confident keypoints with `cv2.findHomography` (the `ViewTransformer`); used for a top-down "radar" view.
- Pretrained `football-pitch-detection.pt` downloadable from Google Drive via `setup.sh`.

## Relevance
Closest to this codebase: same Ultralytics stack, so it exports to ONNX/TensorRT like our detectors. `notebooks/training/03_train_pitch_detector.ipynb` already targets this 32-keypoint format but was never run. No published accuracy in metres.
