"""Entry point: python main.py"""

import sys
from pathlib import Path

from src.config import load_config
from src.logger import get_logger
from src.serve.pipeline.pipeline import Pipeline

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.yaml"
_log = get_logger(__name__)


def main() -> None:
    """Load config and run the pipeline on the configured video source."""
    config = load_config(CONFIG_PATH)
    source = config.video.source

    if not Path(source).exists():
        _log.error("video_not_found", path=source)
        sys.exit(1)

    _log.info("pipeline_start", source=source)
    pipeline = Pipeline(config)
    result = pipeline.run(source)
    _log.info("pipeline_done", total_frames=result.frame_count)


if __name__ == "__main__":
    main()
