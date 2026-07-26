"""Entry point: python -m src"""

import sys
from pathlib import Path

import yaml

from src.controller.pipeline import Pipeline

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def main() -> None:
    """Load config and run the pipeline on the configured video source."""
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    source = config["video"]["source"]
    if not Path(source).exists():
        print(f"Video not found: {source}", file=sys.stderr)
        print("Update 'video.source' in config/default.yaml", file=sys.stderr)
        sys.exit(1)

    pipeline = Pipeline(config)
    pipeline.run(source)


if __name__ == "__main__":
    main()
