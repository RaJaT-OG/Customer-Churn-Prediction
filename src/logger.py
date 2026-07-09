import logging
import logging.config
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

config_path = PROJECT_ROOT / "config" / "logging.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

logging.config.dictConfig(config)

logger = logging.getLogger(__name__)
