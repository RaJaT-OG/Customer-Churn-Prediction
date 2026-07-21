"""
Auto-generated background retraining runner.
Regenerated automatically each time retraining is triggered from the app —
do not edit by hand, changes will be overwritten.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
STATUS_FILE = PROJECT_ROOT / "retrain_status.json"
LOG_FILE = PROJECT_ROOT / "retrain_log.txt"


def write_status(data):
    STATUS_FILE.write_text(json.dumps(data))


def main():
    started = time.time()
    write_status({"status": "running", "started": started})
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.unified_churn_pipeline"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        LOG_FILE.write_text((result.stdout or "") + "\n" + (result.stderr or ""))
        write_status(
            {
                "status": "success" if result.returncode == 0 else "failed",
                "started": started,
                "finished": time.time(),
                "returncode": result.returncode,
            }
        )
    except Exception as exc:  # runner itself crashed
        LOG_FILE.write_text(f"Runner crashed before/while launching training: {exc}")
        write_status(
            {
                "status": "failed",
                "started": started,
                "finished": time.time(),
                "returncode": -1,
                "error": str(exc),
            }
        )


if __name__ == "__main__":
    main()
