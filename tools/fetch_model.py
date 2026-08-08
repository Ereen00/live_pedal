"""Download the MediaPipe hand landmark model.

The model is ~8 MB of binary, so it is not committed to the repo.

    python tools/fetch_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "models" / "hand_landmarker.task"
URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main() -> int:
    if DEST.exists() and DEST.stat().st_size > 1_000_000:
        print(f"already present: {DEST} ({DEST.stat().st_size/1e6:.1f} MB)")
        return 0

    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}\n         -> {DEST}")
    try:
        tmp = DEST.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(DEST)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print("Download the URL above manually and save it to the path shown.",
              file=sys.stderr)
        return 1

    print(f"done: {DEST.stat().st_size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
