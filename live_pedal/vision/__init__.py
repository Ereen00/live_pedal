"""Camera capture, hand tracking, gesture features, and the preview window.

MediaPipe and its TensorFlow Lite backend log a lot of noise at import time and
then repeat a harmless NORM_RECT warning on every frame. While you are playing,
the console is a status display -- input level, DSP load, dropouts -- and that
warning scrolling past hides the numbers that matter. These environment
variables have to be set before the native libraries load, which is why they
live in the package __init__ rather than next to the import that needs them.
"""

from __future__ import annotations

import os

os.environ.setdefault("GLOG_minloglevel", "2")        # errors only
os.environ.setdefault("GLOG_logtostderr", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

try:  # absl keeps its own verbosity knob, independent of the env vars
    from absl import logging as _absl_logging

    _absl_logging.set_verbosity(_absl_logging.ERROR)
except Exception:
    pass
