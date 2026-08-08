"""Camera capture, tuned for latency rather than picture quality.

Frame rate is the gesture-latency budget. Every frame you do not get is another
33 ms your hand can move before the sound knows about it, and on a typical
laptop webcam the default settings throw most of them away:

**Auto-exposure is the main culprit.** In anything less than bright light the
driver lengthens the integration time to brighten the image, and frame rate
collapses with it -- measured on the development machine, 10 fps with auto
exposure against 30 fps with it pinned manually. Brightness you can fix with a
lamp; frames you cannot get back. So this module forces manual exposure by
default and lets you raise it if the picture is too dark to track.

**Capture runs on its own thread.** ``cap.read()`` blocks until the next frame
arrives. If the main loop then spends 12 ms on inference, it re-enters read()
after the following frame has already been and gone, and you silently run at
half rate. A background thread that keeps only the newest frame decouples the
two: inference that overruns skips a frame instead of halving the rate.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

# DirectShow's convention for CAP_PROP_AUTO_EXPOSURE.
DSHOW_EXPOSURE_MANUAL = 0.25
DSHOW_EXPOSURE_AUTO = 0.75

# Exposure is logarithmic here: -6 is about 1/64 s. Shorter is darker and
# faster; on the development camera -4 already halved the frame rate.
DEFAULT_EXPOSURE = -6.0


class CameraStream:
    """Background-threaded capture that only ever holds the newest frame."""

    def __init__(self, cap, backend: str, size: tuple[int, int]):
        self._cap = cap
        self.backend = backend
        self.size = size
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="live_pedal-capture")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read_latest(self, last_seq: int, timeout: float = 1.0):
        """Block until a frame newer than ``last_seq`` exists, then return it.

        Returns ``(frame, seq)`` or ``(None, last_seq)`` on timeout. Waiting for
        a *new* sequence number is what stops the caller burning CPU running
        inference twice on the same image.
        """
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._seq != last_seq and self._frame is not None:
                    return self._frame, self._seq
            time.sleep(0.001)
        return None, last_seq

    def measure_fps(self, seconds: float = 1.0) -> float:
        start_seq = self._seq
        t0 = time.perf_counter()
        time.sleep(seconds)
        return (self._seq - start_seq) / max(time.perf_counter() - t0, 1e-6)

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._cap.release()
        except Exception:
            pass


def open_camera(
    index: int,
    width: int,
    height: int,
    fps: int,
    exposure: float | str = DEFAULT_EXPOSURE,
) -> CameraStream:
    """Open a capture device and configure it for speed.

    ``exposure`` may be a number (manual, logarithmic, more negative = faster)
    or the string "auto" to leave the driver in charge.
    """
    backends = []
    if sys.platform == "win32":
        # DirectShow exposes exposure control reliably; MSMF often does not.
        backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]
    backends.append(("default", cv2.CAP_ANY))

    errors = []
    for label, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            errors.append(f"{label}: could not open device {index}")
            continue

        # MJPG where the camera supports it; many cheap sensors only do raw
        # YUY2 and will silently ignore this.
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        _apply_exposure(cap, backend, exposure)

        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            return CameraStream(cap, label, (w, h))

        cap.release()
        errors.append(f"{label}: opened but returned no frame")

    raise RuntimeError(
        f"could not start camera {index}.\n  "
        + "\n  ".join(errors)
        + "\nCheck that nothing else is using the webcam, and that camera "
          "access is allowed in Windows privacy settings."
    )


def _apply_exposure(cap, backend, exposure) -> None:
    if isinstance(exposure, str) and exposure.lower() == "auto":
        try:
            if backend == cv2.CAP_DSHOW:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, DSHOW_EXPOSURE_AUTO)
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        except Exception:
            pass
        return

    try:
        if backend == cv2.CAP_DSHOW:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, DSHOW_EXPOSURE_MANUAL)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
    except Exception:
        pass
