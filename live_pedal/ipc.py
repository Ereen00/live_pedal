"""Lock-free shared memory channels between the vision process and the audio process.

The audio callback runs on a real-time thread and must never block. Anything
that can wait -- a queue, a pipe, a mutex -- can stall it long enough to miss a
buffer, which you hear as a click. So the two processes talk through plain
shared memory using a seqlock: the writer brackets its update with an odd/even
counter, and the reader retries a bounded number of times and falls back to the
last good value rather than waiting.

Two channels, both ``VectorChannel``:

    targets   vision -> audio   desired value of every chain parameter
    display   audio  -> vision   smoothed values + telemetry, for the overlay

Note what does *not* cross the boundary: raw gesture features. The mapping from
hand position to parameter value happens on the vision side, so the audio thread
never does anything more than copy a block of floats. Mapping at camera rate
(30-60 Hz) loses nothing, because the parameter smoother in the audio thread
interpolates between updates anyway and the camera is the real rate limit.
"""

from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np

# ---------------------------------------------------------------------------
# Gesture feature vector
# ---------------------------------------------------------------------------
# These are the names you use on the left-hand side of a mapping in YAML.

FEATURE_NAMES: tuple[str, ...] = (
    "present",       # 1.0 while a tracked hand is visible
    "openness",      # 0 = tight fist, 1 = flat open palm
    "pinch",         # 0 = thumb and index touching, 1 = far apart
    "spread",        # 0 = fingers together, 1 = splayed
    "x",             # palm centre across the frame, 0 = left, 1 = right
    "y",             # palm centre up the frame, 0 = top, 1 = bottom
    "size",          # apparent hand size: a proxy for distance to the camera
    "roll",          # palm rotation, 0 = pointing left, 1 = pointing right
    "tilt",          # wrist-to-middle-finger axis, 0 = down, 1 = up
    "curl_thumb",
    "curl_index",
    "curl_middle",
    "curl_ring",
    "curl_pinky",
    "speed",         # normalised movement speed of the palm centre
    "pose_fist",
    "pose_open",
    "pose_point",
    "pose_peace",
    "pose_ok",
    "pose_thumbup",
    "fps",           # vision loop rate, diagnostics only
)

FEATURE_INDEX: dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)

# Flag bits carried in the target channel header. Set by the overlay window's
# keyboard shortcuts, read by the audio engine.
FLAG_BYPASS = 1 << 0     # pass the guitar through untouched
FLAG_QUIT = 1 << 1       # shut the whole rig down
FLAG_HOLD = 1 << 2       # freeze parameters where they are, ignore the hand
FLAG_PANIC = 1 << 3      # mute and clear all effect state
FLAG_TUNER = 1 << 4      # reserved

# Telemetry slots in the display channel header.
STAT_XRUNS = 2
STAT_CPU_PERMILLE = 3

_HEADER_WORDS = 4
_HEADER_BYTES = _HEADER_WORDS * 4


class VectorChannel:
    """A fixed-length numeric vector shared between two processes.

    Single writer, single reader. Create it in the parent with ``create=True``,
    pass ``.name`` to the child, and open it there with ``create=False``.
    """

    def __init__(
        self,
        n: int,
        dtype=np.float64,
        name: str | None = None,
        create: bool = False,
    ):
        self.n = max(int(n), 1)
        self.dtype = np.dtype(dtype)
        size = _HEADER_BYTES + self.n * self.dtype.itemsize

        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=size)
        else:
            if name is None:
                raise ValueError("name is required when create=False")
            self._shm = shared_memory.SharedMemory(name=name)

        self._header = np.ndarray((_HEADER_WORDS,), dtype=np.uint32, buffer=self._shm.buf)
        self._data = np.ndarray(
            (self.n,), dtype=self.dtype, buffer=self._shm.buf, offset=_HEADER_BYTES
        )
        if create:
            self._header[:] = 0
            self._data[:] = 0

        # Reader-side fallback so read_into() never allocates and never blocks.
        self._last_good = np.zeros(self.n, dtype=self.dtype)

    @property
    def name(self) -> str:
        return self._shm.name

    # -- writer side --------------------------------------------------------

    def write(self, values: np.ndarray) -> None:
        seq = int(self._header[0])
        self._header[0] = (seq + 1) & 0xFFFFFFFF        # odd: update in progress
        self._data[:] = values[: self.n]
        self._header[0] = (seq + 2) & 0xFFFFFFFF        # even: complete

    def seed(self, values: np.ndarray) -> None:
        """Set the initial contents without bumping the sequence counter."""
        self._data[:] = values[: self.n]
        self._last_good[:] = self._data

    # -- reader side --------------------------------------------------------

    def read_into(self, out: np.ndarray, retries: int = 3) -> bool:
        """Copy the latest vector into ``out``. Allocation-free, never blocks.

        Returns False if every attempt collided with a write in progress, in
        which case ``out`` holds the last cleanly read value.
        """
        for _ in range(retries):
            s1 = int(self._header[0])
            if s1 & 1:
                continue
            out[: self.n] = self._data
            if int(self._header[0]) == s1:
                self._last_good[:] = out[: self.n]
                return True
        out[: self.n] = self._last_good
        return False

    # -- flags and telemetry -----------------------------------------------

    @property
    def flags(self) -> int:
        return int(self._header[1])

    def set_flag(self, flag: int, on: bool = True) -> None:
        cur = int(self._header[1])
        self._header[1] = ((cur | flag) if on else (cur & ~flag)) & 0xFFFFFFFF

    def toggle_flag(self, flag: int) -> bool:
        new = int(self._header[1]) ^ flag
        self._header[1] = new & 0xFFFFFFFF
        return bool(new & flag)

    def clear_flag(self, flag: int) -> None:
        self.set_flag(flag, False)

    def set_stat(self, slot: int, value: int) -> None:
        if 2 <= slot < _HEADER_WORDS:
            self._header[slot] = int(value) & 0xFFFFFFFF

    def get_stat(self, slot: int) -> int:
        return int(self._header[slot]) if 2 <= slot < _HEADER_WORDS else 0

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        try:
            self._header = None      # type: ignore[assignment]
            self._data = None        # type: ignore[assignment]
            self._shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        try:
            self._shm.unlink()
        except Exception:
            pass
