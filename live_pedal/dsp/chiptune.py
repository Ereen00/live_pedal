"""8-bit / chiptune voicing for the guitar.

One effect, four stages, and deliberately nothing spatial anywhere in it. The
sound this is after is dry, close and low-resolution -- an Atari or NES voice
rather than a guitar in a room -- so there is no reverb, no delay and no width
here by design.

    square      replace the waveform with a pulse of the same amplitude
    crush       sample-and-hold plus bit-depth quantisation
    tone        one pole of lowpass, to keep the aliasing from getting shrill
    pluck       a short envelope restarted by every note

The order matters. Squaring first means the crusher is quantising a signal that
is already mostly two-valued, which is why the grit lands on the *edges* and the
decay rather than turning the whole note to mush. The pluck envelope goes last
so it shapes the finished tone instead of being crushed itself -- a quantised
envelope would step audibly on the way down.

``pluck`` is the part that makes it read as chiptune rather than as a broken
amplifier. A real console voice has no sustain to speak of: the note is struck
and gone. Multiplying by a short decay imposes that shape on a guitar note that
would otherwise ring for seconds.
"""

from __future__ import annotations

import numpy as np

from . import kernels as K
from .base import Effect, ParamSpec, db_to_lin

# The DC blocker's corner. An off-centre pulse width leaves a constant offset
# behind, and a constant offset eats headroom and thumps when the note stops.
#
# Very low on purpose. A highpass droops the flat top of a square wave and
# overshoots its edges, and how badly depends on the ratio between the corner
# and the note. A 25 Hz corner is inaudible as a *filter* but visibly rounds
# a low E at 82 Hz -- it undoes a good part of the squaring this effect exists
# to do. Six hertz costs nothing and leaves the shape alone.
DC_HZ = 6.0


class Chiptune(Effect):
    kind = "chiptune"

    PARAMS = (
        # --- pulse shaping -------------------------------------------------
        # square: 0 = the guitar untouched, 1 = a pure pulse wave. Anything in
        # between is a blend, which keeps some of the string's character.
        ParamSpec("square", 0.85, 0.0, 1.0, smooth_ms=30.0),
        # 0.5 is a symmetric square. Away from it the tone thins and gets
        # nasal -- the classic duty-cycle sound.
        ParamSpec("pulse_width", 0.5, 0.05, 0.95, smooth_ms=30.0),

        # --- resolution ----------------------------------------------------
        ParamSpec("bits", 4.0, 1.0, 16.0, smooth_ms=0.0, unit="bit"),
        ParamSpec("crush_hz", 6000.0, 500.0, 24000.0, smooth_ms=0.0,
                  unit="Hz", curve="log"),

        # --- tone ----------------------------------------------------------
        # Capped well below Nyquist: a one-pole's prewarped coefficient grows
        # without bound as the corner approaches it, and the filter starts
        # overshooting square edges instead of merely softening them.
        ParamSpec("tone", 7000.0, 500.0, 12000.0, smooth_ms=30.0,
                  unit="Hz", curve="log"),

        # --- pluck envelope -------------------------------------------------
        ParamSpec("pluck", 0.7, 0.0, 1.0, smooth_ms=30.0),
        ParamSpec("decay_ms", 400.0, 20.0, 3000.0, smooth_ms=0.0, unit="ms"),
        ParamSpec("attack_ms", 3.0, 0.2, 100.0, smooth_ms=0.0, unit="ms"),

        # --- what counts as a new note -------------------------------------
        ParamSpec("threshold_db", -40.0, -70.0, -6.0, smooth_ms=0.0, unit="dB"),
        # How much louder than the moment before counts as a new note. See
        # onset_detect for why anything much above 1.15 hears the first note
        # of a fast run and nothing after it.
        ParamSpec("sensitivity", 1.10, 1.02, 2.0, smooth_ms=0.0),
        ParamSpec("retrigger_ms", 70.0, 10.0, 1000.0, smooth_ms=0.0, unit="ms"),

        # --- output ---------------------------------------------------------
        ParamSpec("level", 1.0, 0.0, 4.0, smooth_ms=30.0),
        ParamSpec("mix", 1.0, 0.0, 1.0, smooth_ms=30.0),
    )

    # ------------------------------------------------------------------

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        self._w = np.zeros(block, dtype=np.float64)
        self._env = np.zeros(block, dtype=np.float64)
        self._crush_state = np.zeros(3, dtype=np.float64)
        self._onset_state = np.zeros(3, dtype=np.float64)
        self._pluck_state = np.zeros(2, dtype=np.float64)
        self._dc_state = np.zeros(1, dtype=np.float64)
        self._tone_state = np.zeros(1, dtype=np.float64)
        self._dc_g = self._tan_g(DC_HZ)
        # Start the pluck envelope open, so a first note that somehow misses
        # the onset detector is still heard rather than silently swallowed.
        self._pluck_state[1] = 1.0

    def reset(self) -> None:
        self._crush_state[:] = 0.0
        self._onset_state[:] = 0.0
        self._dc_state[:] = 0.0
        self._tone_state[:] = 0.0
        self._pluck_state[0] = 0.0
        self._pluck_state[1] = 1.0
        self._w[:] = 0.0
        self._env[:] = 0.0

    def _coef(self, ms: float) -> float:
        return float(1.0 - np.exp(-1.0 / max(self.sr * ms * 0.001, 1.0)))

    # ------------------------------------------------------------------

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        fired = K.onset_detect(
            x,
            db_to_lin(self._v("threshold_db")),
            self._coef(4.0),                    # fast envelope, 4 ms
            self._coef(25.0),                   # baseline, 25 ms
            self._v("sensitivity"),
            self._v("retrigger_ms") * 0.001 * self.sr,
            self._onset_state,
        )

        # --- pulse shaping and resolution reduction -------------------------
        sa, sb = self._ab("square")
        bits = min(max(self._v("bits"), 1.0), 16.0)
        # bits counts the whole word including sign, so the positive half has
        # 2**(bits-1) steps.
        levels = 2.0 ** (bits - 1.0)
        step = max(self.sr / max(self._v("crush_hz"), 20.0), 1.0)

        K.chip_crush(
            x, self._w, sa, sb, self._v("pulse_width"), levels, step,
            self._coef(1.0),        # envelope attack: fast enough to track a pick
            self._coef(30.0),       # release: slow enough not to buzz on decay
            self._crush_state,
        )

        K.svf_1pole(self._w, self._w, self._dc_g, self._dc_g, 1, self._dc_state)

        ta, tb = self._ab("tone")
        K.svf_1pole(self._w, self._w, self._tan_g(ta), self._tan_g(tb), 0,
                    self._tone_state)

        # --- pluck ----------------------------------------------------------
        atk = self._v("attack_ms") * 0.001 * self.sr
        dec = self._v("decay_ms") * 0.001 * self.sr
        # Exponential decay reaching -60 dB of its travel after decay_ms.
        dec_coef = float(np.exp(-6.9078 / max(dec, 1.0)))
        K.pluck_env(
            self._env, fired, 1.0 / max(atk, 1.0), dec_coef,
            1.0 - self._v("pluck"), self._pluck_state,
        )

        la, lb = self._ab("level")
        K.apply_gain_env(self._w, self._env, la, lb)

        ma, mb = self._ab("mix")
        K.crossfade_ramp(y, x, self._w, ma, mb)

    def warmup_hook(self, x: np.ndarray, y: np.ndarray) -> None:
        """The attack branch of the pluck envelope only runs on a note.

        Everything else here is unconditional, but ``pluck_env``'s stage 1 is
        reached only when the onset detector fires, and a kernel branch that
        first executes mid-performance is a dropout.
        """
        for stage, level in ((1.0, 0.0), (0.0, 1.0)):
            self._pluck_state[0] = stage
            self._pluck_state[1] = level
            for _ in range(2):
                self.process(x, y)
        self.reset()
