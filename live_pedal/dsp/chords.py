"""Gesture-selected chord pad, triggered by playing a note.

This is not a background drone. Nothing sounds until you actually strike a
string; the pad then swells in underneath what you played, holds while the note
rings, and fades out slowly when you stop. A hand gesture chooses *which* chord
is armed, and that choice sticks until you make a different gesture.

The three parts:

**Onset detection** compares a fast envelope of the guitar against a slow one.
Detecting the *rise* rather than the level means a note played while the pad is
still decaying retriggers it, instead of only the first note out of silence.

**The oscillator bank** is a set of band-limited (PolyBLEP) oscillators, one per
chord tone times the unison count. Phases persist across chord changes, so
swapping the chord rewrites the frequencies without a click.

**The envelope** is started by the onset and held up by the guitar still
ringing. When the guitar goes quiet, an exponential release takes over -- long
by default, because the whole point is that the chord thins out rather than
being switched off.
"""

from __future__ import annotations

import re

import numpy as np

from . import kernels as K
from .base import Effect, ParamSpec, db_to_lin

# ---------------------------------------------------------------------------
# Note names
# ---------------------------------------------------------------------------

_SEMITONE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]*)(-?\d+)$")

# Scientific pitch notation: C4 = MIDI 60, A4 = 69 = 440 Hz.
MAX_NOTES = 8
MAX_UNISON = 5


def note_to_midi(name) -> int:
    """'E2' -> 40, 'G#3' -> 56, 'Ab3' -> 56. Plain integers pass through."""
    if isinstance(name, (int, float)):
        return int(name)
    text = str(name).strip()
    m = _NOTE_RE.match(text)
    if not m:
        raise ValueError(
            f"cannot read note {name!r}. Use scientific pitch notation like "
            f"E2, G#3, Ab3, or a raw MIDI number."
        )
    letter, accidentals, octave = m.groups()
    value = _SEMITONE[letter.lower()]
    for a in accidentals:
        value += 1 if a == "#" else -1
    return value + (int(octave) + 1) * 12


def midi_to_hz(midi: float) -> float:
    return float(440.0 * 2.0 ** ((midi - 69.0) / 12.0))


DEFAULT_CHORDS = [
    {"name": "Em", "notes": ["E2", "B2", "E3", "G3", "B3"]},
    {"name": "E", "notes": ["E2", "B2", "E3", "G#3", "B3"]},
    {"name": "Fm", "notes": ["F2", "C3", "F3", "Ab3", "C4"]},
    {"name": "B7b9", "notes": ["B2", "D#3", "F#3", "A3", "C4"]},
]


class ChordPad(Effect):
    kind = "chordpad"
    SETUP_KEYS = ("chords",)

    PARAMS = (
        # Which chord is armed. Discrete: never interpolate between chords.
        ParamSpec("chord", 0.0, 0.0, 15.0, smooth_ms=0.0),
        ParamSpec("level", 0.35, 0.0, 1.5, smooth_ms=40.0),

        # Envelope. The long release is the character of the whole thing.
        ParamSpec("attack_ms", 70.0, 1.0, 3000.0, smooth_ms=0.0, unit="ms"),
        ParamSpec("decay_ms", 400.0, 10.0, 4000.0, smooth_ms=0.0, unit="ms"),
        ParamSpec("sustain", 0.75, 0.0, 1.0, smooth_ms=0.0),
        ParamSpec("release_ms", 1800.0, 50.0, 12000.0, smooth_ms=0.0, unit="ms"),

        # Triggering.
        ParamSpec("threshold_db", -34.0, -70.0, -6.0, smooth_ms=0.0, unit="dB"),
        ParamSpec("sensitivity", 1.6, 1.05, 6.0, smooth_ms=0.0),
        ParamSpec("retrigger_ms", 180.0, 20.0, 2000.0, smooth_ms=0.0, unit="ms"),
        ParamSpec("hold_db", -46.0, -80.0, -12.0, smooth_ms=0.0, unit="dB"),

        # Tone.
        ParamSpec("cutoff", 2200.0, 150.0, 14000.0, smooth_ms=30.0,
                  unit="Hz", curve="log"),
        ParamSpec("resonance", 1.1, 0.5, 12.0, smooth_ms=30.0),
        ParamSpec("wave", 0.0, 0.0, 3.0, smooth_ms=0.0),
        ParamSpec("detune", 9.0, 0.0, 40.0, smooth_ms=0.0, unit="cents"),
        ParamSpec("unison", 3.0, 1.0, float(MAX_UNISON), smooth_ms=0.0),
        ParamSpec("octave", 0.0, -2.0, 2.0, smooth_ms=0.0),

        # 1 = a new gesture takes effect at the next note you play, so the
        # chord never changes underneath a ringing pad. 0 = change immediately.
        ParamSpec("change_on_trigger", 1.0, 0.0, 1.0, smooth_ms=0.0),
    )

    def __init__(self, name=None, setup=None, **overrides):
        super().__init__(name, setup, **overrides)
        spec = self.setup.get("chords") or DEFAULT_CHORDS
        self.chords, self.chord_names = self._parse_chords(spec)

    @staticmethod
    def _parse_chords(spec) -> tuple[list[list[int]], list[str]]:
        if not isinstance(spec, list) or not spec:
            raise ValueError("chordpad: 'chords' must be a non-empty list")
        chords: list[list[int]] = []
        names: list[str] = []
        for i, entry in enumerate(spec):
            if isinstance(entry, dict):
                notes = entry.get("notes")
                label = str(entry.get("name", f"chord{i}"))
            else:
                notes, label = entry, f"chord{i}"
            if not notes:
                raise ValueError(f"chordpad: chord #{i + 1} has no notes")
            if len(notes) > MAX_NOTES:
                raise ValueError(
                    f"chordpad: chord {label!r} has {len(notes)} notes; "
                    f"the maximum is {MAX_NOTES}"
                )
            chords.append([note_to_midi(n) for n in notes])
            names.append(label)
        return chords, names

    # ------------------------------------------------------------------

    def prepare(self, sr: int, block: int) -> None:
        super().prepare(sr, block)
        n_max = MAX_NOTES * MAX_UNISON
        self._phases = np.zeros(n_max, dtype=np.float64)
        self._incs = np.zeros(n_max, dtype=np.float64)
        # Spread the starting phases so all the voices do not punch in together
        # on the first cycle, which would make the attack thump.
        self._phases[:] = np.linspace(0.0, 0.97, n_max)

        self._synth = np.zeros(block, dtype=np.float64)
        self._env = np.zeros(block, dtype=np.float64)
        self._env_state = np.zeros(2, dtype=np.float64)
        self._onset_state = np.zeros(3, dtype=np.float64)
        self._filter_state = np.zeros(2, dtype=np.float64)
        self._level_state = np.zeros(1, dtype=np.float64)

        self._n_voices = 0
        self._active_chord = -1
        self._voicing = None
        self._rebuild(0)

    def reset(self) -> None:
        self._env_state[:] = 0.0
        self._onset_state[:] = 0.0
        self._filter_state[:] = 0.0
        self._level_state[:] = 0.0
        self._synth[:] = 0.0
        self._env[:] = 0.0

    # ------------------------------------------------------------------

    def _rebuild(self, index: int) -> None:
        """Recompute oscillator frequencies for a chord. Phases are untouched."""
        index = min(max(index, 0), len(self.chords) - 1)
        notes = self.chords[index]
        unison = int(min(max(self._v("unison"), 1), MAX_UNISON))
        detune = self._v("detune")
        octave = int(round(self._v("octave")))

        v = 0
        for midi in notes:
            base = midi + 12 * octave
            for u in range(unison):
                # Symmetric detune spread: 1 voice is dead centre, 3 gives
                # -d, 0, +d, and so on.
                offset = 0.0 if unison == 1 else (u / (unison - 1) - 0.5) * 2.0
                hz = midi_to_hz(base) * 2.0 ** (offset * detune / 1200.0)
                self._incs[v] = min(hz / self.sr, 0.45)
                v += 1
        self._n_voices = v
        self._active_chord = index
        self._voicing = (unison, detune, octave)

    def _coef(self, ms: float) -> float:
        """Per-sample coefficient for a time constant in milliseconds."""
        return float(1.0 - np.exp(-1.0 / max(self.sr * ms * 0.001, 1.0)))

    def process(self, x: np.ndarray, y: np.ndarray) -> None:
        # --- did the player just strike a note? ---------------------------
        fired = K.onset_detect(
            x,
            db_to_lin(self._v("threshold_db")),
            self._coef(4.0),                       # fast envelope, 4 ms
            self._coef(180.0),                     # slow envelope, 180 ms
            self._v("sensitivity"),
            self._v("retrigger_ms") * 0.001 * self.sr,
            self._onset_state,
        )

        # --- has a different chord been armed by a gesture? ----------------
        wanted = int(min(max(round(self._v("chord")), 0), len(self.chords) - 1))
        immediate = self._v("change_on_trigger") < 0.5
        voicing = (
            int(min(max(self._v("unison"), 1), MAX_UNISON)),
            self._v("detune"),
            int(round(self._v("octave"))),
        )
        # Waiting for the next note means a gesture never changes the chord
        # underneath a pad that is still ringing.
        if voicing != self._voicing or (
            wanted != self._active_chord and (fired > 0.5 or immediate)
        ):
            self._rebuild(wanted if (fired > 0.5 or immediate) else self._active_chord)

        # --- nothing sounding and nothing triggering: do no work -----------
        # Checked *before* running the envelope, because if the release had
        # finished mid-block the tail would still be in this block's envelope
        # and skipping it would chop the end off with a click.
        if self._env_state[0] == 0.0 and fired < 0.5:
            y[:] = x
            return

        # --- envelope ------------------------------------------------------
        # The slow envelope is the honest measure of "is the guitar still
        # ringing"; the fast one dips between picks and would drop the pad.
        gate_on = 1.0 if self._onset_state[1] > db_to_lin(self._v("hold_db")) else 0.0

        atk = self._v("attack_ms") * 0.001 * self.sr
        dec = self._v("decay_ms") * 0.001 * self.sr
        sustain = self._v("sustain")
        rel_ms = self._v("release_ms")
        # Exponential release reaching -60 dB after release_ms.
        rel_coef = float(np.exp(-6.9078 / max(rel_ms * 0.001 * self.sr, 1.0)))

        K.pad_env(
            self._env, fired, gate_on,
            1.0 / max(atk, 1.0),
            max(1.0 - sustain, 0.0) / max(dec, 1.0),
            sustain, rel_coef, self._env_state,
        )

        K.osc_bank(
            self._synth, self._phases, self._incs, self._n_voices,
            int(min(max(self._v("wave"), 0), 3)),
            1.0 / max(self._n_voices, 1),
        )

        ca, cb = self._ab("cutoff")
        ra, rb = self._ab("resonance")
        K.svf(
            self._synth, self._synth,
            self._tan_g(ca), self._tan_g(cb),
            1.0 / max(ra, 0.5), 1.0 / max(rb, 0.5),
            0, self._filter_state,
        )

        la, lb = self._ab("level")
        K.mix_env(y, x, self._synth, self._env, la, lb)

    def warmup_hook(self, x: np.ndarray, y: np.ndarray) -> None:
        """Drive the synthesis path even though no note has been played.

        Without this the pad stays in its idle branch for the whole warmup, and
        the oscillator bank, envelope and output mixer compile on the first note
        you play -- a guaranteed dropout at the worst possible moment.
        """
        for stage in (1.0, 2.0, 3.0, 4.0):
            self._env_state[0] = stage
            self._env_state[1] = 0.5
            for _ in range(2):
                self.process(x, y)
        for wave in range(4):
            self._env_state[0] = 3.0
            self._env_state[1] = 0.5
            saved = float(self.pb[self._idx["wave"]])
            self.pb[self._idx["wave"]] = float(wave)
            self.process(x, y)
            self.pb[self._idx["wave"]] = saved
        self.reset()
        self._rebuild(0)

    # ------------------------------------------------------------------

    def describe_chords(self) -> str:
        lines = []
        for i, (label, notes) in enumerate(zip(self.chord_names, self.chords)):
            spelled = " ".join(_midi_name(m) for m in notes)
            lines.append(f"    {i}: {label:<8} {spelled}")
        return "\n".join(lines)


_NAMES_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _midi_name(midi: int) -> str:
    return f"{_NAMES_SHARP[midi % 12]}{midi // 12 - 1}"
