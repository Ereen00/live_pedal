"""DSP effect registry.

``EFFECTS`` maps the name you write in a YAML chain to the class that
implements it. Adding an effect means writing the class and adding it here;
nothing else in the system needs to know about it.
"""

from __future__ import annotations

from .base import Effect, ParamSpec, db_to_lin
from .drive import Drive
from .filters import EQ3, Filter, Gate, Wah
from .modulation import Chorus, Flanger, Phaser, Tremolo, Vibrato
from .pitch import Octaver, PitchShift
from .time_fx import Delay, Reverb
from .utility import Tone, Volume

ALL_EFFECTS: tuple[type[Effect], ...] = (
    Gate,
    Volume,
    Octaver,
    Drive,
    Wah,
    Filter,
    EQ3,
    Tone,
    PitchShift,
    Phaser,
    Flanger,
    Chorus,
    Vibrato,
    Tremolo,
    Delay,
    Reverb,
)

EFFECTS: dict[str, type[Effect]] = {cls.kind: cls for cls in ALL_EFFECTS}

__all__ = ["EFFECTS", "ALL_EFFECTS", "Effect", "ParamSpec", "db_to_lin"]
