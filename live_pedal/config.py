"""Configuration loading.

One YAML file describes a whole rig: which devices to open, how the camera is
set up, what the effect chain looks like, and which gesture drives which
parameter. Presets in ``configs/`` are just complete versions of this file.

A preset may set ``extends: other.yaml`` to inherit and override, so you can
keep your device settings in one place and swap only the chain and mappings.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


@dataclass
class AudioConfig:
    samplerate: int = 48000
    blocksize: int = 256
    input_device: Any = None          # index, name substring, or None for auto
    output_device: Any = None
    hostapi: str = "auto"             # auto|asio|wasapi|wdm-ks|directsound|mme
    wasapi_exclusive: bool = True
    input_channel: int = 0            # which input channel the guitar is on
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    output_limit: float = 0.99        # hard ceiling applied before the DAC
    dry_monitor: float = 0.0          # blend of unprocessed signal into output
    latency: str = "low"              # PortAudio hint: "low"|"high"|seconds


@dataclass
class VisionConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 60
    # Manual exposure keeps the frame rate up; "auto" brightens the picture but
    # can cut the rate by two thirds indoors. More negative = faster and darker.
    exposure: float | str = -6.0
    hand: str = "any"                 # left|right|any -- which hand controls FX
    mirror: bool = True               # mirror the preview so it reads naturally
    show_window: bool = True
    model: str = "models/hand_landmarker.task"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smoothing: float = 0.35           # 0 = raw landmarks, ->1 = heavy smoothing
    release_ms: float = 250.0         # hold last values this long after losing the hand


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    chain: list[dict] = field(default_factory=list)
    mappings: list[dict] = field(default_factory=list)
    name: str = "default"
    source_path: Path | None = None


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path, _seen: set[Path] | None = None) -> dict:
    _seen = _seen or set()
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular 'extends' involving {path}")
    _seen.add(path)

    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")

    parent = data.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        data = _deep_merge(_load_raw(parent_path, _seen), data)
    return data


def _fill(cls, data: dict, where: str):
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"{where}: unknown option(s) {sorted(unknown)}; valid: {sorted(valid)}"
        )
    return cls(**data)


def load_config(path: str | Path | None = None) -> AppConfig:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        # Allow bare preset names: "crybaby" -> configs/crybaby.yaml
        if not p.exists() and not p.suffix:
            cand = CONFIG_DIR / f"{p}.yaml"
            if cand.exists():
                p = cand
        if not p.exists():
            cand = CONFIG_DIR / p
            if cand.exists():
                p = cand
    raw = _load_raw(Path(p))

    cfg = AppConfig(
        audio=_fill(AudioConfig, raw.get("audio", {}) or {}, "audio"),
        vision=_fill(VisionConfig, raw.get("vision", {}) or {}, "vision"),
        chain=list(raw.get("chain", []) or []),
        mappings=list(raw.get("mappings", []) or []),
        name=str(raw.get("name", Path(p).stem)),
        source_path=Path(p).resolve(),
    )
    if not cfg.chain:
        raise ValueError(f"{p}: 'chain' is empty -- there is nothing to process with")
    return cfg


def resolve_path(cfg: AppConfig, value: str) -> Path:
    """Resolve a path from the config relative to the repo root."""
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)
