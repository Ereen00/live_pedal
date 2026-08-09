#!/usr/bin/env python
"""live_pedal -- gesture-controlled guitar effects. Entry point.

    python run.py                          # default rig
    python run.py -c configs/wah.yaml      # a specific preset
    python run.py --list-devices           # what sound cards can I see?
    python run.py --no-window              # headless, lowest CPU
    python run.py --dry-run                # check config, do not open hardware

Two processes: this one owns the sound card, a child owns the camera. They share
memory and nothing else. See live_pedal/vision/process.py for why.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _preflight() -> None:
    """Fail with instructions rather than a traceback if deps are missing.

    The usual cause is running the system Python when the dependencies were
    installed into the project's virtualenv, so say so explicitly and give the
    exact command instead of making the user infer it from an ImportError.
    """
    required = {
        "numpy": "numpy",
        "numba": "numba",
        "sounddevice": "sounddevice",
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "yaml": "PyYAML",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if not missing:
        return

    venv_py = REPO / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = REPO / ".venv" / "bin" / "python"
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)

    print("live_pedal cannot start: missing " + ", ".join(missing), file=sys.stderr)
    print(f"\nrunning: {sys.executable}", file=sys.stderr)

    if venv_py.exists() and not in_venv:
        print(
            "\nThis project has a virtualenv with the dependencies already "
            "installed,\nbut you are running the system Python. Use one of:\n"
            "\n    .venv\\Scripts\\activate     (then: python run.py)"
            "\n    .venv\\Scripts\\python.exe run.py"
            "\n    run.bat                     (picks the right one for you)",
            file=sys.stderr,
        )
    else:
        print(
            "\nInstall the dependencies:\n"
            "\n    python -m venv .venv"
            "\n    .venv\\Scripts\\activate"
            "\n    pip install -r requirements.txt"
            "\n    python tools/fetch_model.py",
            file=sys.stderr,
        )
    raise SystemExit(1)


_preflight()

from live_pedal.config import load_config, resolve_path       # noqa: E402
from live_pedal.ipc import FLAG_QUIT, STAT_XRUNS, VectorChannel  # noqa: E402
from live_pedal.mapping import Mapper, MappingError           # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Gesture-controlled real-time guitar effects processor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-c", "--config", default=None,
                   help="preset to load (path, or a name in configs/)")
    p.add_argument("--list-devices", action="store_true",
                   help="print every audio device PortAudio can see, then exit")
    p.add_argument("--no-window", action="store_true",
                   help="run without the camera preview (lower CPU)")
    p.add_argument("--no-vision", action="store_true",
                   help="audio only: run the chain at its configured defaults")
    p.add_argument("--dry-run", action="store_true",
                   help="validate config and print the rig, without opening hardware")
    p.add_argument("--hostapi", default=None,
                   help="force a host API: asio, wdm-ks, wasapi, directsound, mme")
    p.add_argument("--blocksize", type=int, default=None,
                   help="override block size in samples (lower = less latency)")
    p.add_argument("--samplerate", type=int, default=None)
    p.add_argument("--input-device", default=None,
                   help="device index or a substring of its name")
    p.add_argument("--output-device", default=None)
    p.add_argument("--camera", type=int, default=None, help="camera index")
    p.add_argument("--seconds", type=float, default=None,
                   help="stop automatically after N seconds (for soak testing)")
    return p.parse_args(argv)


def _maybe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def apply_overrides(cfg, args) -> None:
    if args.hostapi:
        cfg.audio.hostapi = args.hostapi
    if args.blocksize:
        cfg.audio.blocksize = args.blocksize
    if args.samplerate:
        cfg.audio.samplerate = args.samplerate
    if args.input_device is not None:
        cfg.audio.input_device = _maybe_int(args.input_device)
    if args.output_device is not None:
        cfg.audio.output_device = _maybe_int(args.output_device)
    if args.camera is not None:
        cfg.vision.camera_index = args.camera
    if args.no_window:
        cfg.vision.show_window = False


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_devices:
        from live_pedal.audio import devices
        print(devices.format_device_table())
        return 0

    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    apply_overrides(cfg, args)

    print(f"live_pedal -- preset '{cfg.name}'  ({cfg.source_path})")
    print()

    # ------------------------------------------------------------------
    # Dry run: build the chain offline and validate the mappings.
    # ------------------------------------------------------------------
    if args.dry_run:
        from live_pedal.audio.chain import Chain
        try:
            chain = Chain.from_spec(cfg.chain, cfg.audio.samplerate,
                                    cfg.audio.blocksize)
        except Exception as exc:
            print(f"chain error: {exc}", file=sys.stderr)
            return 2
        layout = chain.layout()
        try:
            mapper = Mapper(cfg.mappings, layout)
        except MappingError as exc:
            print(f"mapping error: {exc}", file=sys.stderr)
            return 2
        print("chain:")
        print(chain.describe())
        for fx in chain.effects:
            if getattr(fx, "chord_names", None):
                print(f"\nchords ({fx.name}):")
                print(fx.describe_chords())
        print("\nmappings:")
        print(mapper.describe())
        for w in mapper.warnings:
            print(f"\nwarning: {w}")
        print("\nConfig is valid. No hardware was opened.")
        return 0

    # ------------------------------------------------------------------
    # Build the engine (this opens and probes the sound card).
    # ------------------------------------------------------------------
    from live_pedal.audio.engine import Engine

    try:
        engine = Engine(cfg.audio, cfg.chain)
    except Exception as exc:
        print(f"audio error: {exc}", file=sys.stderr)
        return 3

    layout = engine.chain.layout()
    n_params = engine.chain.n_params

    # Validate mappings before spawning anything, so errors surface here.
    if not args.no_vision:
        try:
            for w in Mapper(cfg.mappings, layout).warnings:
                print(f"warning: {w}\n", file=sys.stderr)
        except MappingError as exc:
            print(f"mapping error: {exc}", file=sys.stderr)
            return 2

    # ------------------------------------------------------------------
    # Shared memory + vision process
    # ------------------------------------------------------------------
    target_chan = VectorChannel(n_params, np.float64, create=True)
    display_chan = VectorChannel(n_params, np.float64, create=True)
    target_chan.seed(np.array(layout.defaults, dtype=np.float64))
    display_chan.seed(np.array(layout.defaults, dtype=np.float64))

    engine.target_channel = target_chan
    engine.display_channel = display_chan

    vision_proc = None
    try:
        if not args.no_vision:
            from live_pedal.vision.process import run_vision

            model = resolve_path(cfg, cfg.vision.model)
            if not model.exists():
                print(
                    f"hand model missing: {model}\n"
                    f"Run:  python tools/fetch_model.py",
                    file=sys.stderr,
                )
                return 4

            # Chord names live with the effect, but the label has to be drawn
            # by the process that owns the window. Ship the plain strings.
            chord_labels = {
                f"{fx.name}.chord": list(fx.chord_names)
                for fx in engine.chain.effects
                if getattr(fx, "chord_names", None)
            }

            vision_proc = mp.Process(
                target=run_vision,
                args=(cfg.vision, layout, cfg.mappings,
                      target_chan.name, display_chan.name, str(model),
                      chord_labels),
                daemon=True,
                name="live_pedal-vision",
            )
            vision_proc.start()

        engine.start()
        print(engine.summary())
        print()
        if args.no_vision:
            print("running audio only (no gesture control)")
        print("Ctrl+C here, or 'q' in the camera window, to stop.\n")

        _status_loop(engine, target_chan, vision_proc, args.seconds)

    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        target_chan.set_flag(FLAG_QUIT)
        engine.stop()
        if vision_proc is not None:
            vision_proc.join(timeout=3.0)
            if vision_proc.is_alive():
                vision_proc.terminate()
        target_chan.close()
        target_chan.unlink()
        display_chan.close()
        display_chan.unlink()

    s = engine.status
    budget_us = engine.blocksize / engine.samplerate * 1e6
    print(f"\n{s.blocks} blocks processed, {s.xruns} dropout(s), "
          f"worst callback {s.worst_us:.0f} us "
          f"({s.worst_us / budget_us * 100:.1f}% of the {budget_us:.0f} us budget)")
    return 0


def _status_loop(engine, target_chan, vision_proc, seconds=None) -> None:
    budget_us = engine.blocksize / engine.samplerate * 1e6
    last = 0.0
    deadline = time.time() + seconds if seconds else None
    while True:
        if target_chan.flags & FLAG_QUIT:
            break
        if deadline and time.time() >= deadline:
            print("\nreached --seconds limit; stopping.")
            break
        if vision_proc is not None and not vision_proc.is_alive():
            print("\nvision process exited; stopping.")
            break
        if not engine.running:
            print("\nthe audio driver stopped the stream; stopping.")
            break

        now = time.time()
        if now - last >= 0.5:
            last = now
            s = engine.status
            cpu = s.last_us / budget_us * 100.0
            bar_in = _meter(s.peak_in)
            bar_out = _meter(s.peak_out)
            sys.stdout.write(
                f"\r in {bar_in} {_dbfs(s.peak_in):>6}  "
                f"out {bar_out} {_dbfs(s.peak_out):>6}  "
                f"dsp {cpu:4.1f}%  xruns {s.xruns}   "
            )
            sys.stdout.flush()
        time.sleep(0.05)
    print()


def _meter(peak: float, width: int = 16) -> str:
    filled = int(min(max(peak, 0.0), 1.0) ** 0.5 * width)
    return "#" * filled + "-" * (width - filled)


def _dbfs(peak: float) -> str:
    if peak < 1e-6:
        return "--"
    return f"{20.0 * np.log10(peak):.0f}dB"


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
