"""Find out which output device you can actually hear.

"No sound" is almost always a routing problem, not a processing problem: the
audio is going somewhere real, just not somewhere connected to your ears. An
audio interface appears to Windows as its own sound card, so sending audio to it
and then listening to your laptop speakers gets you silence with everything
working perfectly.

    python tools/test_output.py --sweep     # beep every output in turn
    python tools/test_output.py             # beep the configured output
    python tools/test_output.py --device 13 # beep one specific device

The sweep is the one that solves it. It plays a tone through each output device
in turn and prints which is which, so you just listen for the one you hear.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_pedal.audio import devices                           # noqa: E402
from live_pedal.config import load_config                      # noqa: E402

AMPLITUDE = 0.25          # comfortable, not startling


def make_tone(sr: int, seconds: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    tone = np.sin(2 * np.pi * freq * t)
    # Fade the ends so it does not click.
    fade = int(sr * 0.02)
    env = np.ones_like(tone)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (tone * env * AMPLITUDE).astype(np.float32)


def beep(device_index: int, seconds: float, freq: float) -> tuple[bool, str]:
    """Play a tone on one device. Returns (worked, message)."""
    try:
        info = sd.query_devices(device_index)
        sr = int(info["default_samplerate"])
        ch = max(1, min(int(info["max_output_channels"]), 2))
        tone = make_tone(sr, seconds, freq)
        data = np.tile(tone[:, None], (1, ch))
        sd.play(data, samplerate=sr, device=device_index, blocking=True)
        sd.stop()
        return True, "played"
    except Exception as exc:
        return False, str(exc).strip().splitlines()[0]


def sweep(seconds: float, freq: float) -> int:
    outs = [
        (i, d)
        for i, d in enumerate(sd.query_devices())
        if d["max_output_channels"] > 0
    ]
    if not outs:
        print("no output devices at all", file=sys.stderr)
        return 1

    print(f"Playing a {freq:.0f} Hz tone through each of {len(outs)} outputs.")
    print("Listen for the one you can hear, then note its index.\n")

    heard: list[int] = []
    for i, d in outs:
        api = sd.query_hostapis(d["hostapi"])["name"]
        print(f"  [{i:>2}] {api:<20} {d['name'][:48]:<48} ", end="", flush=True)
        ok, msg = beep(i, seconds, freq)
        print("ok" if ok else f"skipped ({msg[:40]})")
        if ok:
            heard.append(i)
        time.sleep(0.15)

    print(
        "\nWhichever index you heard is the one to put in your config:\n"
        "\n    audio:\n      output_device: <index>\n"
        "\nor pass it on the command line:\n"
        "\n    run.bat --output-device <index>\n"
    )
    print(
        "Note: input and output should ideally be the same physical device.\n"
        "Using your interface for input and the laptop speakers for output\n"
        "works, but the two run on independent clocks and will drift, which\n"
        "you may eventually hear as clicks. It is fine for testing."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="play a tone through every output device in turn")
    ap.add_argument("--device", default=None,
                    help="device index or name substring to test")
    ap.add_argument("--seconds", type=float, default=1.2)
    ap.add_argument("--freq", type=float, default=440.0)
    ap.add_argument("-c", "--config", default=None)
    args = ap.parse_args()

    if args.sweep:
        return sweep(args.seconds, args.freq)

    if args.device is not None:
        try:
            index = int(args.device)
        except ValueError:
            matches = [
                i for i, d in enumerate(sd.query_devices())
                if d["max_output_channels"] > 0
                and args.device.lower() in d["name"].lower()
            ]
            if not matches:
                print(f"no output device matching {args.device!r}", file=sys.stderr)
                return 1
            index = matches[0]
    else:
        cfg = load_config(args.config)
        choice = devices.resolve(
            input_spec=cfg.audio.input_device,
            output_spec=cfg.audio.output_device,
            hostapi=cfg.audio.hostapi,
            samplerate=cfg.audio.samplerate,
            wasapi_exclusive=cfg.audio.wasapi_exclusive,
        )
        print("This is where live_pedal is currently sending audio:\n")
        print(choice.summary())
        print()
        index = choice.output_index

    d = sd.query_devices(index)
    print(f"playing {args.freq:.0f} Hz for {args.seconds:.1f}s on "
          f"[{index}] {d['name']}")
    ok, msg = beep(index, args.seconds, args.freq)
    if not ok:
        print(f"failed: {msg}", file=sys.stderr)
        return 1

    print(
        "\nIf you heard that, this device works and your guitar path is fine.\n"
        "If you did not, the sound is going somewhere you are not listening to:\n"
        "  - plug headphones into the audio interface itself, or\n"
        "  - run:  python tools/test_output.py --sweep"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
