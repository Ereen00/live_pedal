"""List audio devices and say what latency each host API can realistically give.

    python tools/list_devices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sounddevice as sd                                       # noqa: E402

from live_pedal.audio import devices                           # noqa: E402


def main() -> int:
    print(f"PortAudio: {sd.get_portaudio_version()[1]}\n")
    print(devices.format_device_table())

    apis = devices.available_hostapis()
    print("\nhost APIs present, best first:")
    ranked = [k for k in devices.HOSTAPI_PREFERENCE if k in apis]
    ranked += [k for k in apis if k not in ranked]
    for key in ranked:
        marker = "->" if key == ranked[0] else "  "
        print(f"  {marker} {key:<12} typical round trip "
              f"{devices.TYPICAL_RTT_MS.get(key, '?')}")

    if "asio" not in apis:
        print(
            "\nASIO is not available on this machine.\n"
            "  That is the single biggest thing standing between you and low\n"
            "  latency. Install your interface's own ASIO driver if it has one,\n"
            "  otherwise ASIO4ALL (https://asio4all.org) wraps almost any\n"
            "  Windows device and usually gets you into single-digit ms."
        )
    else:
        print("\nASIO is available -- use it: hostapi: asio")

    print("\nWhat live_pedal would pick with the current default config:")
    try:
        choice = devices.resolve(
            input_spec="USB AUDIO", output_spec="USB AUDIO", hostapi="auto"
        )
        print("  " + choice.summary().replace("\n", "\n  "))
    except Exception as exc:
        print(f"  (no match for 'USB AUDIO': {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
