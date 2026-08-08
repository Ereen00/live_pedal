"""Audio device discovery and selection.

Latency on Windows is decided almost entirely here, by which host API you end
up on. In descending order of how good they are for live playing:

    ASIO           2-8 ms round trip. What you want. Needs a vendor driver
                   (or ASIO4ALL as a generic wrapper).
    WDM-KS         5-15 ms. Kernel streaming, bypasses the Windows mixer.
    WASAPI         8-20 ms in exclusive mode, worse in shared mode.
    DirectSound    30 ms+. Not for playing through.
    MME            60 ms+. Unusable for a live instrument.

So the resolver prefers them in that order unless you name one explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd

# Ordered best-to-worst for live monitoring.
HOSTAPI_PREFERENCE = ("asio", "wdm-ks", "wasapi", "directsound", "mme")

_ALIASES = {
    "asio": ("asio",),
    "wdm-ks": ("windows wdm-ks", "wdm-ks", "wdmks"),
    "wasapi": ("windows wasapi", "wasapi"),
    "directsound": ("windows directsound", "directsound", "dsound"),
    "mme": ("mme",),
}

# Rough expectation per host API, used only to warn the user.
TYPICAL_RTT_MS = {
    "asio": "2-8 ms",
    "wdm-ks": "5-15 ms",
    "wasapi": "8-20 ms",
    "directsound": "30 ms+",
    "mme": "60 ms+",
}


@dataclass
class DeviceChoice:
    input_index: int
    output_index: int
    input_name: str
    output_name: str
    hostapi_name: str
    hostapi_key: str
    samplerate: int
    extra_settings: object | None = None

    def summary(self) -> str:
        return (
            f"host API : {self.hostapi_name}  (typical round trip "
            f"{TYPICAL_RTT_MS.get(self.hostapi_key, '?')})\n"
            f"input    : [{self.input_index}] {self.input_name}\n"
            f"output   : [{self.output_index}] {self.output_name}"
        )


def hostapi_key(name: str) -> str:
    low = name.lower()
    for key, aliases in _ALIASES.items():
        if any(a in low for a in aliases):
            return key
    return low


def available_hostapis() -> dict[str, int]:
    """Map canonical host API key -> PortAudio host API index."""
    out: dict[str, int] = {}
    for i, ha in enumerate(sd.query_hostapis()):
        if ha["devices"]:
            out.setdefault(hostapi_key(ha["name"]), i)
    return out


def _matches(dev: dict, spec) -> bool:
    if isinstance(spec, int):
        return False        # handled separately, by index
    return str(spec).lower() in dev["name"].lower()


def _candidates(hostapi_idx: int, want_input: bool) -> list[tuple[int, dict]]:
    devs = []
    for i, d in enumerate(sd.query_devices()):
        if d["hostapi"] != hostapi_idx:
            continue
        if want_input and d["max_input_channels"] < 1:
            continue
        if not want_input and d["max_output_channels"] < 1:
            continue
        devs.append((i, d))
    return devs


def resolve(
    input_spec=None,
    output_spec=None,
    hostapi: str = "auto",
    samplerate: int = 48000,
    wasapi_exclusive: bool = True,
) -> DeviceChoice:
    """Pick a concrete input/output device pair.

    ``input_spec``/``output_spec`` may be a PortAudio device index, a substring
    of the device name, or None to take the host API's default.
    """
    apis = available_hostapis()
    if not apis:
        raise RuntimeError("PortAudio reports no usable host APIs")

    if hostapi and hostapi != "auto":
        key = hostapi_key(hostapi)
        if key not in apis:
            raise RuntimeError(
                f"host API {hostapi!r} is not available. Available: "
                f"{sorted(apis)}."
                + (
                    "\nASIO is missing: install your interface's ASIO driver, "
                    "or ASIO4ALL as a generic fallback."
                    if key == "asio"
                    else ""
                )
            )
        order = [key]
    else:
        order = [k for k in HOSTAPI_PREFERENCE if k in apis]
        order += [k for k in apis if k not in order]

    errors: list[str] = []
    for key in order:
        idx = apis[key]
        ha = sd.query_hostapis(idx)
        try:
            in_i = _pick(input_spec, idx, want_input=True, default=ha["default_input_device"])
            out_i = _pick(output_spec, idx, want_input=False, default=ha["default_output_device"])
        except LookupError as exc:
            errors.append(f"  {ha['name']}: {exc}")
            continue

        din = sd.query_devices(in_i)
        dout = sd.query_devices(out_i)

        extra = None
        if key == "wasapi" and wasapi_exclusive:
            try:
                extra = (
                    sd.WasapiSettings(exclusive=True),
                    sd.WasapiSettings(exclusive=True),
                )
            except Exception:
                extra = None

        return DeviceChoice(
            input_index=in_i,
            output_index=out_i,
            input_name=din["name"],
            output_name=dout["name"],
            hostapi_name=ha["name"],
            hostapi_key=key,
            samplerate=samplerate,
            extra_settings=extra,
        )

    raise RuntimeError(
        "could not find a usable input/output pair.\n" + "\n".join(errors)
    )


def _pick(spec, hostapi_idx: int, want_input: bool, default: int) -> int:
    role = "input" if want_input else "output"

    if isinstance(spec, int):
        d = sd.query_devices(spec)
        ch = d["max_input_channels"] if want_input else d["max_output_channels"]
        if ch < 1:
            raise LookupError(f"device {spec} has no {role} channels")
        return spec

    cands = _candidates(hostapi_idx, want_input)
    if not cands:
        raise LookupError(f"no {role} devices on this host API")

    if spec:
        hits = [i for i, d in cands if _matches(d, spec)]
        if not hits:
            raise LookupError(
                f"no {role} device matching {spec!r} "
                f"(candidates: {[d['name'] for _, d in cands]})"
            )
        return hits[0]

    if default is not None and default >= 0:
        if any(i == default for i, _ in cands):
            return default
    return cands[0][0]


def check_samplerate(choice: DeviceChoice, samplerate: int, channels_in: int,
                     channels_out: int) -> bool:
    """Ask PortAudio whether this exact format will open, without opening it."""
    try:
        sd.check_input_settings(
            device=choice.input_index, channels=channels_in, samplerate=samplerate,
            extra_settings=choice.extra_settings[0] if choice.extra_settings else None,
        )
        sd.check_output_settings(
            device=choice.output_index, channels=channels_out, samplerate=samplerate,
            extra_settings=choice.extra_settings[1] if choice.extra_settings else None,
        )
        return True
    except Exception:
        return False


def format_device_table() -> str:
    lines = ["idx  host API           in out    default sr   low-latency in/out   name"]
    lines.append("-" * 100)
    for i, d in enumerate(sd.query_devices()):
        ha = sd.query_hostapis(d["hostapi"])["name"]
        lines.append(
            f"{i:>3}  {ha:<18} {d['max_input_channels']:>2} {d['max_output_channels']:>2}  "
            f"{d['default_samplerate']:>10.0f}   "
            f"{d['default_low_input_latency']*1000:>6.1f} / "
            f"{d['default_low_output_latency']*1000:<6.1f}ms  {d['name']}"
        )
    return "\n".join(lines)
