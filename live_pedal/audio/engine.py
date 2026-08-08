"""The real-time audio engine.

Everything in the callback is written to the same rule: **do not allocate**.
Python's garbage collector can pause any thread that allocates, and a pause
longer than one block period is an audible dropout. So the callback only ever
writes into preallocated arrays, using ``out=`` forms of the numpy ufuncs, and
the chain does the same.

The engine also freezes and disables the GC while the stream is running. During
streaming the main thread does nothing but wait, and the callback produces no
garbage, so there is nothing for a collection to reclaim -- only latency to
lose by running one.
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from ..config import AudioConfig
from ..dsp.kernels import peak_level, sanitise
from ..ipc import FLAG_BYPASS, FLAG_HOLD, FLAG_PANIC, STAT_CPU_PERMILLE, STAT_XRUNS
from ..ipc import VectorChannel
from . import devices
from .chain import Chain


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


@dataclass
class EngineStatus:
    xruns: int = 0
    blocks: int = 0
    peak_in: float = 0.0
    peak_out: float = 0.0
    worst_us: float = 0.0        # steady state only, see SETTLE_BLOCKS
    first_us: float = 0.0        # cost of the very first callback
    last_us: float = 0.0


# The first few callbacks pay one-off costs no amount of warmup removes:
# the driver's own thread starting, first touch of freshly mapped pages, the
# Python thread state being created. They are not representative of what the
# engine does while you are playing, so they are reported separately rather
# than allowed to poison the worst-case figure.
SETTLE_BLOCKS = 4


class Engine:
    """Owns the sound card stream and the effect chain."""

    def __init__(
        self,
        cfg: AudioConfig,
        chain_spec: list[dict],
        target_channel: VectorChannel | None = None,
        display_channel: VectorChannel | None = None,
        verbose: bool = True,
    ):
        self.cfg = cfg
        self.verbose = verbose
        self.status = EngineStatus()
        self._targets_in = None
        self.target_channel = target_channel
        self.display_channel = display_channel
        self._gc_was_enabled = gc.isenabled()

        # --- pick devices --------------------------------------------------
        self.choice = devices.resolve(
            input_spec=cfg.input_device,
            output_spec=cfg.output_device,
            hostapi=cfg.hostapi,
            samplerate=cfg.samplerate,
            wasapi_exclusive=cfg.wasapi_exclusive,
        )

        din = sd.query_devices(self.choice.input_index)
        dout = sd.query_devices(self.choice.output_index)
        self.in_channels = max(1, min(int(din["max_input_channels"]), 2))
        self.out_channels = max(1, min(int(dout["max_output_channels"]), 2))
        self.input_channel = min(max(cfg.input_channel, 0), self.in_channels - 1)

        # --- agree on a sample rate ---------------------------------------
        self.samplerate = self._negotiate_samplerate(cfg.samplerate, din, dout)

        # --- agree on a block size ----------------------------------------
        # Some host APIs (WASAPI exclusive in particular) impose their own
        # period size regardless of what we ask for. Building the chain around
        # the wrong size would mean either silence or an allocation in the
        # callback, so find out for real before committing.
        self.blocksize = self._probe_blocksize(cfg.blocksize)

        # --- build the chain ----------------------------------------------
        self.chain = Chain.from_spec(chain_spec, self.samplerate, self.blocksize)

        n = self.blocksize
        self._x = np.zeros(n, dtype=np.float64)
        self._y = np.zeros(n, dtype=np.float64)
        self._dry = np.zeros(n, dtype=np.float64)
        self._out = np.zeros(n, dtype=np.float32)
        self._targets_in = np.zeros(self.chain.n_params, dtype=np.float64)

        self._in_gain = db_to_lin(cfg.input_gain_db)
        self._out_gain = db_to_lin(cfg.output_gain_db)
        self._limit = float(cfg.output_limit)
        self._dry_mix = float(cfg.dry_monitor)

        self._stream: sd.Stream | None = None
        self._publish_every = max(1, int(self.samplerate / self.blocksize / 25))
        self._panic_seen = False

    # ------------------------------------------------------------------
    # Negotiation
    # ------------------------------------------------------------------

    def _negotiate_samplerate(self, wanted: int, din: dict, dout: dict) -> int:
        candidates = [wanted, int(din["default_samplerate"]),
                      int(dout["default_samplerate"]), 48000, 44100]
        seen = []
        for sr in candidates:
            if sr in seen:
                continue
            seen.append(sr)
            if devices.check_samplerate(self.choice, sr, self.in_channels,
                                        self.out_channels):
                if sr != wanted and self.verbose:
                    print(
                        f"  note: {wanted} Hz was refused by the device; "
                        f"using {sr} Hz instead.",
                        file=sys.stderr,
                    )
                return sr
        raise RuntimeError(
            f"no usable sample rate. Tried {seen} on\n{self.choice.summary()}\n"
            "If this is WASAPI exclusive mode, the rate must match the one set "
            "in Windows Sound settings for this device."
        )

    def _probe_blocksize(self, wanted: int) -> int:
        """Open a throwaway stream to learn the block size we will really get."""
        seen: list[int] = []

        def cb(indata, outdata, frames, t, status):
            outdata.fill(0)
            seen.append(frames)
            raise sd.CallbackStop

        try:
            with sd.Stream(
                device=(self.choice.input_index, self.choice.output_index),
                samplerate=self.samplerate,
                blocksize=wanted,
                dtype="float32",
                channels=(self.in_channels, self.out_channels),
                latency=self.cfg.latency,
                extra_settings=self.choice.extra_settings,
                callback=cb,
            ):
                deadline = time.time() + 1.5
                while not seen and time.time() < deadline:
                    time.sleep(0.01)
        except Exception as exc:
            raise RuntimeError(
                f"could not open the audio device.\n{self.choice.summary()}\n"
                f"PortAudio said: {exc}"
            ) from exc

        actual = seen[0] if seen else wanted
        if actual != wanted and self.verbose:
            print(
                f"  note: asked for {wanted}-sample blocks, the driver gives "
                f"{actual}. Using {actual}.",
                file=sys.stderr,
            )
        return int(actual)

    # ------------------------------------------------------------------
    # Real-time callback
    # ------------------------------------------------------------------

    def _callback(self, indata, outdata, frames, time_info, status) -> None:
        t0 = time.perf_counter()

        if status:
            # input_overflow / output_underflow: we missed the deadline.
            if status.input_overflow or status.output_underflow:
                self.status.xruns += 1

        if frames != self.blocksize:
            # Should not happen -- the block size was probed. Fail safe.
            outdata.fill(0)
            return

        # --- pull the latest parameter targets -----------------------------
        flags = 0
        if self.target_channel is not None:
            self.target_channel.read_into(self._targets_in)
            flags = self.target_channel.flags
            if flags & FLAG_PANIC:
                if not self._panic_seen:
                    self.chain.reset()
                    self._panic_seen = True
                outdata.fill(0)
                return
            self._panic_seen = False
            if not (flags & FLAG_HOLD):
                self.chain.target[:] = self._targets_in

        # --- input: one channel, to float64, with input trim ---------------
        np.multiply(indata[:, self.input_channel], self._in_gain, out=self._x)
        self.status.peak_in = peak_level(self._x)

        # --- process --------------------------------------------------------
        if flags & FLAG_BYPASS:
            self._y[:] = self._x
        else:
            if self._dry_mix > 0.0:
                self._dry[:] = self._x
            self.chain.process(self._x, self._y)
            if self._dry_mix > 0.0:
                np.multiply(self._dry, self._dry_mix, out=self._dry)
                np.add(self._y, self._dry, out=self._y)

        # --- output ---------------------------------------------------------
        np.multiply(self._y, self._out_gain, out=self._y)
        sanitise(self._y, self._limit)
        self.status.peak_out = peak_level(self._y)

        np.copyto(self._out, self._y, casting="same_kind")
        outdata[:, 0] = self._out
        for c in range(1, outdata.shape[1]):
            outdata[:, c] = self._out

        # --- telemetry ------------------------------------------------------
        self.status.blocks += 1
        us = (time.perf_counter() - t0) * 1e6
        self.status.last_us = us
        if self.status.blocks == 1:
            self.status.first_us = us
        elif self.status.blocks > SETTLE_BLOCKS and us > self.status.worst_us:
            self.status.worst_us = us

        if self.display_channel is not None and \
                self.status.blocks % self._publish_every == 0:
            # A straight memcpy of the current values; the vision process
            # normalises them for display so the audio thread does not have to.
            self.display_channel.write(self.chain.current)
            self.display_channel.set_stat(STAT_XRUNS, self.status.xruns)
            budget = self.blocksize / self.samplerate * 1e6
            self.display_channel.set_stat(
                STAT_CPU_PERMILLE, int(min(us / budget * 1000.0, 65535))
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        # Compile every kernel before the sound card is listening. Doing this
        # lazily in the callback costs the better part of a second on the first
        # block, which a strict driver treats as a fatal overrun.
        if self.verbose:
            print("  warming up DSP (compiling kernels)...", end="", flush=True)
        took = self.chain.warmup()

        # The chain is not the whole callback: these two kernels live in the
        # engine itself and would otherwise compile on the first block.
        self._x[:] = 0.001
        for _ in range(2):
            peak_level(self._x)
            sanitise(self._x, self._limit)
            self.chain.process(self._x, self._y)
            np.copyto(self._out, self._y, casting="same_kind")
        self._x[:] = 0.0
        self._y[:] = 0.0
        self._out[:] = 0.0
        self.chain.reset()

        # Touch the IPC pages too, so the first real read is not a page fault.
        if self.target_channel is not None:
            self.target_channel.read_into(self._targets_in)
        if self.display_channel is not None:
            self.display_channel.write(self.chain.current)

        if self.verbose:
            print(f" {took:.1f}s")

        self._stream = sd.Stream(
            device=(self.choice.input_index, self.choice.output_index),
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype="float32",
            channels=(self.in_channels, self.out_channels),
            latency=self.cfg.latency,
            extra_settings=self.choice.extra_settings,
            callback=self._callback,
        )
        # Everything allocated by now is permanent; freeze it out of the GC's
        # sight and stop collecting entirely while we are live.
        gc.collect()
        gc.freeze()
        gc.disable()
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        if self._gc_was_enabled:
            gc.enable()
        gc.unfreeze()

    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        """False if the driver stopped the stream underneath us."""
        return self._stream is not None and self._stream.active

    @property
    def latency_ms(self) -> tuple[float, float]:
        if self._stream is None:
            return (0.0, 0.0)
        lat = self._stream.latency
        return (float(lat[0]) * 1000.0, float(lat[1]) * 1000.0)

    def summary(self) -> str:
        li, lo = self.latency_ms
        block_ms = self.blocksize / self.samplerate * 1000.0
        lines = [
            self.choice.summary(),
            f"format   : {self.samplerate} Hz, {self.blocksize}-sample blocks "
            f"({block_ms:.2f} ms per block)",
            f"channels : in {self.in_channels} (using #{self.input_channel}), "
            f"out {self.out_channels}",
        ]
        if self._stream is not None:
            lines.append(
                f"latency  : driver reports {li:.1f} ms in + {lo:.1f} ms out "
                f"= {li + lo:.1f} ms round trip"
            )
        return "\n".join(lines)
