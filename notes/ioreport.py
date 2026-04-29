# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Sudo-free Apple Silicon telemetry via IOReport (libIOReport.dylib).

IOReport is the framework powermetrics itself uses to read counters.
The pattern is:

  1. IOReportCopyAllChannels()         -> dict of all available channels
  2. IOReportCreateSubscription(...)   -> subscribe to a filtered subset
  3. IOReportCreateSamples(sub) at t0  -> snapshot
  4. ... wait window_ms ...
  5. IOReportCreateSamples(sub) at t1  -> snapshot
  6. IOReportCreateSamplesDelta(t0, t1) -> per-channel deltas
  7. iterate the delta dict, read SimpleGet/StateGet values

The consumer chooses the sample window — there's no driver-side
publishing cadence to wait on (which is what made ioreg's
utilization fields useless in exp 007).

Two modes:

  uv run notes/ioreport.py --list
      enumerate every channel the system exposes (group / subgroup /
      name / format). Useful to discover what to subscribe to.

  uv run notes/ioreport.py
      live dashboard at 1 s cadence with GPU energy -> power, GPU
      package energy if present. Optional --csv writes per-sample
      rows including monotonic_ns for joining with experiment data.

No sudo required (verified on M4 Max / macOS 26.4.1).

Implementation note: libIOReport returns CoreFoundation types
(CFDictionary, CFArray, CFString, CFNumber). We use ctypes against
both libIOReport and CoreFoundation directly rather than pulling in
PyObjC for this -- keeps the dependency surface to the stdlib + the
system frameworks already on the box. Reference for the API shape:
https://github.com/vladkens/macmon (Rust, well-commented).
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------
# Library loading

_IOREPORT_PATH = "/usr/lib/libIOReport.dylib"
_CF_PATH = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"

ioreport = ctypes.CDLL(_IOREPORT_PATH)
cf = ctypes.CDLL(_CF_PATH)

# CoreFoundation
cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
cf.CFArrayGetCount.restype = ctypes.c_long
cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cf.CFDictionaryGetValue.restype = ctypes.c_void_p
cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
cf.CFStringCreateWithCString.restype = ctypes.c_void_p
cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
cf.CFStringGetCString.restype = ctypes.c_bool
cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
cf.CFStringGetLength.restype = ctypes.c_long
cf.CFRelease.argtypes = [ctypes.c_void_p]
cf.CFRelease.restype = None
cf.CFRetain.argtypes = [ctypes.c_void_p]
cf.CFRetain.restype = ctypes.c_void_p

K_CFSTRING_ENCODING_UTF8 = 0x08000100

# IOReport — signatures cross-checked against macmon (Rust)
ioreport.IOReportCopyAllChannels.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
ioreport.IOReportCopyAllChannels.restype = ctypes.c_void_p

ioreport.IOReportCopyChannelsInGroup.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
    ctypes.c_uint64, ctypes.c_uint64,
]
ioreport.IOReportCopyChannelsInGroup.restype = ctypes.c_void_p

ioreport.IOReportCreateSubscription.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_uint64, ctypes.c_void_p,
]
ioreport.IOReportCreateSubscription.restype = ctypes.c_void_p

ioreport.IOReportCreateSamples.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
ioreport.IOReportCreateSamples.restype = ctypes.c_void_p

ioreport.IOReportCreateSamplesDelta.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
ioreport.IOReportCreateSamplesDelta.restype = ctypes.c_void_p

# Per-channel inspectors. These take a single channel CFDictionary
# (not the top-level samples dict) and return CFString or int64.
ioreport.IOReportChannelGetGroup.argtypes = [ctypes.c_void_p]
ioreport.IOReportChannelGetGroup.restype = ctypes.c_void_p
ioreport.IOReportChannelGetSubGroup.argtypes = [ctypes.c_void_p]
ioreport.IOReportChannelGetSubGroup.restype = ctypes.c_void_p
ioreport.IOReportChannelGetChannelName.argtypes = [ctypes.c_void_p]
ioreport.IOReportChannelGetChannelName.restype = ctypes.c_void_p
ioreport.IOReportChannelGetFormat.argtypes = [ctypes.c_void_p]
ioreport.IOReportChannelGetFormat.restype = ctypes.c_int
ioreport.IOReportChannelGetUnitLabel.argtypes = [ctypes.c_void_p]
ioreport.IOReportChannelGetUnitLabel.restype = ctypes.c_void_p
ioreport.IOReportSimpleGetIntegerValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
ioreport.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64

# STATE-channel inspectors. Same shape as the SIMPLE getter -- take a
# channel CFDictionary, plus an integer state index for the per-state
# variants. Signatures cross-checked against macmon (Rust).
ioreport.IOReportStateGetCount.argtypes = [ctypes.c_void_p]
ioreport.IOReportStateGetCount.restype = ctypes.c_int32
ioreport.IOReportStateGetNameForIndex.argtypes = [ctypes.c_void_p, ctypes.c_int32]
ioreport.IOReportStateGetNameForIndex.restype = ctypes.c_void_p
ioreport.IOReportStateGetResidency.argtypes = [ctypes.c_void_p, ctypes.c_int32]
ioreport.IOReportStateGetResidency.restype = ctypes.c_int64

# Format codes (from IOKit headers; what we actually care about):
IOREPORT_FORMAT_NONE = 0
IOREPORT_FORMAT_SIMPLE = 1
IOREPORT_FORMAT_STATE = 2
IOREPORT_FORMAT_HISTOGRAM = 3
IOREPORT_FORMAT_SIMPLE_ARRAY = 4

FORMAT_NAMES = {
    0: "NONE", 1: "SIMPLE", 2: "STATE", 3: "HISTOGRAM", 4: "SIMPLE_ARRAY",
}


# ---------------------------------------------------------------------
# CF helpers

def cfstring_to_str(cf_ptr) -> str | None:
    """Read a CFStringRef into a Python str. Returns None on null."""
    if not cf_ptr:
        return None
    # CFStringGetLength is in UTF-16 code units; over-allocate.
    n_units = cf.CFStringGetLength(cf_ptr)
    buf_size = max(n_units * 4 + 1, 64)
    buf = ctypes.create_string_buffer(buf_size)
    ok = cf.CFStringGetCString(cf_ptr, buf, buf_size, K_CFSTRING_ENCODING_UTF8)
    if not ok:
        return None
    return buf.value.decode("utf-8", errors="replace")


def py_str_to_cfstring(s: str):
    """Allocate a CFStringRef from a Python str. Caller must CFRelease."""
    return cf.CFStringCreateWithCString(None, s.encode("utf-8"), K_CFSTRING_ENCODING_UTF8)


# ---------------------------------------------------------------------
# Sample iteration

def iterate_channels(samples_ptr):
    """Yield each channel CFDictionary in a samples (or delta) dict.
    The dict has key "IOReportChannels" -> CFArray of channel dicts."""
    if not samples_ptr:
        return
    key = py_str_to_cfstring("IOReportChannels")
    arr = cf.CFDictionaryGetValue(samples_ptr, key)
    cf.CFRelease(key)
    if not arr:
        return
    n = cf.CFArrayGetCount(arr)
    for i in range(n):
        chan = cf.CFArrayGetValueAtIndex(arr, i)
        if chan:
            yield chan


def channel_metadata(chan_ptr) -> dict:
    """Pull (group, subgroup, name, format, unit) for one channel."""
    return {
        "group": cfstring_to_str(ioreport.IOReportChannelGetGroup(chan_ptr)),
        "subgroup": cfstring_to_str(ioreport.IOReportChannelGetSubGroup(chan_ptr)),
        "name": cfstring_to_str(ioreport.IOReportChannelGetChannelName(chan_ptr)),
        "format": ioreport.IOReportChannelGetFormat(chan_ptr),
        "unit": cfstring_to_str(ioreport.IOReportChannelGetUnitLabel(chan_ptr)),
    }


# ---------------------------------------------------------------------
# Subscription + sampling

class IOReportSubscription:
    """Subscribe once at construction time, then call sample_delta()
    repeatedly. The previous snapshot is kept internally so each call
    returns delta-since-last-call."""

    def __init__(self, channels_dict_ptr=None):
        if channels_dict_ptr is None:
            channels_dict_ptr = ioreport.IOReportCopyAllChannels(0, 0)
            owns_channels = True
        else:
            owns_channels = False
        if not channels_dict_ptr:
            raise RuntimeError("IOReportCopyAllChannels returned NULL")

        subbed_out = ctypes.c_void_p()
        sub = ioreport.IOReportCreateSubscription(
            None, channels_dict_ptr, ctypes.byref(subbed_out), 0, None
        )
        if not sub:
            if owns_channels:
                cf.CFRelease(channels_dict_ptr)
            raise RuntimeError("IOReportCreateSubscription returned NULL")

        if owns_channels:
            cf.CFRelease(channels_dict_ptr)

        self._sub = sub
        self._subbed = subbed_out  # the dict of subscribed channels (do NOT release)
        self._prev_sample = None

    def sample_once(self):
        """Take one snapshot. Returns CFDictionary pointer the caller is
        responsible for CFRelease()ing."""
        return ioreport.IOReportCreateSamples(self._sub, self._subbed, None)

    def sample_delta(self):
        """Take a snapshot, compute delta vs the previous one, return the
        delta dict (caller releases). The first call returns None because
        there's no previous to delta against."""
        current = self.sample_once()
        if self._prev_sample is None:
            self._prev_sample = current
            return None
        delta = ioreport.IOReportCreateSamplesDelta(
            self._prev_sample, current, None
        )
        cf.CFRelease(self._prev_sample)
        self._prev_sample = current
        return delta

    def close(self):
        if self._prev_sample:
            cf.CFRelease(self._prev_sample)
            self._prev_sample = None
        # The subscription itself: macmon doesn't release it explicitly
        # because it's not clearly +1 from CreateSubscription. Leaving
        # alone to avoid double-free risk.


# ---------------------------------------------------------------------
# Channels of interest for v1.
#
# The Energy Model group on M4 Max enumerates ~150 energy channels: per-
# core, per-cluster, per-cluster-aggregate, and finally a single
# canonical total. We use the canonical totals only -- summing the
# breakdowns would 10x over-count.
#
# Names confirmed present on M4 Max / macOS 26.4.1 by --list:
#   CPU Energy   (mJ) -- total of E + P clusters
#   GPU Energy   (nJ) -- GPU compute, high-precision
#   GPU          (mJ) -- duplicate of GPU Energy at lower precision
#   GPU SRAM     (mJ) -- on-chip GPU memory
#   AFR          (mJ) -- Apple Fixed-function Renderer (display engine)
#   ISP          (mJ) -- Image Signal Processor
#   AVE          (mJ) -- Audio/Video Encoder
#   AMCC         (mJ) -- memory controller (cache coherency)
#   DCS          (mJ) -- DRAM/cache subsystem
#   DRAM         (mJ) -- DRAM
#   DISP/DISPEXT (mJ) -- internal / external display
#   PCIe Port * Energy (uJ)
#
# ANE Energy was NOT in the enumeration on this machine; ANE is
# probably reported under a different group name or only when active.
# We leave the slot in case it appears.

ENERGY_GROUPS = {"Energy Model"}

# Map of "logical bucket" -> list of canonical channel names to sum.
# Each name is matched exactly (no prefix matching) so we never
# double-count breakdowns.
ENERGY_BUCKETS: dict[str, list[str]] = {
    "gpu":         ["GPU Energy"],         # nJ precision
    "cpu":         ["CPU Energy"],         # canonical total
    "ane":         ["ANE Energy"],         # may not be present at idle
    "dram":        ["DRAM"],
    "amcc":        ["AMCC"],
    "dcs":         ["DCS"],
    "afr":         ["AFR"],
    "disp":        ["DISP", "DISPEXT"],
    "isp":         ["ISP"],
    "ave":         ["AVE"],
    "gpu_sram":    ["GPU SRAM"],
}


def power_mw_from_delta_nj(delta_nj: int, window_s: float) -> float:
    """Convert nJ delta over window seconds -> mW. nJ/s = nW; mW = nW/1e6.
    So mW = delta_nj / window_s / 1e6."""
    if window_s <= 0:
        return 0.0
    return delta_nj / window_s / 1e6


# Unit-aware: enumeration confirmed channels can be mJ, uJ, or nJ.
# Convert any energy unit to nJ so power_mw_from_delta_nj works.
def energy_to_nj(value: int, unit: str | None) -> int:
    """Normalize an energy value to nanojoules based on its unit label."""
    if unit is None:
        return value  # assume nJ if unlabeled
    u = unit.strip()
    if u == "nJ":
        return value
    if u == "uJ":
        return value * 1_000
    if u == "mJ":
        return value * 1_000_000
    if u == "J":
        return value * 1_000_000_000
    return value  # unknown unit — treat as nJ to avoid silent miscount


# ---------------------------------------------------------------------
# CLI: --list channels

def cmd_list() -> int:
    print("# enumerating all IOReport channels (group / subgroup / name / format / unit)")
    all_chans = ioreport.IOReportCopyAllChannels(0, 0)
    if not all_chans:
        print("ERROR: IOReportCopyAllChannels returned NULL", file=sys.stderr)
        return 1

    # We need a subscription to iterate; the all-channels dict itself
    # uses a slightly different shape than samples do.
    subbed_out = ctypes.c_void_p()
    sub = ioreport.IOReportCreateSubscription(
        None, all_chans, ctypes.byref(subbed_out), 0, None
    )
    cf.CFRelease(all_chans)
    if not sub:
        print("ERROR: IOReportCreateSubscription returned NULL", file=sys.stderr)
        return 1

    # Take one sample so we get the same channel-list shape as live runs.
    s = ioreport.IOReportCreateSamples(sub, subbed_out, None)
    if not s:
        print("ERROR: IOReportCreateSamples returned NULL", file=sys.stderr)
        return 1

    rows = []
    for chan in iterate_channels(s):
        meta = channel_metadata(chan)
        rows.append(meta)

    rows.sort(key=lambda m: (m["group"] or "", m["subgroup"] or "", m["name"] or ""))
    cur_group = None
    for m in rows:
        if m["group"] != cur_group:
            print(f"\n## group: {m['group']!r}")
            cur_group = m["group"]
        sub_str = f"  [subgroup={m['subgroup']!r}]" if m["subgroup"] else ""
        unit = f"  unit={m['unit']!r}" if m["unit"] else ""
        print(f"  - {m['name']!r}  format={FORMAT_NAMES.get(m['format'], m['format'])}{sub_str}{unit}")

    print(f"\n# total: {len(rows)} channels")
    cf.CFRelease(s)
    return 0


# ---------------------------------------------------------------------
# CLI: live dashboard

def cmd_dashboard(args) -> int:
    sub = IOReportSubscription()

    bucket_keys = list(ENERGY_BUCKETS.keys())   # stable column order
    csv_writer = None
    csv_file = None
    states_writer = None
    states_file = None
    states_csv_path = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        header = ["iso_ts", "monotonic_ns", "window_s"]
        for k in bucket_keys:
            header.append(f"{k}_power_mw")
        for k in bucket_keys:
            header.append(f"{k}_energy_nj")
        csv_writer.writerow(header)

        if args.include_states:
            # Sibling file: foo.csv -> foo-states.csv.
            base = Path(args.csv)
            states_csv_path = base.with_name(base.stem + "-states" + base.suffix)
            states_file = open(states_csv_path, "w", newline="")
            states_writer = csv.writer(states_file)
            states_writer.writerow([
                "iso_ts", "monotonic_ns", "window_s",
                "group", "subgroup", "channel",
                "state_idx", "state_name",
                "residency_24Mticks",
            ])

    print(f"# IOReport telemetry  interval={args.interval_ms}ms"
          + (f"  csv={args.csv}" if args.csv else "")
          + (f"  states_csv={states_csv_path}" if states_csv_path else ""))
    print(f"# Ctrl-C to stop")
    print(f"# {'time':<8}  " + "  ".join(f"{k:>5} mW" for k in bucket_keys))

    def shutdown(_signum=None, _frame=None):
        if csv_file:
            csv_file.close()
        if states_file:
            states_file.close()
        sub.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    last_t = time.monotonic()
    while True:
        time.sleep(args.interval_ms / 1000.0)
        delta = sub.sample_delta()
        now = time.monotonic()
        window_s = now - last_t
        last_t = now
        if delta is None:
            continue   # first iteration, no prev to delta against

        iso_ts = datetime.now().isoformat(timespec="microseconds")
        mono_ns = time.monotonic_ns()

        # Build a name-indexed map for O(1) bucket-name lookup.
        per_name_nj: dict[str, int] = {}
        state_rows: list[list] = []
        for chan in iterate_channels(delta):
            meta = channel_metadata(chan)
            fmt = meta["format"]
            if fmt == IOREPORT_FORMAT_SIMPLE:
                if meta["group"] not in ENERGY_GROUPS:
                    continue
                name = meta["name"] or ""
                value = ioreport.IOReportSimpleGetIntegerValue(chan, None)
                per_name_nj[name] = per_name_nj.get(name, 0) + energy_to_nj(value, meta["unit"])
            elif fmt == IOREPORT_FORMAT_STATE and states_writer is not None:
                if (args.state_groups_set is not None
                        and meta["group"] not in args.state_groups_set):
                    continue
                n_states = ioreport.IOReportStateGetCount(chan)
                for idx in range(n_states):
                    name_cf = ioreport.IOReportStateGetNameForIndex(chan, idx)
                    sname = cfstring_to_str(name_cf) if name_cf else None
                    resid = ioreport.IOReportStateGetResidency(chan, idx)
                    state_rows.append([
                        iso_ts, mono_ns, round(window_s, 6),
                        meta["group"], meta["subgroup"], meta["name"],
                        idx, sname, resid,
                    ])

        cf.CFRelease(delta)

        bucket_nj = {k: sum(per_name_nj.get(n, 0) for n in names)
                     for k, names in ENERGY_BUCKETS.items()}
        bucket_mw = {k: power_mw_from_delta_nj(v, window_s)
                     for k, v in bucket_nj.items()}

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  {ts}  " + "  ".join(f"{bucket_mw[k]:>5.0f} mW" for k in bucket_keys),
              flush=True)

        if csv_writer:
            row = [
                iso_ts,
                mono_ns,
                round(window_s, 6),
            ]
            for k in bucket_keys:
                row.append(round(bucket_mw[k], 3))
            for k in bucket_keys:
                row.append(bucket_nj[k])
            csv_writer.writerow(row)
            csv_file.flush()

        if states_writer:
            for srow in state_rows:
                states_writer.writerow(srow)
            states_file.flush()


# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--list", action="store_true",
                    help="enumerate all IOReport channels and exit")
    ap.add_argument("--interval-ms", type=int, default=1000,
                    help="sample interval in ms (default 1000)")
    ap.add_argument("--csv", type=str, default=None,
                    help="write per-sample CSV with monotonic_ns")
    ap.add_argument("--include-states", action="store_true",
                    help="also emit a sibling -states.csv with per-state "
                         "residency for STATE-format channels (GPUPH, "
                         "BSTGPUPH, PWRCTRL, etc.) in --state-groups")
    ap.add_argument("--state-groups", type=str, default="GPU Stats",
                    help="comma-separated group names to include in the "
                         "states CSV (default: 'GPU Stats'). System-wide "
                         "STATE channels number 1700+ and produce huge "
                         "CSVs if not filtered. Pass 'ALL' to disable.")
    args = ap.parse_args()
    if args.state_groups.strip().upper() == "ALL":
        args.state_groups_set = None
    else:
        args.state_groups_set = {g.strip() for g in args.state_groups.split(",")}

    if args.list:
        return cmd_list()
    return cmd_dashboard(args)


if __name__ == "__main__":
    sys.exit(main())
