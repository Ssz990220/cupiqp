"""Compare host clocks and CUDA events against the Windows host clock (WSL).

Prints JSON with the ratio of each Linux clock and of CUDA event time to the
Windows QueryPerformanceCounter clock over --seconds (busy wait on
CLOCK_MONOTONIC_RAW). A ratio far from 1 means that clock must not be used for
wall times. Usage: .venv/bin/python native/bench/tools/check_clocks.py [--seconds 20]
"""
import argparse
import json
import subprocess
import time

PS = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def windows_seconds():
    out = subprocess.run([PS, "-NoProfile", "-Command",
                          "[System.Diagnostics.Stopwatch]::GetTimestamp(); [System.Diagnostics.Stopwatch]::Frequency"],
                         capture_output=True, text=True, check=True).stdout.split()
    return int(out[0]) / int(out[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    import warp as wp
    wp.config.quiet = True
    wp.init()
    stream = wp.Stream("cuda:0")
    e0, e1 = (wp.Event(device="cuda:0", enable_timing=True) for _ in range(2))
    clocks = {"MONOTONIC": time.CLOCK_MONOTONIC, "MONOTONIC_RAW": time.CLOCK_MONOTONIC_RAW,
              "REALTIME": time.CLOCK_REALTIME}
    w0 = windows_seconds()
    t0 = {k: time.clock_gettime(c) for k, c in clocks.items()}
    stream.record_event(e0)
    end = time.clock_gettime(time.CLOCK_MONOTONIC_RAW) + args.seconds
    while time.clock_gettime(time.CLOCK_MONOTONIC_RAW) < end:
        pass
    stream.record_event(e1)
    wp.synchronize_stream(stream)
    t1 = {k: time.clock_gettime(c) for k, c in clocks.items()}
    w1 = windows_seconds()
    win = w1 - w0
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "windows_qpc_s": win,
           "ratio_to_windows": {k: (t1[k] - t0[k]) / win for k in clocks},
           "note": "windows interval includes two powershell launches (~0.2-0.4 s); CUDA events cover only the busy wait"}
    out["ratio_to_windows"]["CUDA_EVENTS"] = wp.get_event_elapsed_time(e0, e1) / 1e3 / win
    out["cuda_events_over_monotonic_raw"] = (wp.get_event_elapsed_time(e0, e1) / 1e3) / (t1["MONOTONIC_RAW"] - t0["MONOTONIC_RAW"])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
