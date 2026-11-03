"""ipm_kernels parity tests: optional parity summary.

Set CUPIQP_IPM_PARITY_REPORT=<path> to write, at the end of the session, the
worst max-ulp / max-abs / max-rel difference per (kernel, output array) seen
by the parity comparisons.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session", autouse=True)
def _ipm_parity_report():
    yield
    path = os.environ.get("CUPIQP_IPM_PARITY_REPORT")
    if not path:
        return
    from ipm_kernels_harness import REPORT
    worst = {}
    for label, key, ulp, abs_err, rel_err in REPORT.rows:
        kernel = label.split("[")[0].rstrip(": ")
        w = worst.setdefault((kernel, key), [0.0, 0.0, 0.0])
        w[0], w[1], w[2] = max(w[0], ulp), max(w[1], abs_err), max(w[2], rel_err)
    with open(path, "w") as f:
        f.write(f"{'kernel':42s} {'array':24s} {'max_ulp':>8s} {'max_abs':>10s} {'max_rel':>10s}\n")
        for (kernel, key), (u, a, r) in sorted(worst.items()):
            f.write(f"{kernel:42s} {key:24s} {u:8.0f} {a:10.2e} {r:10.2e}\n")
