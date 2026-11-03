"""Export benchmark QP snapshots to a binary bundle for the native C++ benchmark.

Reads a qp-benchmark bundle (manifest.json + cases/<id>/problem.npz, loaded with
the benchmark's own ``qp_io.load_problem``), converts P, A, C to sorted CSR with
explicit zeros kept (same as benchmark/replay_mpc_updates.py), groups cases by
identical (P, A, C) pattern, and writes one file per pattern group:

    <out>/group_<k>.bin   little-endian:
        char[8]  magic "CPQPSNP1"
        int32    n, p, m, nnz_P, nnz_A, nnz_C, num_cases
        int32    P.indptr[n+1], P.indices[nnz_P]
        int32    A.indptr[p+1], A.indices[nnz_A]
        int32    C.indptr[m+1], C.indices[nnz_C]
        float64  per case, packed in harness order: P, A, C, q, b, l, u
    <out>/manifest.json   groups, case ids per group in file order, case metadata

Usage (WSL):
    .venv/bin/python -I native/bench/tools/export_snapshots.py \
        --bundle /mnt/c/Users/91354/Projects/gpu-qp-benchmark/qp_benchmark_v2_20261102 \
        --cohort main --out native/bench/data/mpc
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

BENCH_ROOT = Path("/mnt/c/Users/91354/Projects/gpu-qp-benchmark")
NAMES = ("P", "A", "C", "q", "b", "l", "u")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--cohort", default="main")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qp-io", type=Path, default=BENCH_ROOT / "qp_benchmark_v2_20261102" / "tools",
                    help="directory containing the benchmark's qp_io.py")
    args = ap.parse_args()
    sys.path.insert(0, str(args.qp_io))
    from qp_io import load_problem  # noqa: E402  (benchmark-owned loader)

    manifest = json.loads((args.bundle / "manifest.json").read_text())
    cases = [c for c in manifest["cases"] if c.get("cohort", "main") == args.cohort]
    groups: dict[str, dict] = {}
    for c in cases:
        p = load_problem(args.bundle / "cases" / c["id"])
        mats = {}
        for k in ("P", "A", "C"):
            m = p[k].tocsr()
            m.sort_indices()  # explicit zeros kept (no eliminate_zeros)
            mats[k] = m
        h = hashlib.sha256()
        for k in ("P", "A", "C"):
            h.update(np.int64(mats[k].shape).tobytes())
            h.update(mats[k].indptr.astype(np.int32).tobytes())
            h.update(mats[k].indices.astype(np.int32).tobytes())
        key = h.hexdigest()
        g = groups.setdefault(key, {"mats": mats, "cases": [], "values": []})
        flat = np.concatenate([mats[k].data if k in mats else np.asarray(p[k], dtype=np.float64)
                               for k in NAMES]).astype(np.float64)
        g["cases"].append(c)
        g["values"].append(flat)

    args.out.mkdir(parents=True, exist_ok=True)
    out_manifest = {"bundle": str(args.bundle), "cohort": args.cohort, "value_order": list(NAMES),
                    "groups": []}
    for gi, (key, g) in enumerate(groups.items()):
        P, A, C = g["mats"]["P"], g["mats"]["A"], g["mats"]["C"]
        n, p_, m = P.shape[0], A.shape[0], C.shape[0]
        path = args.out / f"group_{gi}.bin"
        with path.open("wb") as f:
            f.write(b"CPQPSNP1")
            f.write(np.array([n, p_, m, P.nnz, A.nnz, C.nnz, len(g["cases"])], dtype="<i4").tobytes())
            for M in (P, A, C):
                f.write(M.indptr.astype("<i4").tobytes())
                f.write(M.indices.astype("<i4").tobytes())
            for flat in g["values"]:
                f.write(flat.astype("<f8").tobytes())
        out_manifest["groups"].append({
            "file": path.name, "pattern_sha256": key, "n": n, "p": p_, "m": m,
            "nnz": {"P": P.nnz, "A": A.nnz, "C": C.nnz},
            "explicit_zeros": {k: int((g["mats"][k].data == 0).sum()) for k in ("P", "A", "C")},
            "cases": [c["id"] for c in g["cases"]],
            "case_meta": [{k: c.get(k) for k in ("id", "trajectory_name", "trajectory", "control_step",
                                                 "sqp_iteration")} for c in g["cases"]],
        })
        print(f"group {gi}: n={n} p={p_} m={m} nnz P/A/C={P.nnz}/{A.nnz}/{C.nnz} cases={len(g['cases'])} -> {path}")
    (args.out / "manifest.json").write_text(json.dumps(out_manifest, indent=1))


if __name__ == "__main__":
    main()
