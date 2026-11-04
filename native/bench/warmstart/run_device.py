"""Phase-2 runner: the device-side warm start (SparseSolver.solve_warm_start).

For each run, one native solver (pattern fixed). Every configuration is run
as a chain over all QPs of the run: QP 0 cold, QP k+1 warm started on the
device from the solver's own result of QP k (no host round trip). Each chain
is repeated --passes times (reset_warm_start() between passes, the graphs
stay captured); the first pass is a warm-up, the reported time of a QP is the
median over the remaining passes. Iterations and solutions are deterministic
and are checked to be identical in every pass.

Config spec: name[:key=val,...]
  cold                         plain solve() (default path)
  dev[:s_min=..,z_min=..,z_unm=..,y=0|1,match=identity|pair|pair_global|label,
      K=..,back=..,corrupt=unmatched|mismatch,frac=..,seed=..]
match:
  identity      RowMatch.Identity
  pair          keys on the collision block only: group id = sample index, key = pair id
  pair_block    the same keys, groups as fixed blocks of 16 rows (key_group_size, no group ids)
  pair_global   every row keyed (hash of the phase-1 (sample, pair) / label key), no groups
  label         every row keyed by a hash of its raw label (phase-1 map=label)

Writes <out>/records.jsonl (one line per config, run, k).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws_common import RUNS, RunSolver, kkt_check, kkt_pass, load_qp, num_qps, raw_ns, scipy_mats  # noqa: E402
from ws_strategies import row_keys  # noqa: E402

GROUP = 16  # collision slots per sample


def parse_spec(spec):
    name, _, kws = spec.partition(":")
    kw = {}
    for t in (t for t in kws.split(",") if t):
        k, v = t.split("=")
        kw[k] = v
    return name, kw


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def collision_block(labels):
    idx = [i for i, l in enumerate(labels) if l.startswith("collision")]
    first, count = idx[0], len(idx)
    assert idx == list(range(first, first + count)) and count % GROUP == 0
    return first, count


def pair_id(label):
    s = label[label.index("pair=") + 5:]
    return int(s[:s.index("(")])


class KeyPlan:
    """Keys of every QP of a run for one match mode, plus the true collision
    match (by (sample, pair)) for the realized-corruption statistics."""

    def __init__(self, qps, mode, corrupt=None, frac=0.0, seed=0):
        self.mode = mode
        first, count = collision_block(qps[0]["labels"])
        self.first, self.count = first, count
        N = len(qps)
        sample = np.arange(count) // GROUP
        pairs = [np.array([pair_id(l) for l in q["labels"][first:first + count]], dtype=np.int64) for q in qps]
        # true match of collision row i of QP k in QP k-1
        self.truth = [None]
        for k in range(1, N):
            pos = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(sample, pairs[k - 1]))}
            self.truth.append(np.array([pos.get((int(s), int(p)), -1) for s, p in zip(sample, pairs[k])]))
        self.keys, self.groups = [], []
        if mode == "identity":
            return
        if mode in ("pair", "pair_block"):
            rng = np.random.default_rng(seed)
            for k in range(N):
                key = pairs[k].copy()
                if corrupt and k > 0 and frac > 0:
                    sel = rng.random(count) < frac
                    for i in np.nonzero(sel)[0]:
                        if corrupt == "unmatched":
                            key[i] = -(k * 1_000_000 + i) - 1  # never used by any other QP
                        else:  # mismatch: the committed key of another row of the same sample in QP k-1
                            g0 = (i // GROUP) * GROUP
                            cand = [j for j in range(g0, g0 + GROUP)
                                    if self.keys[k - 1][j] != key[i] and j != self.truth[k][i]]
                            if not cand:  # every other committed key equals this one (heavy corruption)
                                cand = [j for j in range(g0, g0 + GROUP) if j != self.truth[k][i]]
                            key[i] = self.keys[k - 1][rng.choice(cand)]
                self.keys.append(key)
                self.groups.append(sample.astype(np.int32) if mode == "pair" else None)
            return
        for q in qps:
            ks = row_keys(list(q["labels"]), "pair" if mode == "pair_global" else "label")
            self.keys.append(np.array([h64(s) for s in ks], dtype=np.int64))
            self.groups.append(None)

    def options(self, cn):
        if self.mode == "identity":
            return dict(row_match=cn.RowMatch.Identity)
        if self.mode == "pair":
            return dict(row_match=cn.RowMatch.Keys, row_groups=True, key_row_begin=self.first,
                        key_row_count=self.count)
        if self.mode == "pair_block":
            return dict(row_match=cn.RowMatch.Keys, row_groups=False, key_group_size=GROUP,
                        key_row_begin=self.first, key_row_count=self.count)
        return dict(row_match=cn.RowMatch.Keys, row_groups=False, key_row_begin=0, key_row_count=-1)

    def realized(self, k, match):
        """Fraction of collision rows of QP k whose device match differs from
        the (sample, pair) truth, and the fraction left unmatched."""
        if k == 0 or match is None:
            return None
        t = self.truth[k]
        if self.mode in ("pair", "pair_block"):
            m = match
        else:  # whole-m keys: restrict to the collision block, shift indices
            m = match[self.first:self.first + self.count]
            m = np.where(m >= 0, m - self.first, -1)
        return dict(wrong=float(np.mean(m != t)), unmatched=float(np.mean(m < 0)),
                    wrong_matched=float(np.mean((m != t) & (m >= 0))))


def read_match(rs):
    for name, ptr, rows, cols, _ in rs.s.native.debug_buffers():
        if name == "warm.match":
            a = rs.wp.array(ptr=ptr, dtype=rs.wp.int32, shape=(cols,), device="cuda:0")
            return a.numpy().copy()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--passes", type=int, default=4, help="chains per config; the first is a warm-up")
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    ap.add_argument("--max-qps", type=int, default=0)
    ap.add_argument("--no-kkt", action="store_true")
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rec_f = open(args.out / "records.jsonl", "a")

    for run in args.runs:
        N = num_qps(run) if not args.max_qps else min(args.max_qps, num_qps(run))
        qps = [load_qp(run, k) for k in range(N)]
        mats = [scipy_mats(q) for q in qps]
        rs = RunSolver(qps[0])
        cn = rs.cn
        cold_x = None
        for spec in args.configs:
            name, kw = parse_spec(spec)
            plan, kb = None, None
            if name == "dev":
                mode = kw.get("match", "identity")
                plan = KeyPlan(qps, mode, kw.get("corrupt"), float(kw.get("frac", 0)), int(kw.get("seed", 0)))
                opts = dict(s_min=float(kw.get("s_min", 0.1)), z_min=float(kw.get("z_min", 1e-4)),
                            z_unmatched=float(kw.get("z_unm", 1e-4)), previous_y=bool(int(kw.get("y", 1))),
                            snapshot_slots=int(kw.get("K", 0)), snapshot_back=int(kw.get("back", 0)))
                opts.update(plan.options(cn))
                rs.s.set_warm_start(**opts)
                kb = rs.s.row_key_buffers(host=True) if plan.mode != "identity" else None
            per_k = [dict(ms=[]) for _ in range(N)]
            for ps in range(args.passes):
                if plan is not None:
                    rs.s.reset_warm_start()
                for k in range(N):
                    rs.load(qps[k])
                    if kb is not None and plan is not None:
                        kb["keys"][:] = plan.keys[k]
                        if kb["groups"] is not None:
                            kb["groups"][:] = plan.groups[k]
                    c0 = rs.s.native.counters()
                    t0 = raw_ns()
                    if plan is None:
                        rs.s.native.solve()
                    else:
                        rs.s.native.solve_warm_start(kb is not None)
                    rs.s.native.synchronize()
                    t = (raw_ns() - t0) * 1e-6
                    c1 = rs.s.native.counters()
                    i = rs.info()
                    d = per_k[k]
                    if ps:
                        d["ms"].append(t)
                    if ps == 0:
                        d.update(status=i["status"], iter=i["iter"])
                    else:
                        assert d["iter"] == i["iter"] and d["status"] == i["status"], (spec, k, "nondeterministic")
                    if ps == args.passes - 1:
                        d["launches"] = c1["graph_launches"] - c0["graph_launches"]
                        d["syncs"] = c1["host_syncs"] - c0["host_syncs"]
                        d["captures"] = c1["graph_captures"] - c0["graph_captures"]
                        r = rs.result()
                        if not args.no_kkt:
                            chk = kkt_check(qps[k], r["x"], r["y"], r["z_l"], r["z_u"], mats[k])
                            d.update(kkt=chk, kkt_pass=kkt_pass(chk))
                        d["x"] = r["x"]
                        if plan is not None and plan.mode != "identity":
                            d["match"] = plan.realized(k, read_match(rs))
            it = [d["iter"] for d in per_k[1:]]
            print(f"{run[:14]} {spec:60s} iter med {np.median(it):.0f} mean {np.mean(it):.2f} max {max(it)} "
                  f"ms med {np.median([np.median(d['ms']) if d['ms'] else np.nan for d in per_k[1:]]):.2f}", flush=True)
            if spec == "cold":
                cold_x = [d["x"] for d in per_k]
            for k, d in enumerate(per_k):
                x = d.pop("x")
                ms_all = d.pop("ms")
                rec = dict(config=spec, run=run, k=k, ms=float(np.median(ms_all)) if ms_all else None,
                           ms_all=ms_all, **d)
                if cold_x is not None:
                    rec["dx_inf"] = float(np.max(np.abs(x - cold_x[k])))
                rec_f.write(json.dumps(rec) + "\n")
            rec_f.flush()
        del rs
    rec_f.close()


if __name__ == "__main__":
    main()
