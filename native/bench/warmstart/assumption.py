"""Test of the working assumption "QP k+1 is derived from the solution of QP k".

Needs the cold solutions written by run_study.py (cold_<run>.npz). Per
transition k -> k+1 it reports:
  * linear-row consistency: on the joint rows (position / velocity /
    acceleration / jerk / delta_t rows, linear in the trajectory), the bounds
    of QP k+1 should equal the bounds of QP k shifted by C_k x_k. A least-squares
    step length alpha and the residual of that fit tell whether QP k+1 was
    linearized at trajectory + x_k (alpha = 1) or after a damped step;
  * slack prediction: slack of QP k at its solution, s_k = h_u,k - C_k x_k
    (and the lower side), against the slack of QP k+1 at x = 0, per row group;
  * active-set persistence per row group (z > tol), collision rows matched by
    label vs by position;
  * relative change of y and z (label-mapped).
Writes <out>/assumption.jsonl and prints a markdown summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws_common import INF, RUNS, load_qp, num_qps, scipy_mats  # noqa: E402
from ws_strategies import label_map  # noqa: E402


def group_of(label):
    if label.startswith("collision"):
        return "collision"
    if ":eef:" in label:
        return "eef"
    if "delta_t" in label:
        return "delta_t"
    for g in ("Q(position)", "dQ(velocity)", "ddQ(acceleration)", "dddQ(jerk)"):
        if g in label:
            return g
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--z-tol", type=float, default=1e-4)
    args = ap.parse_args()
    out = open(args.out / "assumption.jsonl", "w")
    for run in RUNS:
        cz = np.load(args.cold_dir / f"cold_{run}.npz")
        N = num_qps(run)
        qps = [load_qp(run, k) for k in range(N)]
        groups = np.array([group_of(l) for l in qps[0]["labels"]])
        for k in range(N - 1):
            a, b = qps[k], qps[k + 1]
            Ga = scipy_mats(a)[2]
            xa = cz[f"{k}_x"]
            za = cz[f"{k}_z_u"] - cz[f"{k}_z_l"]
            zb = cz[f"{k + 1}_z_u"] - cz[f"{k + 1}_z_l"]
            ya, yb = cz[f"{k}_y"], cz[f"{k + 1}_y"]
            idx = label_map(a["labels"], b["labels"])  # (sample, pair) for collision rows
            idx_raw = label_map(a["labels"], b["labels"], "label")
            Cx = Ga @ xa
            rec = dict(run=run, k=k, x_inf=float(np.max(np.abs(xa))))
            gb = np.array([group_of(l) for l in b["labels"]])
            # Per group and side: does the bound of QP k+1 equal the bound of
            # QP k shifted by alpha * C_k x_k? err1 = error at alpha = 1, which
            # is also |slack of QP k at x_k - slack of QP k+1 at x = 0|.
            fit = {}
            for g in np.unique(gb):
                for side, key in (("u", "ub"), ("l", "lb")):
                    m = (gb == g) & (idx >= 0)
                    src = idx[m]
                    sel = (np.abs(a[key][src]) < 1e19) & (np.abs(b[key][m]) < 1e19)
                    if not sel.any():
                        continue
                    d = a[key][src][sel] - b[key][m][sel]
                    cx = Cx[src][sel]
                    alpha = float(d @ cx / max(cx @ cx, 1e-300))
                    fit[f"{g}/{side}"] = dict(rows=int(sel.sum()), alpha=alpha,
                                              err1_max=float(np.max(np.abs(d - cx))),
                                              err1_med=float(np.median(np.abs(d - cx))),
                                              shift_max=float(np.max(np.abs(cx))),
                                              bound_change_max=float(np.max(np.abs(d))))
            rec["fit"] = fit
            # active sets (z = z_u - z_l, |z| > tol)
            act_a = np.abs(za) > args.z_tol
            act_b = np.abs(zb) > args.z_tol
            pr = {}
            for g in np.unique(gb):
                mb = (gb == g)
                nb = int((act_b & mb).sum())
                kept = int((act_b & mb & (idx >= 0) & act_a[np.maximum(idx, 0)]).sum())
                na = int((act_a & (groups == g)).sum())
                pr[g] = dict(active_k=na, active_k1=nb, active_both=kept)
            # collision by position instead of label
            mcol = gb == "collision"
            pr["collision_by_position"] = dict(active_both=int((act_b & act_a & mcol).sum()))
            rec["active"] = pr
            rec["active_k"] = int(act_a.sum())
            rec["active_k1"] = int(act_b.sum())
            rec["active_both"] = int((act_b & (idx >= 0) & act_a[np.maximum(idx, 0)]).sum())
            zm = np.where(idx >= 0, za[np.maximum(idx, 0)], 0.0)
            rec["dz_rel"] = float(np.linalg.norm(zb - zm) / max(np.linalg.norm(zb), 1e-300))
            zr = np.where(idx_raw >= 0, za[np.maximum(idx_raw, 0)], 0.0)
            rec["dz_rel_raw_label"] = float(np.linalg.norm(zb - zr) / max(np.linalg.norm(zb), 1e-300))
            rec["active_both_raw_label"] = int((act_b & (idx_raw >= 0) & act_a[np.maximum(idx_raw, 0)]).sum())
            rec["col_raw_label_matched"] = int(np.sum(idx_raw[gb == "collision"] >= 0))
            rec["dz_rel_by_position"] = float(np.linalg.norm(zb - za) / max(np.linalg.norm(zb), 1e-300))
            rec["dy_rel"] = float(np.linalg.norm(yb - ya) / max(np.linalg.norm(yb), 1e-300))
            col = gb == "collision"
            rec["col_label_same_position"] = int(np.sum(a["labels"][col] == b["labels"][col]))
            rec["col_label_matched"] = int(np.sum(idx[col] >= 0))
            rec["col_rows"] = int(col.sum())
            out.write(json.dumps(rec) + "\n")
    out.close()
    summarize(args.out / "assumption.jsonl")


def summarize(path):
    recs = [json.loads(l) for l in open(path)]
    for run in RUNS:
        R = [r for r in recs if r["run"] == run]
        print(f"\n### {run} ({len(R)} transitions)\n")
        print("| row group / side | rows (median) | alpha med | alpha min..max | err(alpha=1) med | err(alpha=1) max | shift C x_k max |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        keys = sorted({k for r in R for k in r["fit"]})
        for key in keys:
            F = [r["fit"][key] for r in R if key in r["fit"]]
            al = np.array([f["alpha"] for f in F])
            print(f"| {key} | {int(np.median([f['rows'] for f in F]))} | {np.median(al):.4f} | {al.min():.3f}..{al.max():.3f} | "
                  f"{np.median([f['err1_med'] for f in F]):.1e} | {max(f['err1_max'] for f in F):.1e} | "
                  f"{max(f['shift_max'] for f in F):.1e} |")
        print("\n| row group | active in k+1 (mean) | of those active in k (mean %) | min % |")
        print("|---|---:|---:|---:|")
        groups = sorted({g for r in R for g in r["active"] if g != "collision_by_position"})
        for g in groups + ["all"]:
            if g == "all":
                nb = np.array([r["active_k1"] for r in R]); kb = np.array([r["active_both"] for r in R])
            else:
                nb = np.array([r["active"][g]["active_k1"] for r in R])
                kb = np.array([r["active"][g]["active_both"] for r in R])
            frac = np.where(nb > 0, kb / np.maximum(nb, 1), np.nan)
            mn = np.nanmin(frac) * 100 if np.any(nb > 0) else float("nan")
            print(f"| {g} | {nb.mean():.1f} | {np.nanmean(frac) * 100 if np.any(nb > 0) else float('nan'):.1f} | {mn:.1f} |")
        cb = np.array([r["active"]["collision_by_position"]["active_both"] for r in R])
        cl = np.array([r["active"]["collision"]["active_both"] for r in R])
        print(f"\ncollision rows active in k and k+1, matched by (sample, pair): {cl.sum()}, by position: {cb.sum()}")
        print(f"collision rows per QP {R[0]['col_rows']}: raw label at the same position: mean "
              f"{np.mean([r['col_label_same_position'] for r in R]):.0f}; raw label found anywhere: mean "
              f"{np.mean([r['col_raw_label_matched'] for r in R]):.0f}; (sample, pair) found: mean "
              f"{np.mean([r['col_label_matched'] for r in R]):.0f}")
        print(f"all rows active in k and k+1: (sample, pair) map {sum(r['active_both'] for r in R)}, raw label map "
              f"{sum(r['active_both_raw_label'] for r in R)}, of {sum(r['active_k1'] for r in R)} active in k+1")
        dzr = np.array([r["dz_rel_raw_label"] for r in R])
        print(f"relative dual change with raw-label map: median {np.median(dzr):.3f}, max {dzr.max():.3f}")
        dz = np.array([r["dz_rel"] for r in R]); dzp = np.array([r["dz_rel_by_position"] for r in R])
        dy = np.array([r["dy_rel"] for r in R])
        print(f"relative dual change |z_k+1 - z_k|/|z_k+1| ((sample, pair) map): median {np.median(dz):.3f}, max {dz.max():.3f}; "
              f"by position: median {np.median(dzp):.3f}; y: median {np.median(dy):.3f}, max {dy.max():.3f}")


if __name__ == "__main__":
    main()
