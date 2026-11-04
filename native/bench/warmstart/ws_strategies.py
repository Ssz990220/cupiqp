"""Warm-start strategies: build the staged start for QP k+1 from the state of
QP k (re-solved cold) and the data of QP k+1. All arrays in user coordinates.

State dict `prev`: x, y, s_l, s_u, z_l, z_u of QP k (s/z exactly 0 on infinite
bounds). `qp` is QP k+1. Every strategy returns (start, fields, settings)
where settings may override rho_init / delta_init for this solve.
"""
from __future__ import annotations

import numpy as np

from ws_common import INF, scipy_mats


def row_keys(labels, mode="pair"):
    """Keys that identify an inequality row across SQP iterations.

    mode "label": the raw ineq_labels string. Collision labels carry the
    sample time ("t=0.549s"), which moves whenever delta_t changes, so the
    raw label of an unchanged row can differ between QP k and QP k+1.
    mode "pair" (default): collision rows keyed by (sample index, link pair),
    the sample index being the row's position in the collision block (16
    slots per sample); every other row by its label.
    mode "position": every row keyed by its index.
    """
    if mode == "position":
        return [str(i) for i in range(len(labels))]
    if mode == "label":
        return list(labels)
    keys = []
    first = None
    for i, l in enumerate(labels):
        if l.startswith("collision"):
            if first is None:
                first = i
            pair = l[l.index("pair="):]
            keys.append(f"collision#{(i - first) // 16}|{pair}")
        else:
            keys.append(l)
    return keys


def label_map(prev_labels, next_labels, mode="pair"):
    """idx[i] = row of QP k matching row i of QP k+1 (see row_keys), or -1."""
    pk, nk = row_keys(prev_labels, mode), row_keys(next_labels, mode)
    pos = {l: i for i, l in enumerate(pk)}
    assert len(pos) == len(pk), "row keys not unique"
    return np.array([pos.get(l, -1) for l in nk], dtype=np.int64)


def mapped(v, idx, fill):
    out = np.where(idx >= 0, v[np.maximum(idx, 0)], fill)
    return out


def data_slacks(qp, x, G=None):
    """s_l = Gx - h_l, s_u = h_u - Gx on finite sides (0 elsewhere)."""
    if G is None:
        G = scipy_mats(qp)[2]
    Gx = G @ x
    fl, fu = qp["h_l"] > -INF, qp["h_u"] < INF
    return np.where(fl, Gx - qp["h_l"], 0.0), np.where(fu, qp["h_u"] - Gx, 0.0), fl, fu


def clamp_pos(v, f, lo):
    return np.where(f, np.maximum(v, lo), 0.0)


def center_band(s, z, f, mu0, beta):
    """Gondzio-style: move z so that s_i z_i lies in [beta mu0, mu0 / beta]."""
    lo, hi = beta * mu0, mu0 / beta
    zz = np.where(f, np.clip(z, lo / np.maximum(s, 1e-300), hi / np.maximum(s, 1e-300)), 0.0)
    return zz


def build(name, prev, qp, ctx, **kw):
    """ctx: dict with 'idx' (label map), 'G' (QP k+1 G), 'n', 'p'."""
    idx, G = ctx["idx"], ctx["G"]
    n, p = qp["n"], qp["p"]
    s_min = kw.get("s_min", 1e-4)
    z_min = kw.get("z_min", 1e-4)
    z_fill = kw.get("z_fill", 1e-4)
    zl = mapped(prev["z_l"], idx, z_fill)
    zu = mapped(prev["z_u"], idx, z_fill)
    st = {}
    if name == "full_prev":  # control: everything of QP k, s untouched
        sl = mapped(prev["s_l"], idx, 1.0)
        su = mapped(prev["s_u"], idx, 1.0)
        _, _, fl, fu = data_slacks(qp, prev["x"], G)
        start = dict(x=prev["x"], y=prev["y"], s_l=clamp_pos(sl, fl, s_min), s_u=clamp_pos(su, fu, s_min),
                     z_l=clamp_pos(zl, fl, z_min), z_u=clamp_pos(zu, fu, z_min))
        return start, "xysz", st
    if name == "full_prev_xdata":  # control: x_k with slacks recomputed on QP k+1 at x_k
        sl, su, fl, fu = data_slacks(qp, prev["x"], G)
        start = dict(x=prev["x"], y=prev["y"], s_l=clamp_pos(sl, fl, s_min), s_u=clamp_pos(su, fu, s_min),
                     z_l=clamp_pos(zl, fl, z_min), z_u=clamp_pos(zu, fu, z_min))
        return start, "xysz", st
    x0 = np.zeros(n)
    sl, su, fl, fu = data_slacks(qp, x0, G)
    if name == "x0_duals":  # x = 0, duals of QP k, slacks from QP k+1 data at x = 0
        start = dict(x=x0, y=prev["y"], s_l=clamp_pos(sl, fl, s_min), s_u=clamp_pos(su, fu, s_min),
                     z_l=clamp_pos(zl, fl, z_min), z_u=clamp_pos(zu, fu, z_min))
        return start, "xysz", st
    if name == "x0_duals_center":  # + complementarity band around mu0
        mu0, beta = kw.get("mu0", 1e-2), kw.get("beta", 0.1)
        sl2, su2 = clamp_pos(sl, fl, s_min), clamp_pos(su, fu, s_min)
        start = dict(x=x0, y=prev["y"], s_l=sl2, s_u=su2,
                     z_l=center_band(sl2, clamp_pos(zl, fl, z_min), fl, mu0, beta),
                     z_u=center_band(su2, clamp_pos(zu, fu, z_min), fu, mu0, beta))
        return start, "xysz", st
    if name == "x0_prev_slack":  # x = 0, s and z of QP k (assumption: s_k ~ slack of k+1 at 0)
        sl = mapped(prev["s_l"], idx, 1.0)
        su = mapped(prev["s_u"], idx, 1.0)
        start = dict(x=x0, y=prev["y"], s_l=clamp_pos(sl, fl, s_min), s_u=clamp_pos(su, fu, s_min),
                     z_l=clamp_pos(zl, fl, z_min), z_u=clamp_pos(zu, fu, z_min))
        return start, "xysz", st
    if name == "active_set":  # active rows: small s, large z; others the opposite
        a_tol = kw.get("a_tol", 1e-6)
        small, big = kw.get("small", 1e-3), kw.get("big", 1.0)
        act_l = mapped(prev["z_l"], idx, 0.0) > a_tol
        act_u = mapped(prev["z_u"], idx, 0.0) > a_tol
        mk = lambda act, s, z, f: (np.where(f, np.where(act, small, np.maximum(s, big)), 0.0),  # noqa: E731
                                   np.where(f, np.where(act, np.maximum(z, big), small), 0.0))
        s_l, z_l = mk(act_l, sl, zl, fl)
        s_u, z_u = mk(act_u, su, zu, fu)
        start = dict(x=x0, y=prev["y"], s_l=s_l, s_u=s_u, z_l=z_l, z_u=z_u)
        return start, "xysz", st
    if name in ("gg_slack", "gg_data"):
        # Gondzio and Grothey (2002/2008): start from an earlier, well-centered
        # iterate of QP k (prev = state of QP k after N_k - back iterations,
        # supplied by the runner) instead of its optimum. x = 0.
        if name == "gg_slack":
            sl = mapped(prev["s_l"], idx, 1.0)
            su = mapped(prev["s_u"], idx, 1.0)
        start = dict(x=x0, y=prev["y"], s_l=clamp_pos(sl, fl, s_min), s_u=clamp_pos(su, fu, s_min),
                     z_l=clamp_pos(zl, fl, z_min), z_u=clamp_pos(zu, fu, z_min))
        return start, "xysz", st
    if name == "default_xy_prev_z":  # default initial guess for x, y, s; z of QP k
        start = dict(x=x0, y=prev["y"], s_l=sl, s_u=su, z_l=clamp_pos(zl, fl, z_min), z_u=clamp_pos(zu, fu, z_min))
        return start, "z", st
    raise KeyError(name)
