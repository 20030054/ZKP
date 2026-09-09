"""
Decision-epoch sampling and the Module I feasibility predictor interface.

One "event" is one offloading decision: a requester generates a task, observes
a candidate set (local execution, any in-range RSU, any in-range V2V executor),
and must choose one.
"""

from __future__ import annotations

import numpy as np

from .channel import mac_contention, phy_rate_bps
from .config import Config, MB
from .nodes import NodePool


def sample_event(cfg: Config, rng: np.random.Generator, pool: NodePool):
    """Sample one decision epoch.

    Returns ``(task, candidates, mac_state)``.
    """
    area = cfg.area_m ** 2
    lam_nb = cfg.n_vehicles * np.pi * cfg.comm_range_m ** 2 / area
    n_nb = int(np.clip(rng.poisson(max(lam_nb - 1.0, 0.2)), 0, 14))
    v_ms = cfg.speed_kmh / 3.6

    me = f"v{int(rng.integers(0, cfg.n_vehicles))}"

    # ---- local execution is always available -------------------------- #
    cands = [dict(kind="local", nid=me, dist=0.0, tau=np.inf, price=0.0,
                  trusted=True, rate=np.inf, acc_delay=0.0,
                  rel_speed=0.0, tau_geo=np.inf)]

    # ---- roadside units in range -------------------------------------- #
    for r in range(cfg.n_rsu):
        d = rng.uniform(0.0, cfg.area_m / 2.0)
        if d > cfg.comm_range_m:
            continue
        tau_geo = (cfg.comm_range_m - d) / max(v_ms, 0.5)
        tau = max(min(tau_geo, rng.exponential(cfg.mean_link_life_s())), 0.02)
        cands.append(dict(kind="rsu", nid=f"rsu{r}", dist=d, tau=tau,
                          rel_speed=v_ms, tau_geo=tau_geo,
                          price=rng.uniform(0.6, 1.0),
                          trusted=pool.nodes[f"rsu{r}"]["trusted"]))

    # ---- neighbouring vehicles ---------------------------------------- #
    picked = set()
    for _ in range(n_nb):
        nid = f"v{int(rng.integers(0, cfg.n_vehicles))}"
        if nid == me or nid in picked:
            continue
        picked.add(nid)
        d = rng.uniform(10.0, cfg.comm_range_m)
        # same-direction traffic plus a fraction of oncoming traffic
        rel = abs(rng.normal(0.0, 0.45 * v_ms))
        rel += (1.9 * v_ms if rng.random() < 0.35 else 0.0)
        tau_geo = (cfg.comm_range_m - d) / max(rel, 0.6)
        tau = max(min(tau_geo, rng.exponential(cfg.mean_link_life_s() * 0.55)), 0.02)
        cands.append(dict(kind="v2v", nid=nid, dist=d, tau=tau,
                          rel_speed=rel, tau_geo=tau_geo,
                          price=rng.uniform(0.2, 0.6),
                          trusted=pool.nodes[nid]["trusted"]))

    # ---- MAC contention over the shared channel ----------------------- #
    n_active = 1.0 + (len(cands) - 1) * float(
        np.clip(cfg.arrival_rate * 0.55, 0.05, 1.0))
    p_coll, acc_delay, share = mac_contention(cfg, n_active)
    for c in cands:
        if c["kind"] != "local":
            c["rate"] = max(
                phy_rate_bps(cfg, c["dist"]) * (1.0 - p_coll) * share, 1e5)
            c["acc_delay"] = acc_delay
        c["f"] = pool.nodes[c["nid"]]["f"]
        c["wait"] = pool.wait_s(c["nid"])
        c["load"] = pool.load(c["nid"])

    # ---- task --------------------------------------------------------- #
    b_mb = rng.uniform(cfg.task_mb_lo, cfg.task_mb_hi)
    b_bits = b_mb * MB
    deadline_ms = float(np.clip(
        b_mb * cfg.deadline_per_mb_ms * rng.uniform(0.85, 1.35),
        cfg.deadline_lo_ms, cfg.deadline_hi_ms))
    task = dict(B=b_bits, B_mb=b_mb, Bout=b_bits * cfg.out_ratio,
                C=b_bits * cfg.cycles_per_bit, D=deadline_ms / 1000.0)

    mac = dict(p_coll=p_coll, acc_delay=acc_delay, share=share, n_c=n_active)
    return task, cands, mac


def pair_features(cfg: Config, task: dict, cand: dict, mac: dict, est_latency: float):
    """Feature vector for the Module I feasibility predictor h_omega(obs_ij)."""
    return [
        cand["dist"] / 300.0,
        cand["rel_speed"] / 30.0,
        min(cand["tau_geo"], 30.0) / 30.0,
        np.log(min(cand["rate"], 1e12)) / 25.0,
        cand["wait"] * 10.0,
        cand["f"] / 2e10,
        task["B_mb"] / 10.0,
        task["D"] * 10.0,
        est_latency / max(task["D"], 1e-3),
        mac["p_coll"],
        mac["n_c"] / 15.0,
        1.0 if cand["kind"] == "rsu" else 0.0,
    ]


def predict(cfg: Config, rng: np.random.Generator, cands):
    """Module I: noisy short-horizon estimates of tau, R and w.

    Multiplicative log-normal noise with relative standard deviation
    ``cfg.pred_sigma`` stands in for a deployed predictor's error.  Realised
    outcomes are always computed from the true quantities, so prediction error
    can degrade the *choice* but can never relax a constraint.
    """
    sigma = cfg.pred_sigma
    out = []
    for c in cands:
        ch = dict(c)
        if sigma > 0:
            n_tau, n_rate, n_wait = np.exp(rng.normal(0.0, sigma, 3))
            ch["tau"] = c["tau"] * n_tau
            if c["kind"] != "local":
                ch["rate"] = c["rate"] * n_rate
            ch["wait"] = c["wait"] * n_wait
            ch["load"] = float(np.clip(c["load"] * n_wait, 0.0, 0.97))
        out.append(ch)
    return out
