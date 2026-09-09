"""
Per-candidate latency, energy and monetary cost accounting.
"""

from __future__ import annotations

from typing import Tuple

from .config import Config, GHZ
from .zkp import zkp_costs_ms


def evaluate(cfg: Config, task: dict, cand: dict, verify: bool) -> Tuple[float, float, float, float, float]:
    """Cost of executing ``task`` on ``cand``.

    Returns ``(latency_s, energy_J, monetary, zkp_total_ms, zkp_verify_ms)``.

    Energy is accounted system-wide: requester uplink transmission, requester
    idle listening while awaiting the result, and executor compute power times
    execution time.
    """
    if cand["kind"] == "local":
        t_comm, e_comm = 0.0, 0.0
    else:
        t_up = task["B"] / cand["rate"]
        t_dn = task["Bout"] / cand["rate"]
        t_comm = t_up + t_dn + cand["acc_delay"]
        e_comm = cfg.tx_power_w * (t_up + cand["acc_delay"])

    t_comp = task["C"] / max(cand["f"], 1e6)
    _, ver_ms, _, tot_ms = zkp_costs_ms(cfg, task["B"], verify)

    latency = t_comm + cand["wait"] + t_comp + tot_ms / 1000.0

    p_comp = cfg.rsu_compute_w if cand["kind"] == "rsu" else cfg.veh_compute_w
    if cand["kind"] == "local":
        energy = p_comp * t_comp
    else:
        energy = e_comm + cfg.idle_power_w * latency + p_comp * t_comp

    monetary = cand["price"] * (task["C"] / GHZ) * 1e-3
    return latency, energy, monetary, tot_ms, ver_ms
