"""
Persistent compute-node pool.

Backlogs persist across decision epochs, so congestion is *endogenous*: a
policy that repeatedly favours one executor degrades that executor's queue and
sees the consequence on later tasks.  Without this feedback a greedy baseline
looks artificially strong.
"""

from __future__ import annotations

import numpy as np

from .config import Config


class NodePool:
    """All executors in the scenario, with persistent per-node backlogs."""

    #: wait time (s) treated as full utilisation when normalising load
    LOAD_REF_S = 0.20

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.nodes = {}

        # Roadside units / MEC servers.  A fraction are third-party operators
        # that are not implicitly trusted.
        for r in range(cfg.n_rsu):
            self.nodes[f"rsu{r}"] = dict(
                kind="rsu",
                f=rng.uniform(cfg.rsu_cpu_lo, cfg.rsu_cpu_hi),
                backlog=0.0,
                served=0.0,
                trusted=rng.random() > cfg.untrusted_frac_rsu,
            )

        # Vehicles, any of which may act as a V2V executor.
        for v in range(cfg.n_vehicles):
            self.nodes[f"v{v}"] = dict(
                kind="v2v",
                f=rng.uniform(cfg.veh_cpu_lo, cfg.veh_cpu_hi),
                backlog=0.0,
                trusted=rng.random() > cfg.untrusted_frac,
                served=0.0,
            )

    def drain(self, dt: float) -> None:
        """Advance time by ``dt`` seconds, draining every node's backlog."""
        for nd in self.nodes.values():
            nd["backlog"] = max(0.0, nd["backlog"] - nd["f"] * dt)

    def wait_s(self, nid: str) -> float:
        """Queueing delay a task would experience at node ``nid``."""
        nd = self.nodes[nid]
        return nd["backlog"] / nd["f"]

    def load(self, nid: str) -> float:
        """Normalised utilisation proxy in [0, 0.97]."""
        return float(np.clip(self.wait_s(nid) / self.LOAD_REF_S, 0.0, 0.97))

    def admit(self, nid: str, cycles: float) -> None:
        """Enqueue ``cycles`` of work at node ``nid``."""
        self.nodes[nid]["backlog"] += cycles
        self.nodes[nid]["served"] += cycles
