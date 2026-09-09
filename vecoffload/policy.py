"""
The offloading policy and its ablation variants.

Three components can be independently disabled, which is what the ablation
study in the manuscript exercises:

* **Module I**  -- feasibility prediction and action masking.
* **Module II** -- budgeted ZKP verification as a controllable action z_i(t).
* **Module III**-- Lagrangian (primal-dual) multiplier updates on realised
  constraint signals.

Variants
--------
``full``           all three modules active
``no_mask``        Module I disabled (policy may sample any reachable candidate)
``never_zkp``      Module II degenerate: z_i(t) == 0
``always_zkp``     Module II degenerate: z_i(t) == 1
``no_dual``        Module III disabled (multipliers pinned at zero)
``unconstrained``  Modules I and III both disabled
``heuristic``      greedy on *apparent instantaneous* latency (ignores queueing)
``random``         uniform over reachable candidates
"""

from __future__ import annotations

import numpy as np

from .config import Config, GHZ
from .cost import evaluate
from .nodes import NodePool
from .scenario import pair_features, predict, sample_event
from .zkp import zkp_costs_ms

POLICIES = ("full", "no_mask", "no_dual", "always_zkp", "never_zkp",
            "unconstrained", "heuristic", "random")

#: variants for the ablation table, in presentation order
ABLATION = (
    ("Proposed (full framework)", "full"),
    ("w/o Module I (no feasibility mask)", "no_mask"),
    ("w/o Module II (never verify)", "never_zkp"),
    ("w/o Module II (always verify)", "always_zkp"),
    ("w/o Module III (no Lagrangian duals)", "no_dual"),
    ("w/o Modules I+III (unconstrained RL)", "unconstrained"),
)


class Runner:
    """Executes one policy variant over a sequence of decision epochs."""

    def __init__(self, cfg: Config, policy: str = "full", predictor=None):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
        self.cfg = cfg
        self.policy = policy
        self.predictor = predictor
        self.use_mask = policy in ("full", "no_dual", "always_zkp", "never_zkp")
        self.use_dual = policy in ("full", "no_mask", "always_zkp", "never_zkp")
        self.lam = {k: (cfg.lam_init if self.use_dual else 0.0) for k in cfg.limits}

    # ------------------------------------------------------------------ #
    # Module II: verification decision
    # ------------------------------------------------------------------ #
    def _verify_decision(self, cfg, task, cand, slack_s) -> int:
        if self.policy == "always_zkp":
            return 1
        if self.policy in ("never_zkp", "random", "heuristic"):
            return 0
        *_, total_ms = zkp_costs_ms(cfg, task["B"], True)
        if total_ms > cfg.zkp_budget_ms:
            return 0                      # would overrun the budget
        if cand["kind"] == "local":
            return 0                      # nothing to verify
        if slack_s < total_ms / 1000.0:
            return 0                      # would overrun the deadline
        return 1

    # ------------------------------------------------------------------ #
    # Module I: feasible-set construction
    # ------------------------------------------------------------------ #
    def _feasible(self, cfg, task, pred, est, mac):
        if self.predictor is None:
            return [k for k, ch in enumerate(pred)
                    if est[k][0] <= task["D"]
                    and est[k][0] <= cfg.safety_kappa * ch["tau"]]

        # learned classifier drives the mask
        feats, idx = [], []
        for k, ch in enumerate(pred):
            if ch["kind"] == "local":
                continue
            feats.append(pair_features(cfg, task, ch, mac, est[k][0]))
            idx.append(k)
        feasible = []
        if feats:
            proba = self.predictor.predict_proba(np.asarray(feats))[:, 1]
            threshold = 0.5 + 0.35 * (1.0 - cfg.safety_kappa)
            feasible = [idx[j] for j in range(len(idx)) if proba[j] >= threshold]
        if est[0][0] <= task["D"]:
            feasible = [0] + feasible
        return feasible

    # ------------------------------------------------------------------ #
    def step(self, rng, pool, st, dt):
        cfg = self.cfg
        task, cands, mac = sample_event(cfg, rng, pool)
        pred = predict(cfg, rng, cands)
        est = [evaluate(cfg, task, ch, False) for ch in pred]

        feasible = self._feasible(cfg, task, pred, est, mac)
        allowed = feasible if (self.use_mask and feasible) else list(range(len(pred)))

        # ---- action selection ----------------------------------------- #
        if self.policy == "random":
            k = int(rng.choice(allowed))
        elif self.policy == "heuristic":
            # myopic: scores on apparent instantaneous latency, ignoring the
            # queueing backlog, and therefore self-congests popular nodes
            myopic = [evaluate(cfg, task, dict(ch, wait=0.0), False)[0] for ch in pred]
            k = int(min(allowed, key=lambda j: myopic[j]))
        else:
            scores = []
            for j in allowed:
                latency, energy, monetary = est[j][0], est[j][1], est[j][2]
                base = (cfg.alpha * (latency / task["D"])
                        + cfg.beta * energy + cfg.chi * monetary)
                penalty = 0.0
                if self.use_dual:
                    g_dl = max(0.0, latency / task["D"] - 1.0)
                    g_ct = max(0.0, latency / max(pred[j]["tau"], 1e-3) - 1.0)
                    g_rel = 0.0 if pred[j]["trusted"] else 0.5
                    g_fair = pred[j]["load"]
                    penalty = (self.lam["dl"] * g_dl + self.lam["ct"] * g_ct
                               + self.lam["rel"] * g_rel + self.lam["fair"] * g_fair)
                scores.append(-(base + penalty))
            scores = np.asarray(scores)
            probs = np.exp((scores - scores.max()) / cfg.temperature)
            k = int(rng.choice(allowed, p=probs / probs.sum()))

        chosen_pred, chosen_true = pred[k], cands[k]
        slack = task["D"] - est[k][0]
        z = self._verify_decision(cfg, task, chosen_pred, slack)

        # ---- realised outcome ------------------------------------------ #
        latency, energy, monetary, tot_ms, ver_ms = evaluate(cfg, task, chosen_true, z)
        pool.admit(chosen_true["nid"], task["C"])

        miss_dl = latency > task["D"]
        miss_ct = latency > chosen_true["tau"]
        bad = not chosen_true["trusted"]
        corrupt = bool(bad and rng.random() < cfg.zkp_fail_prob)

        caught = undetected = recovered = False
        if corrupt and z:
            # verification rejects the result; fall back to a local retry when
            # the remaining deadline slack still permits it
            caught = True
            t_retry = task["C"] / cands[0]["f"]
            recovered = (latency + t_retry) <= task["D"]
            if recovered:
                latency = latency + t_retry
        elif corrupt and not z:
            undetected = True

        success = ((not miss_dl) and (not miss_ct)
                   and (not undetected) and ((not caught) or recovered))

        # ---- bookkeeping ------------------------------------------------ #
        st["n"] += 1
        st["succ"] += success
        st["dl"] += miss_dl
        st["ct"] += miss_ct
        st["tf"] += int(undetected or (caught and not recovered))
        st["caught"] += int(caught)
        st["recov"] += int(recovered)
        st["silent"] += int(undetected)
        st["ver_tot"] += tot_ms
        st["ver_only"] += ver_ms
        st["ver_n"] += z
        st["bviol"] += int(tot_ms > cfg.zkp_budget_ms + 1e-9)
        st["cost"] += monetary
        st["p_coll"] += mac["p_coll"]
        st["acc"] += mac["acc_delay"]
        st["nc"] += mac["n_c"]
        st["energy"] += energy
        if success:
            st["lat"] += latency
            st["lat_n"] += 1
            st["e_ok"] += energy
        if chosen_true["kind"] != "local":
            st["util"][chosen_true["nid"]] = (
                st["util"].get(chosen_true["nid"], 0.0) + task["C"] / GHZ)
            st["off"] += 1
        if chosen_true["kind"] == "rsu":
            st["rsu_util"][chosen_true["nid"]] = (
                st["rsu_util"].get(chosen_true["nid"], 0.0) + task["C"] / GHZ)
            if chosen_true["trusted"]:
                st["rsu_ok"][chosen_true["nid"]] = (
                    st["rsu_ok"].get(chosen_true["nid"], 0.0) + task["C"] / GHZ)

        # ---- Module III: primal-dual multiplier update ------------------ #
        if self.use_dual:
            g = dict(dl=float(miss_dl), ct=float(miss_ct),
                     ver=float(tot_ms > cfg.zkp_budget_ms),
                     rel=float(not success),
                     fair=float(chosen_true["load"] > 0.70))
            for key, limit in cfg.limits.items():
                self.lam[key] = max(0.0, self.lam[key]
                                    + cfg.eta_lambda * (g[key] - limit))

        pool.drain(dt)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _blank_stats():
        return dict(n=0, succ=0, dl=0, ct=0, tf=0, ver_tot=0.0, ver_only=0.0,
                    ver_n=0, bviol=0, cost=0.0, p_coll=0.0, acc=0.0, nc=0.0,
                    energy=0.0, lat=0.0, lat_n=0, e_ok=0.0, off=0,
                    util={}, rsu_util={}, rsu_ok={}, caught=0, recov=0, silent=0)

    def run(self, seed=None):
        """Run one episode and return summary metrics."""
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed if seed is None else seed)
        pool = NodePool(cfg, rng)
        arrival = max(cfg.arrival_rate * cfg.n_vehicles * cfg.requester_frac, 1e-3)
        dt = 1.0 / arrival

        st = self._blank_stats()
        warmup = max(400, cfg.n_events // 8)
        trace = []
        for t in range(cfg.n_events + warmup):
            self.step(rng, pool, st, dt)
            if t == warmup - 1:
                st = self._blank_stats()      # discard warm-up transient
            if t >= warmup and (t - warmup) % 250 == 249:
                trace.append(100.0 * (1.0 - st["succ"] / max(st["n"], 1)))
        return self._summarise(st, trace)

    def _summarise(self, s, trace):
        n = max(s["n"], 1)

        def jain(values, n_slots=None):
            v = np.asarray(list(values), dtype=float)
            slots = len(v) if n_slots is None else n_slots
            if slots == 0 or v.sum() <= 0:
                return 0.0
            if n_slots is not None and len(v) < n_slots:
                v = np.concatenate([v, np.zeros(n_slots - len(v))])
            return float((v.sum() ** 2) / (slots * (v ** 2).sum()))

        return dict(
            success=100.0 * s["succ"] / n,
            latency_ms=1000.0 * s["lat"] / max(s["lat_n"], 1),
            energy_j=s["e_ok"] / max(s["lat_n"], 1),
            deadline_miss=100.0 * s["dl"] / n,
            contact_break=100.0 * s["ct"] / n,
            trust_fail=100.0 * s["tf"] / n,
            silent_corruption=100.0 * s["silent"] / n,
            caught_rate=100.0 * s["caught"] / n,
            recovered_rate=100.0 * s["recov"] / max(s["caught"], 1),
            jain=jain(s["rsu_ok"].values(), max(len(s["rsu_ok"]), 1)),
            jain_all=jain(s["util"].values()),
            offload_ratio=100.0 * s["off"] / n,
            zkp_rate=100.0 * s["ver_n"] / n,
            zkp_verify_ms=s["ver_only"] / max(s["ver_n"], 1),
            zkp_total_ms=s["ver_tot"] / max(s["ver_n"], 1),
            zkp_amortised_ms=s["ver_tot"] / n,
            budget_violation=100.0 * s["bviol"] / n,
            p_coll=100.0 * s["p_coll"] / n,
            mac_delay_ms=1000.0 * s["acc"] / n,
            n_contenders=s["nc"] / n,
            viol_trace=trace,
        )


DEFAULT_SEEDS = (11, 23, 47, 71, 97)


def run_policy(cfg: Config, policy: str, seeds=DEFAULT_SEEDS, predictor=None):
    """Run ``policy`` across several seeds and return mean metrics.

    The per-metric standard deviation across seeds is returned under ``_std``.
    """
    outs = [Runner(cfg, policy, predictor).run(seed=s) for s in seeds]
    keys = [k for k in outs[0] if k != "viol_trace"]
    res = {k: float(np.mean([o[k] for o in outs])) for k in keys}
    res["_std"] = {k: float(np.std([o[k] for o in outs])) for k in keys}
    shortest = min(len(o["viol_trace"]) for o in outs)
    res["viol_trace"] = np.mean(
        [o["viol_trace"][:shortest] for o in outs], axis=0).tolist()
    return res
