"""
Every experiment reported in the manuscript, one function each.

Mapping to the paper
--------------------
=========================  ===================================  ==================
Function                   Paper artefact                       Reviewer comment
=========================  ===================================  ==================
``ablation``               Section 5.3, Table 4                 R2-1
``prediction_error``       Section 5.4, Fig. 9(a), Table 5      R1-1
``safety_margin``          Section 5.4, Fig. 9(b), Table 5      R1-1
``contention``             Section 5.5, Fig. 10                 R1-3
``zkp_cost_profile``       Section 5.6, Table 6                 R1-2
``predictor_capacity``     Section 5.7, Table 7                 R2-3
``dual_step_size``         Section 5.7, Table 8                 R2-3
``verification_budget``    Section 5.7, Fig. 11                 R2-3
``regime_configurations``  Section 5.7, Table 9                 R2-3
=========================  ===================================  ==================

Each function returns a :class:`pandas.DataFrame` and is independent of the
others, so a single experiment can be re-run without repeating the suite.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import Config, MB
from .policy import ABLATION, DEFAULT_SEEDS, run_policy
from .zkp import zkp_profile

DEFAULT_EVENTS = 8000


# ---------------------------------------------------------------------- #
# R2-1 : ablation study
# ---------------------------------------------------------------------- #
def ablation(n_events: int = DEFAULT_EVENTS, seeds=DEFAULT_SEEDS) -> pd.DataFrame:
    """Disable one framework component at a time, holding everything else fixed."""
    rows = []
    for label, policy in ABLATION:
        r = run_policy(Config(n_events=n_events), policy, seeds=seeds)
        rows.append(dict(
            variant=label, policy=policy,
            success=r["success"], success_sd=r["_std"]["success"],
            latency_ms=r["latency_ms"], energy_j=r["energy_j"],
            deadline_miss=r["deadline_miss"], contact_break=r["contact_break"],
            silent_corruption=r["silent_corruption"], trust_fail=r["trust_fail"],
            jain=r["jain"], zkp_rate=r["zkp_rate"],
            zkp_overhead_ms=r["zkp_amortised_ms"],
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# R1-1 : robustness to imperfect prediction
# ---------------------------------------------------------------------- #
def prediction_error(sigmas=(0.0, 0.05, 0.10, 0.20, 0.30),
                     n_events: int = DEFAULT_EVENTS, seeds=DEFAULT_SEEDS) -> pd.DataFrame:
    """Inject multiplicative prediction noise into tau-hat, R-hat and w-hat."""
    rows = []
    for sigma in sigmas:
        r = run_policy(Config(n_events=n_events, pred_sigma=sigma), "full", seeds=seeds)
        rows.append(dict(sigma_pct=100 * sigma, success=r["success"],
                         success_sd=r["_std"]["success"],
                         latency_ms=r["latency_ms"],
                         deadline_miss=r["deadline_miss"],
                         contact_break=r["contact_break"]))
    return pd.DataFrame(rows)


def safety_margin(kappas=(1.0, 0.9, 0.8, 0.7, 0.6), sigma: float = 0.20,
                  n_events: int = DEFAULT_EVENTS, seeds=DEFAULT_SEEDS) -> pd.DataFrame:
    """Effect of the contact-feasibility safety margin at fixed prediction error."""
    rows = []
    for kappa in kappas:
        cfg = Config(n_events=n_events, pred_sigma=sigma, safety_kappa=kappa)
        r = run_policy(cfg, "full", seeds=seeds)
        rows.append(dict(kappa=kappa, success=r["success"],
                         latency_ms=r["latency_ms"],
                         deadline_miss=r["deadline_miss"],
                         contact_break=r["contact_break"],
                         offload_ratio=r["offload_ratio"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# R1-3 : MAC-layer contention and scalability
# ---------------------------------------------------------------------- #
def contention(densities=(30, 45, 60, 75, 90, 105, 120),
               n_events: int = DEFAULT_EVENTS, seeds=DEFAULT_SEEDS) -> pd.DataFrame:
    """Sweep vehicle density with and without the EDCA contention model.

    The contention-free arm isolates the MAC-layer contribution from compute
    congestion.
    """
    rows = []
    for n in densities:
        on = run_policy(Config(n_events=n_events, n_vehicles=n,
                               model_contention=True), "full", seeds=seeds)
        off = run_policy(Config(n_events=n_events, n_vehicles=n,
                                model_contention=False), "full", seeds=seeds)
        rows.append(dict(n_vehicles=n,
                         contenders=on["n_contenders"],
                         p_coll=on["p_coll"],
                         mac_delay_ms=on["mac_delay_ms"],
                         success_contention=on["success"],
                         success_ideal=off["success"],
                         latency_contention=on["latency_ms"],
                         latency_ideal=off["latency_ms"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# R1-2 : ZKP cost profile
# ---------------------------------------------------------------------- #
def zkp_cost_profile(task_mb=range(1, 11)) -> pd.DataFrame:
    """Groth16/BN254 profile across task sizes (analytic, no simulation)."""
    return pd.DataFrame(zkp_profile(Config(), list(task_mb)))


# ---------------------------------------------------------------------- #
# R2-3 : hyper-parameter sensitivity
# ---------------------------------------------------------------------- #
def _predictor_dataset(n: int = 30000, seed: int = 3):
    """Harvest a feasibility-classification dataset from the simulator."""
    from .cost import evaluate
    from .nodes import NodePool
    from .scenario import pair_features, sample_event

    cfg = Config()
    rng = np.random.default_rng(seed)
    pool = NodePool(cfg, rng)
    dt = 1.0 / (cfg.arrival_rate * cfg.n_vehicles * cfg.requester_frac)

    X, y = [], []
    while len(X) < n:
        task, cands, mac = sample_event(cfg, rng, pool)
        for c in cands:
            if c["kind"] == "local":
                continue
            latency = evaluate(cfg, task, c, False)[0]
            X.append(pair_features(cfg, task, c, mac, latency))
            y.append(int(latency <= task["D"] and latency <= c["tau"]))
        k = int(rng.integers(0, len(cands)))
        pool.admit(cands[k]["nid"], task["C"])
        pool.drain(dt)
    return np.asarray(X[:n]), np.asarray(y[:n])


def predictor_capacity(n_events: int = 4000, seeds=(11, 23, 47)) -> pd.DataFrame:
    """Train feasibility classifiers of increasing depth and run each in the loop."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    X, y = _predictor_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)

    architectures = [
        ("1 hidden layer (32)", (32,)),
        ("2 hidden layers (64-32)", (64, 32)),
        ("3 hidden layers (64-32-16)", (64, 32, 16)),
        ("4 hidden layers (128-64-32-16)", (128, 64, 32, 16)),
    ]
    rows = []
    for label, hidden in architectures:
        clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                            max_iter=300, random_state=0,
                            early_stopping=True, n_iter_no_change=15)
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        train_s = time.time() - t0

        t0 = time.time()
        pred = clf.predict(X_te)
        infer_us = (time.time() - t0) / len(X_te) * 1e6

        r = run_policy(Config(n_events=n_events), "full",
                       seeds=seeds, predictor=clf)
        rows.append(dict(
            arch=label,
            n_params=int(sum(w.size for w in clf.coefs_)),
            accuracy=100 * accuracy_score(y_te, pred),
            precision=100 * precision_score(y_te, pred, zero_division=0),
            recall=100 * recall_score(y_te, pred, zero_division=0),
            f1=100 * f1_score(y_te, pred, zero_division=0),
            train_s=train_s, infer_us=infer_us,
            success=r["success"], latency_ms=r["latency_ms"],
            deadline_miss=r["deadline_miss"],
            contact_break=r["contact_break"]))
    return pd.DataFrame(rows)


def dual_step_size(etas=(1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0),
                   n_events: int = DEFAULT_EVENTS, seeds=DEFAULT_SEEDS) -> pd.DataFrame:
    """Sweep the Lagrangian step size over four orders of magnitude.

    ``violation_swing`` is the standard deviation of the violation rate over the
    final eight evaluation windows, and measures oscillation.
    """
    rows = []
    for eta in etas:
        r = run_policy(Config(n_events=n_events, eta_lambda=eta), "full", seeds=seeds)
        trace = np.asarray(r["viol_trace"])
        rows.append(dict(eta=eta, success=r["success"],
                         latency_ms=r["latency_ms"],
                         deadline_miss=r["deadline_miss"],
                         contact_break=r["contact_break"],
                         violation_final=float(trace[-1]),
                         violation_swing=float(np.std(trace[-8:]))))
    return pd.DataFrame(rows)


def verification_budget(budgets=(2, 4, 5, 6, 7, 8, 10, 12),
                        n_events: int = DEFAULT_EVENTS, seeds=DEFAULT_SEEDS) -> pd.DataFrame:
    """Sweep the per-task verification budget."""
    rows = []
    for b in budgets:
        r = run_policy(Config(n_events=n_events, zkp_budget_ms=float(b)),
                       "full", seeds=seeds)
        rows.append(dict(budget_ms=b, success=r["success"],
                         latency_ms=r["latency_ms"],
                         zkp_rate=r["zkp_rate"],
                         zkp_overhead_ms=r["zkp_amortised_ms"],
                         silent_corruption=r["silent_corruption"],
                         budget_violation=r["budget_violation"]))
    return pd.DataFrame(rows)


def regime_configurations(n_events: int = 4000, seeds=(11, 23, 47),
                          return_grid: bool = False):
    """Grid-search the recommended operating point in three traffic regimes."""
    regimes = [
        ("Sparse / high speed (30 veh, 60 km/h)", dict(n_vehicles=30, speed_kmh=60)),
        ("Medium (60 veh, 40 km/h)", dict(n_vehicles=60, speed_kmh=40)),
        ("Dense / congested (120 veh, 25 km/h)", dict(n_vehicles=120, speed_kmh=25)),
    ]
    grid_eta = (1e-3, 1e-2, 5e-2)
    grid_budget = (6.0, 8.0, 12.0)
    grid_kappa = (1.0, 0.8)

    all_rows, best_rows = [], []
    for label, kw in regimes:
        best = None
        for eta in grid_eta:
            for budget in grid_budget:
                for kappa in grid_kappa:
                    cfg = Config(n_events=n_events, eta_lambda=eta,
                                 zkp_budget_ms=budget, safety_kappa=kappa, **kw)
                    r = run_policy(cfg, "full", seeds=seeds)
                    utility = (r["success"] - 0.04 * r["latency_ms"]
                               - 1.5 * r["silent_corruption"])
                    row = dict(regime=label, eta_lambda=eta, budget_ms=budget,
                               kappa=kappa, success=r["success"],
                               latency_ms=r["latency_ms"],
                               deadline_miss=r["deadline_miss"],
                               contact_break=r["contact_break"],
                               silent_corruption=r["silent_corruption"],
                               utility=utility)
                    all_rows.append(row)
                    if best is None or utility > best["utility"]:
                        best = row
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows)
    if return_grid:
        return best_df, pd.DataFrame(all_rows)
    return best_df


#: name -> callable, used by the CLI
REGISTRY = {
    "ablation": ablation,
    "prediction_error": prediction_error,
    "safety_margin": safety_margin,
    "contention": contention,
    "zkp_cost_profile": zkp_cost_profile,
    "predictor_capacity": predictor_capacity,
    "dual_step_size": dual_step_size,
    "verification_budget": verification_budget,
    "regime_configurations": regime_configurations,
}
