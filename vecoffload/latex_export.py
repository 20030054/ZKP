"""
Convert experiment DataFrames into the LaTeX table rows and pgfplots coordinate
blocks that appear in the manuscript.

This keeps the paper and the code in sync: regenerate a result, re-export, and
paste. Because the manuscript now ships pre-rendered figure PDFs (to keep the
Overleaf compile inside the free-tier timeout), the coordinate blocks are
primarily useful for regenerating those figures from source.
"""

from __future__ import annotations


def coords(xs, ys, precision: int = 2) -> str:
    """Format an (x, y) sequence as a pgfplots ``coordinates {...}`` body."""
    return " ".join(f"({x:g},{y:.{precision}f})" for x, y in zip(xs, ys))


def ablation_rows(df) -> str:
    out = []
    for _, r in df.iterrows():
        out.append(
            f"{r.variant} & {r.success:.1f} & {r.latency_ms:.1f} & "
            f"{r.deadline_miss:.2f} & {r.contact_break:.2f} & "
            f"{r.silent_corruption:.2f} & {r.trust_fail:.2f} & "
            f"{r.zkp_rate:.0f} & {r.zkp_overhead_ms:.2f} \\\\")
    return "\n".join(out)


def safety_margin_rows(df) -> str:
    out = []
    for _, r in df.iterrows():
        out.append(f"{r.kappa:.1f} & {r.success:.2f} & {r.deadline_miss:.2f} & "
                   f"{r.contact_break:.2f} & {r.latency_ms:.2f} & "
                   f"{r.offload_ratio:.1f} \\\\")
    return "\n".join(out)


def zkp_rows(df, sizes=(1, 2, 4, 6, 8, 10)) -> str:
    out = []
    for _, r in df.iterrows():
        if int(r.task_mb) in sizes:
            out.append(f"{r.task_mb:.0f} & {r.gen_ms:.2f} & {r.verify_ms:.2f} & "
                       f"{r.commit_ms:.2f} & {r.total_ms:.2f} & "
                       f"{int(r.proof_bytes)} \\\\")
    return "\n".join(out)


def predictor_rows(df) -> str:
    out = []
    for _, r in df.iterrows():
        out.append(f"{r.arch} & {int(r.n_params)} & {r.accuracy:.2f} & "
                   f"{r.f1:.2f} & {r.infer_us:.2f} & {r.success:.2f} \\\\")
    return "\n".join(out)


def eta_rows(df) -> str:
    out = []
    for _, r in df.iterrows():
        out.append(f"{r.eta:g} & {r.success:.2f} & {r.latency_ms:.2f} & "
                   f"{r.deadline_miss:.2f} & {r.violation_final:.2f} & "
                   f"{r.violation_swing:.3f} \\\\")
    return "\n".join(out)


def regime_rows(df) -> str:
    out = []
    for _, r in df.iterrows():
        out.append(f"{r.regime} & {r.eta_lambda:g} & {r.budget_ms:.0f} & "
                   f"{r.kappa:.1f} & {r.success:.2f} & {r.deadline_miss:.2f} \\\\")
    return "\n".join(out)


def prediction_error_plots(df) -> str:
    return "\n".join([
        "success:      " + coords(df.sigma_pct, df.success),
        "deadline:     " + coords(df.sigma_pct, df.deadline_miss),
        "contact:      " + coords(df.sigma_pct, df.contact_break),
    ])


def contention_plots(df) -> str:
    return "\n".join([
        "p_coll:       " + coords(df.n_vehicles, df.p_coll),
        "mac_delay:    " + coords(df.n_vehicles, df.mac_delay_ms, 3),
        "succ_cont:    " + coords(df.n_vehicles, df.success_contention),
        "succ_ideal:   " + coords(df.n_vehicles, df.success_ideal),
    ])


def budget_plots(df) -> str:
    return "\n".join([
        "verified:     " + coords(df.budget_ms, df.zkp_rate),
        "silent:       " + coords(df.budget_ms, df.silent_corruption, 3),
        "overhead:     " + coords(df.budget_ms, df.zkp_overhead_ms, 3),
    ])


EXPORTERS = {
    "ablation": ablation_rows,
    "safety_margin": safety_margin_rows,
    "zkp_cost_profile": zkp_rows,
    "predictor_capacity": predictor_rows,
    "dual_step_size": eta_rows,
    "regime_configurations": regime_rows,
    "prediction_error": prediction_error_plots,
    "contention": contention_plots,
    "verification_budget": budget_plots,
}
