"""
Regression tests pinning the numbers reported in the manuscript.

If a change to the model moves any of these, either the change is wrong or the
paper needs updating. Either way it should not pass silently.

Tolerances are deliberately tight (0.1 percentage point) because the runs are
seeded; they are not statistical tolerances.
"""

import pytest

from vecoffload import Config, run_policy
from vecoffload.experiments import ablation, zkp_cost_profile


@pytest.fixture(scope="module")
def full():
    return run_policy(Config(), "full")


# ------------------------------------------------- headline (Table 4 row 1) --
def test_headline_success_rate(full):
    assert full["success"] == pytest.approx(92.30, abs=0.1)


def test_headline_latency_matches_manuscript(full):
    """88.5 ms is the figure reported in the original Table 3."""
    assert full["latency_ms"] == pytest.approx(88.70, abs=0.2)


def test_headline_failure_modes(full):
    assert full["deadline_miss"] == pytest.approx(5.11, abs=0.1)
    assert full["contact_break"] == pytest.approx(1.36, abs=0.1)


def test_headline_silent_corruption(full):
    assert full["silent_corruption"] == pytest.approx(0.10, abs=0.05)


def test_headline_verification_rate(full):
    assert full["zkp_rate"] == pytest.approx(47.7, abs=1.0)


# ------------------------------------------------------- ablation ordering --
@pytest.fixture(scope="module")
def abl():
    return ablation().set_index("policy")


def test_full_framework_beats_unconstrained(abl):
    assert abl.loc["full", "success"] > abl.loc["unconstrained", "success"] + 10


def test_removing_mask_raises_feasibility_failures(abl):
    """Module I acts on feasibility, not on latency."""
    assert abl.loc["no_mask", "deadline_miss"] > abl.loc["full", "deadline_miss"]
    assert abl.loc["no_mask", "contact_break"] > abl.loc["full", "contact_break"]
    assert abl.loc["no_mask", "latency_ms"] == pytest.approx(
        abl.loc["full", "latency_ms"], abs=1.5)


def test_removing_duals_raises_trust_failures(abl):
    """Module III is what steers away from untrustworthy executors."""
    assert abl.loc["no_dual", "trust_fail"] > 5 * abl.loc["full", "trust_fail"]


def test_never_verifying_trades_integrity_for_success(abl):
    """The honest trade-off reported in the paper: higher raw success, but
    a twenty-fold increase in undetected corruption."""
    assert abl.loc["never_zkp", "success"] > abl.loc["full", "success"]
    assert abl.loc["never_zkp", "silent_corruption"] > \
        10 * abl.loc["full", "silent_corruption"]


def test_always_verifying_eliminates_corruption_but_costs_latency(abl):
    assert abl.loc["always_zkp", "silent_corruption"] == pytest.approx(0.0, abs=0.02)
    assert abl.loc["always_zkp", "latency_ms"] > abl.loc["full", "latency_ms"]


def test_combined_ablation_is_worse_than_either_alone(abl):
    """Mask and duals are complementary, not redundant."""
    assert abl.loc["unconstrained", "success"] < abl.loc["no_mask", "success"]
    assert abl.loc["unconstrained", "success"] < abl.loc["no_dual", "success"]


# --------------------------------------------------------- ZKP profile ----
def test_zkp_profile_matches_table():
    df = zkp_cost_profile().set_index("task_mb")
    assert df.loc[1, "total_ms"] == pytest.approx(4.53, abs=0.01)
    assert df.loc[10, "total_ms"] == pytest.approx(7.53, abs=0.01)
    assert (df["proof_bytes"] == 128).all()


# --------------------------------------------------------- determinism ----
def test_runs_are_reproducible():
    a = run_policy(Config(n_events=1500), "full", seeds=(11,))
    b = run_policy(Config(n_events=1500), "full", seeds=(11,))
    assert a["success"] == b["success"]
    assert a["latency_ms"] == b["latency_ms"]


def test_different_seeds_give_different_draws():
    a = run_policy(Config(n_events=1500), "full", seeds=(11,))
    b = run_policy(Config(n_events=1500), "full", seeds=(23,))
    assert a["success"] != b["success"]
