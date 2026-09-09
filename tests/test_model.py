"""Unit tests for the physical and cryptographic model components."""

import numpy as np
import pytest

from vecoffload import (Config, MB, NodePool, evaluate, mac_contention,
                        phy_rate_bps, sample_event, zkp_costs_ms)


# ----------------------------------------------------------------- PHY --
def test_phy_rate_decreases_with_distance():
    cfg = Config()
    rates = [phy_rate_bps(cfg, d) for d in (10, 50, 100, 200, 300)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_phy_rate_respects_ceiling():
    cfg = Config()
    ceiling = cfg.phy_cap_mbps * 1e6 * cfg.phy_scale
    assert phy_rate_bps(cfg, 1.0) <= ceiling + 1e-6


# ----------------------------------------------------------------- MAC --
def test_collision_probability_is_monotone_in_density():
    cfg = Config()
    p = [mac_contention(cfg, n)[0] for n in (1, 2, 4, 8, 16)]
    assert p[0] == pytest.approx(0.0, abs=1e-12)
    assert all(a <= b for a, b in zip(p, p[1:]))
    assert all(0.0 <= x <= 1.0 for x in p)


def test_access_delay_grows_and_airtime_share_shrinks():
    cfg = Config()
    delays = [mac_contention(cfg, n)[1] for n in (1, 4, 8, 16)]
    shares = [mac_contention(cfg, n)[2] for n in (1, 4, 8, 16)]
    assert all(a <= b for a, b in zip(delays, delays[1:]))
    assert all(a >= b for a, b in zip(shares, shares[1:]))
    assert shares[0] == pytest.approx(1.0)


def test_contention_can_be_disabled():
    cfg = Config(model_contention=False)
    assert mac_contention(cfg, 20) == (0.0, 0.0, 1.0)


# ----------------------------------------------------------------- ZKP --
def test_zkp_disabled_costs_nothing():
    assert zkp_costs_ms(Config(), 5 * MB, False) == (0.0, 0.0, 0.0, 0.0)


def test_proving_and_verification_are_constant_in_task_size():
    """The central claim of the response to Reviewer 1, comment 2."""
    cfg = Config()
    gens, vers = [], []
    for mb in range(1, 11):
        gen, ver, _, _ = zkp_costs_ms(cfg, mb * MB, True)
        gens.append(gen)
        vers.append(ver)
    assert len(set(gens)) == 1
    assert len(set(vers)) == 1
    assert gens[0] == cfg.zkp_gen_ms
    assert vers[0] == cfg.zkp_ver_ms


def test_commitment_term_is_linear_in_task_size():
    cfg = Config()
    c1 = zkp_costs_ms(cfg, 1 * MB, True)[2]
    c10 = zkp_costs_ms(cfg, 10 * MB, True)[2]
    assert c10 == pytest.approx(10 * c1, rel=1e-9)


def test_total_overhead_is_sublinear_in_task_size():
    cfg = Config()
    t1 = zkp_costs_ms(cfg, 1 * MB, True)[3]
    t10 = zkp_costs_ms(cfg, 10 * MB, True)[3]
    assert t10 < 10 * t1          # sub-linear overall
    assert t10 > t1               # but still growing


# --------------------------------------------------------------- queues --
def test_backlog_accumulates_and_drains():
    cfg = Config()
    pool = NodePool(cfg, np.random.default_rng(0))
    nid = "rsu0"
    assert pool.wait_s(nid) == 0.0
    pool.admit(nid, 1e9)
    assert pool.wait_s(nid) > 0.0
    pool.drain(10.0)
    assert pool.wait_s(nid) == 0.0


def test_load_is_bounded():
    cfg = Config()
    pool = NodePool(cfg, np.random.default_rng(0))
    pool.admit("v0", 1e15)
    assert 0.0 <= pool.load("v0") <= 0.97


# ------------------------------------------------------------- scenario --
def test_event_always_offers_local_execution():
    cfg = Config()
    rng = np.random.default_rng(1)
    pool = NodePool(cfg, rng)
    for _ in range(50):
        _, cands, _ = sample_event(cfg, rng, pool)
        assert cands[0]["kind"] == "local"
        assert np.isinf(cands[0]["tau"])


def test_task_deadline_stays_within_declared_range():
    cfg = Config()
    rng = np.random.default_rng(2)
    pool = NodePool(cfg, rng)
    for _ in range(200):
        task, _, _ = sample_event(cfg, rng, pool)
        assert cfg.deadline_lo_ms / 1000 - 1e-9 <= task["D"]
        assert task["D"] <= cfg.deadline_hi_ms / 1000 + 1e-9
        assert cfg.task_mb_lo <= task["B_mb"] <= cfg.task_mb_hi


def test_verification_only_adds_latency():
    cfg = Config()
    rng = np.random.default_rng(3)
    pool = NodePool(cfg, rng)
    task, cands, _ = sample_event(cfg, rng, pool)
    for c in cands:
        if c["kind"] == "local":
            continue
        assert evaluate(cfg, task, c, True)[0] > evaluate(cfg, task, c, False)[0]
