"""
PHY and MAC abstractions.

The PHY term is a log-distance path-loss / Shannon abstraction capped at the
802.11p rate ceiling.  The MAC term is an IEEE 802.11p EDCA model using the
standard Bianchi saturation approximation, added during revision in response
to a reviewer question about collision and backoff dynamics at high density.
"""

from __future__ import annotations

import numpy as np

from .config import Config


def phy_rate_bps(cfg: Config, dist_m: float) -> float:
    """Achievable PHY rate over a link of length ``dist_m``.

    Log-distance path loss feeds a Shannon capacity, capped at the 802.11p
    ceiling and then scaled by ``cfg.phy_scale`` (see the calibration note in
    :mod:`vecoffload.config`).
    """
    d = max(float(dist_m), 1.0)
    pl_db = 32.4 + 10.0 * cfg.pathloss_exp * np.log10(d)
    snr_db = cfg.ptx_dbm - pl_db - cfg.noise_dbm
    snr = 10.0 ** (snr_db / 10.0)
    rate = cfg.bandwidth_hz * np.log2(1.0 + max(snr, 1e-6))
    return min(rate, cfg.phy_cap_mbps * 1e6) * cfg.phy_scale


def mac_contention(cfg: Config, n_contenders: float):
    """IEEE 802.11p EDCA CSMA/CA abstraction.

    Parameters
    ----------
    n_contenders:
        Number of backlogged stations contending in the carrier-sense
        neighbourhood of the requester-executor pair.

    Returns
    -------
    (collision_probability, mean_access_delay_seconds, airtime_share)

    Notes
    -----
    Reproduces aggregate EDCA behaviour (collision probability, mean access
    delay, airtime sharing).  It does not model per-frame capture effects,
    hidden terminals, or the full four-access-category queueing discipline;
    a packet-level ns-3 study would be required for those.
    """
    n = max(float(n_contenders), 1.0)
    if not cfg.model_contention:
        return 0.0, 0.0, 1.0

    # per-slot transmission attempt probability
    varsigma = 2.0 / (cfg.cw_min + 1.0)
    p_coll = 1.0 - (1.0 - varsigma) ** (n - 1.0)

    # expected backoff slots under binary exponential backoff
    exp_slots, cw, p_stage = 0.0, cfg.cw_min, 1.0
    for _ in range(cfg.max_retries + 1):
        exp_slots += p_stage * (cw / 2.0)
        p_stage *= p_coll
        cw = min(cw * 2, cfg.cw_max)

    access_delay = (cfg.difs_us + exp_slots * cfg.slot_us) * 1e-6
    access_delay *= (1.0 + 1.6 * (n - 1.0) / 10.0)

    # airtime is shared among simultaneously backlogged contenders
    share = 1.0 / (1.0 + 0.30 * (n - 1.0))
    return p_coll, access_delay, share
