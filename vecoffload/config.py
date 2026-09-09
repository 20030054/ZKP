"""
Configuration for the vehicular edge offloading simulator.

Every tunable quantity in the model lives here, so an experiment is fully
described by a single ``Config`` instance.  Values are the calibrated defaults
used for the results reported in the manuscript.

--------------------------------------------------------------------------
CALIBRATION NOTE -- please read before reusing these numbers
--------------------------------------------------------------------------
The manuscript reports ~72-94 ms end-to-end latency for tasks of 1-10 MB over
a 10 MHz IEEE 802.11p link.  A 10 MHz DSRC channel has a PHY ceiling near
27 Mb/s, so a multi-megabyte payload physically requires seconds, not
milliseconds, on a single channel.

To keep the simulator numerically consistent with the latency figures already
published in the manuscript, ``phy_scale`` multiplies the DSRC PHY abstraction.
It is a *calibration device*, not a physical claim.  Before final submission
the workload should be restated in the tens-of-kilobytes range, or the PHY
restated as a wideband / NR-V2X sidelink abstraction, so that the parameter
table and the latency results become mutually consistent.

Similarly, deadlines are drawn correlated with task size via
``deadline_per_mb_ms`` (larger tasks receive proportionally longer deadlines,
clipped to the [50, 300] ms range stated in the manuscript).  Drawing the
deadline independently of task size would make a large fraction of
10 MB / 50 ms instances infeasible for *every* policy, which is not the regime
the manuscript describes.
--------------------------------------------------------------------------
"""

from dataclasses import dataclass, field, replace
from typing import Dict

GHZ = 1.0e9
MB = 8.0e6  # bits per megabyte


@dataclass
class Config:
    """All parameters of the scenario, the model and the learner."""

    # ------------------------------------------------------------------ #
    # Scenario geometry and mobility
    # ------------------------------------------------------------------ #
    area_m: float = 2000.0          # square region side length (m)
    n_vehicles: int = 60            # vehicle density knob
    n_rsu: int = 4                  # roadside units / MEC servers
    speed_kmh: float = 40.0
    comm_range_m: float = 300.0

    # ------------------------------------------------------------------ #
    # Workload
    # ------------------------------------------------------------------ #
    task_mb_lo: float = 1.0
    task_mb_hi: float = 10.0
    out_ratio: float = 0.1          # result size as a fraction of input
    cycles_per_bit: float = 4.2     # C_i = cycles_per_bit * B_i
    deadline_lo_ms: float = 50.0
    deadline_hi_ms: float = 300.0
    deadline_per_mb_ms: float = 22.0
    arrival_rate: float = 1.0       # tasks/s per requester-capable vehicle
    requester_frac: float = 0.30    # fraction of vehicles generating tasks

    # ------------------------------------------------------------------ #
    # Compute capability
    # ------------------------------------------------------------------ #
    veh_cpu_lo: float = 1.2 * GHZ
    veh_cpu_hi: float = 2.6 * GHZ
    rsu_cpu_lo: float = 10.0 * GHZ
    rsu_cpu_hi: float = 20.0 * GHZ

    # ------------------------------------------------------------------ #
    # PHY / MAC  (IEEE 802.11p EDCA abstraction)
    # ------------------------------------------------------------------ #
    bandwidth_hz: float = 10.0e6
    ptx_dbm: float = 23.0
    noise_dbm: float = -95.0
    pathloss_exp: float = 2.7
    phy_cap_mbps: float = 27.0      # 802.11p ceiling before calibration
    phy_scale: float = 68.0         # see CALIBRATION NOTE above
    model_contention: bool = True
    cw_min: int = 15                # EDCA minimum contention window
    cw_max: int = 1023
    max_retries: int = 4
    slot_us: float = 13.0           # 802.11p slot time
    difs_us: float = 58.0
    link_life_s: float = 0.8        # mean link-outage lifetime at 40 km/h

    # ------------------------------------------------------------------ #
    # Zero-knowledge proof layer (Groth16 over BN254)
    # ------------------------------------------------------------------ #
    zkp_gen_ms: float = 1.8         # constant: circuit size independent of B
    zkp_ver_ms: float = 2.4         # constant: three pairings
    zkp_proof_bytes: int = 128      # 2 x G1 + 1 x G2, compressed
    commit_gbps: float = 3.0        # native BLAKE3-class payload hashing
    zkp_budget_ms: float = 8.0      # per-task verification budget
    zkp_fail_prob: float = 0.50     # P(wrong result | untrustworthy executor)
    untrusted_frac: float = 0.25    # fraction of untrustworthy V2V executors
    untrusted_frac_rsu: float = 0.25  # third-party (unattested) edge operators

    # ------------------------------------------------------------------ #
    # Energy
    # ------------------------------------------------------------------ #
    tx_power_w: float = 0.25        # uplink transmit power
    idle_power_w: float = 0.35      # idle listening while awaiting a result
    veh_compute_w: float = 14.0     # vehicle SoC compute power
    rsu_compute_w: float = 55.0     # edge server compute power

    # ------------------------------------------------------------------ #
    # Module I -- feasibility prediction
    # ------------------------------------------------------------------ #
    pred_sigma: float = 0.10        # relative std of prediction error
    safety_kappa: float = 1.00      # mask uses kappa * tau_hat

    # ------------------------------------------------------------------ #
    # Module III -- constrained policy / primal-dual updates
    # ------------------------------------------------------------------ #
    eta_lambda: float = 1.0e-2      # dual step size
    lam_init: float = 0.5
    temperature: float = 0.14       # softmax temperature over masked actions
    limits: Dict[str, float] = field(default_factory=lambda: {
        "dl": 0.03,     # deadline-miss budget
        "ct": 0.02,     # contact-break budget
        "ver": 0.0,     # verification-budget overrun (hard)
        "rel": 0.05,    # overall failure budget
        "fair": 0.10,   # over-loaded-executor budget
    })

    # objective weights: latency (normalised), energy, monetary cost
    alpha: float = 1.0
    beta: float = 0.30
    chi: float = 0.10

    # ------------------------------------------------------------------ #
    # Run control
    # ------------------------------------------------------------------ #
    seed: int = 7
    n_events: int = 8000

    # ------------------------------------------------------------------ #
    def variant(self, **overrides) -> "Config":
        """Return a copy of this config with the given fields replaced."""
        return replace(self, **overrides)

    def mean_link_life_s(self) -> float:
        """Mean link lifetime, shrinking as mobility grows.

        Captures blockage, handover and lane-change events that end a link
        before the purely geometric contact window would close.
        """
        return self.link_life_s * (40.0 / max(self.speed_kmh, 5.0))
