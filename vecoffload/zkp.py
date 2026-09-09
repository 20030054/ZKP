"""
Zero-knowledge proof cost model (Groth16 over the BN254 curve).

The circuit proves a fixed-size predicate over a *digest* of the payload, so
proving and verification are both constant in task size.  The payload itself is
committed natively (outside the circuit) with a conventional collision-resistant
hash; that commitment is the only term that grows with the task size, and it
grows linearly with a small constant.

Hashing 10 MB *inside* the circuit would require on the order of 10^6-10^7
constraints and put proving time in the seconds range, which is incompatible
with a 50-300 ms deadline.  Hence the split below.
"""

from __future__ import annotations

from typing import Tuple

from .config import Config


def zkp_costs_ms(cfg: Config, b_bits: float, verify: bool) -> Tuple[float, float, float, float]:
    """Return ``(generation, verification, commitment, total)`` in milliseconds.

    ``generation`` and ``verification`` are constant in ``b_bits``;
    ``commitment`` is linear in it.
    """
    if not verify:
        return 0.0, 0.0, 0.0, 0.0
    commit = (b_bits / 8.0) / (cfg.commit_gbps * 1e9) * 1e3
    total = cfg.zkp_gen_ms + cfg.zkp_ver_ms + commit
    return cfg.zkp_gen_ms, cfg.zkp_ver_ms, commit, total


def zkp_profile(cfg: Config, task_mb_values):
    """Tabulate the verification profile across a range of task sizes."""
    from .config import MB
    rows = []
    for mb in task_mb_values:
        gen, ver, com, tot = zkp_costs_ms(cfg, mb * MB, True)
        rows.append(dict(task_mb=mb, gen_ms=gen, verify_ms=ver, commit_ms=com,
                         total_ms=tot, proof_bytes=cfg.zkp_proof_bytes))
    return rows
