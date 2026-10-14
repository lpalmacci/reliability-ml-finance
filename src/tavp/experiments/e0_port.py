"""E0 (internal, not reported in the thesis): check that the vectorised port reproduces the
buy-and-hold oracle figures of the earlier monthly GBM setting (mu = 0.10, sigma = 0.20,
60 monthly prices, N = 10,000), whose published 95% intervals were Sharpe [0.407, 0.425]
and MaxDD [0.287, 0.291]. Function-level equivalence of TF and VT is covered by the tests."""

from __future__ import annotations

import json

import numpy as np

from tavp.dgp.processes import GBM
from tavp.tasks.backtest import evaluate
from tavp.utils import REPO_ROOT, rng

REFERENCE = {"sharpe": (0.407, 0.425), "maxdd": (0.287, 0.291)}


def main(n: int = 10_000) -> dict:
    d = GBM(dt=1 / 12, mu=0.10, sigma=0.20)
    r = d.simulate_returns(n, 59, rng(20260101, "e0"))
    logp = np.concatenate([np.zeros((n, 2)), np.cumsum(r, axis=1)], axis=1)  # two flat warm-up points
    m = evaluate(logp, np.ones_like(logp), 59, periods=12)
    out = {}
    for k, (lo, hi) in REFERENCE.items():
        mean, se = float(m[k].mean()), float(m[k].std(ddof=1) / np.sqrt(n))
        ref_mid, ref_se = (lo + hi) / 2, (hi - lo) / (2 * 1.96)
        z = (mean - ref_mid) / np.hypot(se, ref_se)
        out[k] = {"port_mean": mean, "port_se": se, "reference_mid": ref_mid, "z": float(z), "pass": bool(abs(z) < 3)}
    path = REPO_ROOT / "docs" / "E0_port_check.json"
    path.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
