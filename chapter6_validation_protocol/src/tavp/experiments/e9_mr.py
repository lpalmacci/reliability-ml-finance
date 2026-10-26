"""E9 (added after the case study, not pre-registered; deviation D6): planted defect P7, a negative
first-order autocorrelation of returns, at the pre-registered severities and at the severities
matched to the TC-VAE.

The case study attributes most failures of the TC-VAE to the negative autocorrelation of its
returns. E9 plants that property alone in the oracles, with the size observed in the TC-VAE
(median first-order autocorrelation of its scenarios per process and regime, from the stylized
facts of E4), and records which levels and cells detect it: a known-answer test of the explanation.
"""

from __future__ import annotations

import pandas as pd

from tavp.experiments import common, runner
from tavp.perturb.operators import ACF1_STEP


def matched_severities(cfg: dict) -> pd.DataFrame:
    """Severity of P7 matched to the median first-order autocorrelation of the TC-VAE, per process and regime."""
    facts = pd.read_parquet(common.results_root(cfg) / "e4" / "facts.parquet")
    x = facts[facts.kind == "tcvae"].groupby(["dgp", "regime"]).acf_r1.median().reset_index()
    x["delta"] = (x.acf_r1 / ACF1_STEP).round(2)
    return x


def settings(cfg: dict) -> list[tuple[str, dict]]:
    mv = cfg["meta_validation"]
    m = matched_severities(cfg)
    out = []
    for key in cfg["dgps"]:
        deltas = {float(d) for d in mv["severities"] if d > 0} | set(m[m.dgp == key].delta.astype(float))
        out += [(key, {"kind": "oracle", "defect": "P7", "delta": d}) for d in sorted(deltas)]
    return out


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    mv = cfg["meta_validation"]
    cfg["protocol"]["L2"]["K"], cfg["protocol"]["L2"]["N"] = mv["K"], mv["N"]  # sizes of E2
    runner.run(cfg, "e9", settings(cfg), common.n_jobs(cfg, args.jobs))
    common.save_table(matched_severities(cfg), "e9", "matched", cfg)


if __name__ == "__main__":
    main()
