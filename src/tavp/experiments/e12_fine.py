"""E12 (added after the results, not pre-registered; deviation D9): fine grid of severities.

E2 detects most planted defects at its smallest severity, delta = 0.25, so it does not show
where detection begins. E12 runs the same defective oracles at delta = 0.05, 0.10, 0.15 and 0.20,
with the contexts, sizes and random streams of E2 (generated draws are shared across severities),
so that, together with E2 and the negative control E1, the power curves of every level are known
on a grid of step 0.05 between 0 and 0.25. The pre-registered results of E2 are not changed.
"""

from __future__ import annotations

from tavp.experiments import common, runner
from tavp.perturb.operators import is_applicable

FINE_SEVERITIES = (0.05, 0.10, 0.15, 0.20)


def settings(cfg: dict) -> list[tuple[str, dict]]:
    dgps = common.dgps(cfg)
    return [(key, {"kind": "oracle", "defect": defect, "delta": float(delta)})
            for defect in cfg["meta_validation"]["defects"] for key, dgp in dgps.items()
            if is_applicable(defect, dgp.name) for delta in FINE_SEVERITIES]


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    mv = cfg["meta_validation"]
    cfg["protocol"]["L2"]["K"], cfg["protocol"]["L2"]["N"] = mv["K"], mv["N"]  # as in E2
    runner.run(cfg, "e12", settings(cfg), common.n_jobs(cfg, args.jobs))


if __name__ == "__main__":
    main()
