"""E2: planted defects P1-P6 of graded severity on every applicable process; the protocol is
run on each defective oracle to measure the power of every level and statistic."""

from __future__ import annotations

from tavp.experiments import common, runner
from tavp.perturb.operators import is_applicable


def settings(cfg: dict) -> list[tuple[str, dict]]:
    dgps = common.dgps(cfg)
    out = []
    for defect in cfg["meta_validation"]["defects"]:
        for key, dgp in dgps.items():
            if not is_applicable(defect, dgp.name):
                continue
            for delta in cfg["meta_validation"]["severities"]:
                if delta > 0:
                    out.append((key, {"kind": "oracle", "defect": defect, "delta": float(delta)}))
    return out


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    mv = cfg["meta_validation"]
    # E2 uses the meta-validation sizes (pre-registered equal to those of L2)
    cfg["protocol"]["L2"]["K"], cfg["protocol"]["L2"]["N"] = mv["K"], mv["N"]
    runner.run(cfg, "e2", settings(cfg), common.n_jobs(cfg, args.jobs))


if __name__ == "__main__":
    main()
