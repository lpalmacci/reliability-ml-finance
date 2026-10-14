"""E1: negative control. Null law of the L1 discrepancies, then the oracle evaluated as a
generator of itself on every process: false-alarm rates of L1, L2 and L3 and the
resolvability of the declared tolerances (L0.4)."""

from __future__ import annotations

from tavp.experiments import common, l1_null, runner


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    jobs = common.n_jobs(cfg, args.jobs)
    l1_null.ensure(cfg, jobs)
    settings = [(d, {"kind": "oracle"}) for d in cfg["dgps"]]
    runner.run(cfg, "e1", settings, jobs)


if __name__ == "__main__":
    main()
