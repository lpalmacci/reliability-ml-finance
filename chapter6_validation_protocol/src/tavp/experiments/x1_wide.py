"""X1 (exploratory, not pre-registered; deviation D2): the non-causal ablation without the
16-unit bottleneck, validated with levels L0-L3 under the same contexts as E4 and E5."""

from __future__ import annotations

from tavp.experiments import common, runner


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    g = cfg["generators"]["vae_mlp_wide"]
    settings = [(d, {"kind": "vae_mlp_wide", "regime": "single", "seed": s}) for d in cfg["dgps"] for s in range(g["seeds"])]
    runner.run(cfg, "x1_wide", settings, common.n_jobs(cfg, args.jobs))


if __name__ == "__main__":
    main()
