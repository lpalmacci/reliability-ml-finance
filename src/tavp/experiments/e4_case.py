"""E4: case study. The Time-Causal VAE (G1), trained on each process in the single-history and
rich regimes with several seeds, is validated with levels L0-L3."""

from __future__ import annotations

from tavp.experiments import common, runner


def settings(cfg: dict) -> list[tuple[str, dict]]:
    g = cfg["generators"]["tcvae"]
    return [(d, {"kind": "tcvae", "regime": r, "seed": s})
            for d in cfg["dgps"] for r in g["regimes"] for s in range(g["seeds"])]


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    runner.run(cfg, "e4", settings(cfg), common.n_jobs(cfg, args.jobs))


if __name__ == "__main__":
    main()
