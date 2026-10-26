"""E5: comparison. The non-causal ablation (G2) and the classical benchmarks, stationary
bootstrap (B1) and GJR-GARCH-t (B2), trained on the same single histories as G1, are
validated with levels L0-L3 under the same contexts and oracle scenarios."""

from __future__ import annotations

from tavp.experiments import common, runner


def settings(cfg: dict) -> list[tuple[str, dict]]:
    G = cfg["generators"]
    out = []
    for d in cfg["dgps"]:
        out += [(d, {"kind": "vae_mlp", "regime": "single", "seed": s}) for s in range(G["vae_mlp"]["seeds"])]
        out += [(d, {"kind": "bootstrap", "seed": s}) for s in range(G["bootstrap"]["seeds"])]
        out += [(d, {"kind": "garch", "seed": s}) for s in range(G["garch"]["seeds"])]
    return out


def main():
    args = common.parse_args(__doc__)
    cfg = common.load(args.config)
    runner.run(cfg, "e5", settings(cfg), common.n_jobs(cfg, args.jobs))


if __name__ == "__main__":
    main()
