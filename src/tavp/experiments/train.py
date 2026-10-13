"""Train the neural generators (G1 TC-VAE, G2 non-causal VAE) on every process, regime and seed.

The pre-registered scaling rule is applied first: G1 is trained on D1 in the rich regime with
power_k = 1; if the annualised volatility of its scenarios deviates from the oracle by more
than 20%, every model is trained on prices raised to power_k = sqrt(21) instead.
"""

from __future__ import annotations

import json
import math

import numpy as np
from joblib import Parallel, delayed

from tavp.experiments import common
from tavp.generators.tcvae import TCVAEGenerator, returns_to_prices, windows_from_history
from tavp.utils import REPO_ROOT, load_yaml, rng


def training_windows(cfg, dgp, regime: str, seed: int) -> np.ndarray:
    if regime == "single":
        h = common.history(cfg, dgp, seed)[: cfg["data"]["train_returns"]]
        return windows_from_history(h, cfg["data"]["scenario_prices"])
    if regime == "rich":
        return returns_to_prices(common.rich_paths(cfg, dgp, seed))
    raise ValueError(regime)


def train_one(cfg, gen_name: str, dgp_key: str, regime: str, seed: int, power_k: float, epochs: int | None):
    import torch

    torch.set_num_threads(int(cfg.get("runtime", {}).get("torch_threads", 2)))
    dgp = common.dgps(cfg)[dgp_key]
    out = common.run_dir(cfg, gen_name, dgp.name, regime, seed)
    if (out / "meta.json").exists():
        return str(out)
    model_cfg = load_yaml(REPO_ROOT / "configs" / cfg["generators"][gen_name]["config"])
    w = training_windows(cfg, dgp, regime, seed)
    TCVAEGenerator(model_cfg, dgp.hash(), name=gen_name, power_k=power_k).fit(
        w, seed=common.stream_seed(cfg, "train", gen_name, dgp.name, regime, seed), out_dir=out, epochs=epochs)
    return str(out)


def decide_scaling(cfg, epochs: int | None) -> dict:
    """Pre-registered scaling rule, decided on D1 (gbm) only."""
    path = common.results_root(cfg) / "train" / "scaling_decision.json"
    if path.exists():
        return json.loads(path.read_text())
    rule = cfg["generators"]["scaling_rule"]
    dgp_key = rule["decide_on"]
    run = train_one(cfg, "tcvae", dgp_key, rule["regime"], 0, 1.0, epochs)
    dgp = common.dgps(cfg)[dgp_key]
    gen = TCVAEGenerator.load(run)
    T = cfg["data"]["scenario_prices"] - 1
    vol_g = float(np.std(gen.sample_returns(10000, T, rng(cfg["master_seed"], "scaling", "G"))) * math.sqrt(252))
    vol_o = float(np.std(dgp.simulate_returns(10000, T, rng(cfg["master_seed"], "scaling", "O"))) * math.sqrt(252))
    rel = abs(vol_g / vol_o - 1.0)
    decision = {"vol_generator": vol_g, "vol_oracle": vol_o, "relative_error": rel,
                "threshold": rule["max_rel_vol_error"],
                "power_k": 1.0 if rel <= rule["max_rel_vol_error"] else rule["fallback_power_k"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**decision, **common.provenance(cfg)}, indent=2))
    if decision["power_k"] != 1.0:  # the k = 1 model cannot be reused
        import shutil
        shutil.rmtree(run)
    return decision


def main():
    args = common.parse_args(__doc__, lambda p: p.add_argument("--only-scaling", action="store_true"))
    cfg = common.load(args.config)
    epochs = cfg.get("runtime", {}).get("epochs")  # None = pre-registered value in the model config
    decision = decide_scaling(cfg, epochs)
    print("scaling decision:", decision)
    if args.only_scaling:
        return
    jobs = []
    for gen_name in ("tcvae", "vae_mlp"):
        g = cfg["generators"][gen_name]
        for dgp_key in cfg["dgps"]:
            for regime in g["regimes"]:
                for s in range(g["seeds"]):
                    jobs.append((gen_name, dgp_key, regime, s))
    done = Parallel(n_jobs=common.n_jobs(cfg, args.jobs), verbose=5)(
        delayed(train_one)(cfg, *j, decision["power_k"], epochs) for j in jobs)
    print(f"{len(done)} models available")


if __name__ == "__main__":
    main()
