"""Wrapper around the vendored Time-Causal VAE code (third_party/tcvae, unmodified).

The same class serves G1 (TC-VAE: causal LSTM encoder and decoder) and G2
(non-causal ablation: MLP encoder and decoder); only the architecture section
of the configuration changes. Price windows are normalised to start at 1 and
the model is conditioned on that initial value, as in the original code, so the
generator is unconditional with respect to the past (Mode S).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tavp.generators.base import Generator
from tavp.utils import REPO_ROOT

_TCVAE_PATH = str(REPO_ROOT / "third_party" / "tcvae")
if _TCVAE_PATH not in sys.path:
    sys.path.insert(0, _TCVAE_PATH)


def _exp_config(model_cfg: dict, n_sample: int):
    import ml_collections

    cfg = dict(model_cfg)
    cfg.update({"n_sample": n_sample, "wandb": False, "base_output_dir": "", "output_dir": ""})
    return ml_collections.ConfigDict(cfg)


def windows_from_history(returns: np.ndarray, length: int) -> np.ndarray:
    """All overlapping price windows of `length` prices (stride 1), normalised to start at 1."""
    logp = np.concatenate([[0.0], np.cumsum(returns)])
    idx = np.arange(len(logp) - length + 1)[:, None] + np.arange(length)[None, :]
    w = logp[idx]
    return np.exp(w - w[:, :1])


def returns_to_prices(returns: np.ndarray) -> np.ndarray:
    """Independent paths of returns (n, T) to price windows (n, T + 1) starting at 1."""
    return np.exp(np.concatenate([np.zeros((len(returns), 1)), np.cumsum(returns, axis=1)], axis=1))


class TCVAEGenerator(Generator):
    """Time-Causal VAE (or its non-causal ablation) trained on price windows."""

    def __init__(self, model_cfg: dict, target_hash: str, name: str = "tcvae", power_k: float = 1.0):
        self.model_cfg = dict(model_cfg)
        self.target_hash = target_hash
        self.name = name
        self.power_k = float(power_k)
        self.length = int(model_cfg["data_length"])
        self.model = None
        self.meta: dict = {}

    # ---------------------------------------------------------------- training
    def fit(self, train_windows: np.ndarray, seed: int, out_dir: str | Path, epochs: int | None = None,
            n_monitor: int = 1000) -> "TCVAEGenerator":
        """Train with the original trainer and keep the model of the last epoch.

        The original trainer always retains the last epoch: its evaluation step
        only logs an MMD on 1000 samples. These monitoring samples are drawn
        from the training windows, so no data are held out.
        """
        from tsvae.dataset.base import BaseDataset
        from tsvae.models.network_pipeline import NetworkPipeline
        from tsvae.trainers.base_trainer_config import BaseTrainerConfig
        from tsvae.trainers.training_pipeline import TrainingPipeline
        from tsvae.utils.random_utils import set_seed

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(self.model_cfg)
        if epochs is not None:
            cfg["epochs"] = int(epochs)
        exp_config = _exp_config(cfg, len(train_windows))

        def to_dataset(w):
            x = torch.tensor(w**self.power_k, dtype=torch.float32)[..., None]
            return BaseDataset(x, torch.ones((len(w), 1)))

        if len(train_windows) < n_monitor:
            raise ValueError(f"the original trainer needs at least {n_monitor} windows")
        monitor = train_windows[np.random.default_rng(seed).permutation(len(train_windows))[:n_monitor]]
        set_seed(seed)
        model = NetworkPipeline()(exp_config)
        tcfg = BaseTrainerConfig(
            output_dir=str(out_dir / "trainer"), learning_rate=cfg["lr"],
            per_device_train_batch_size=cfg["train_batch_size"], per_device_eval_batch_size=cfg["eval_batch_size"],
            optimizer_cls=cfg["optimizer"], num_epochs=cfg["epochs"], seed=seed, no_cuda=True,
            steps_saving=None, steps_predict=None, wandb_callback=False,
        )
        t0 = time.time()
        trainer = TrainingPipeline(model=model, training_config=tcfg, exp_config=exp_config)(
            train_dataset=to_dataset(train_windows), eval_dataset=to_dataset(monitor), device_name="cpu")
        trainer.train(log_output=False)
        elapsed = time.time() - t0
        final = Path(trainer.training_dir) / "final_model"
        shutil.copytree(final, out_dir / "final_model", dirs_exist_ok=True)
        shutil.rmtree(out_dir / "trainer", ignore_errors=True)
        self.meta = {
            "name": self.name, "target_hash": self.target_hash, "seed": seed, "power_k": self.power_k,
            "epochs": cfg["epochs"], "n_train": len(train_windows),
            "train_seconds": round(elapsed, 1), "model_cfg": cfg,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        }
        (out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        self._load_model(out_dir / "final_model", cfg)
        return self

    # ---------------------------------------------------------------- loading
    def _load_model(self, model_dir: Path, cfg: dict) -> None:
        from tsvae.models.network_pipeline import NetworkPipeline

        model = NetworkPipeline()(_exp_config(cfg, 1))
        model.load_from_folder(str(model_dir))
        model.eval()
        model.device = "cpu"
        self.model = model

    @classmethod
    def load(cls, run_dir: str | Path) -> "TCVAEGenerator":
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text())
        gen = cls(meta["model_cfg"], meta["target_hash"], meta["name"], meta["power_k"])
        gen.meta = meta
        gen._load_model(run_dir / "final_model", meta["model_cfg"])
        return gen

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_returns(self, n, T, rng, batch: int = 20000):
        if T != self.length - 1:
            raise ValueError(f"the model generates {self.length - 1} returns, requested {T}")
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        out = []
        for start in range(0, n, batch):
            m = min(batch, n - start)
            x = self.model.generation(n_sample=m, c=torch.ones((m, 1)))
            x = torch.as_tensor(x).double().reshape(m, self.length)
            out.append(x.numpy())
        prices = np.concatenate(out, axis=0)
        logp = np.log(np.clip(prices, 1e-300, None)) / self.power_k
        return np.diff(logp, axis=1)

    def describe(self):
        return {**super().describe(), **{k: v for k, v in self.meta.items() if k != "model_cfg"}}
