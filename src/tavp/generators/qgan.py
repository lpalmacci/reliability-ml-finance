"""G3: Quant GAN (Wiese, Knobloch, Korn and Kretschmer, 2020), pure TCN model, implemented from the paper.

No official code was released and the public replications carry no licence, so this module is an
independent implementation of the specification of the paper (arXiv:1907.06673):

- preprocessing (Section 6): log returns, standardisation, inverse Lambert W x Gaussian transform with
  parameters estimated by maximum likelihood, second standardisation, rolling windows of length equal to
  the receptive field of the discriminator;
- generator and discriminator (Appendix B, Tables 3-4): temporal convolutional networks with skip
  connections, seven temporal blocks of two dilated causal convolutions and two PReLUs each (kernel size
  1 for the first block and 2 for the others, dilations 1, 1, 2, 4, 8, 16, 32), 80 hidden channels and a
  final 1x1 convolution, receptive field 127; three-dimensional i.i.d. Gaussian noise for the generator;
- objective (Section 4, Algorithm 1): the GAN objective with the non-saturating generator loss and the
  R1 gradient penalty of Mescheder et al. (2018), the "GAN stability algorithm" cited in Section 7.1.

Settings that the paper does not report (optimiser, learning rates, batch size, number of steps, weight of
the penalty) and the rule that selects the checkpoint are fixed in configs/generators/qgan.yaml; the
checkpoint is chosen on the training data only, as the authors chose the best of several saved parameter
sets by their evaluation metrics (Section 7.4). The model runs on the Apple GPU (MPS) when available.
Sampling from a trained model is deterministic, but training on the GPU is not reproducible bit for bit,
even on the same machine; the trained models of the thesis are in weights/ (REPRODUCE.md, Determinism).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import optimize, special

from tavp.generators.base import Generator

RFS = 127  # receptive field size of generator and discriminator
BLOCKS = [(1, 1)] + [(2, d) for d in (1, 2, 4, 8, 16, 32)]  # (kernel size, dilation) of the temporal blocks


def device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


# ----------------------------------------------------------------------------- Lambert W x Gaussian
class LambertW:
    """Heavy-tailed Lambert W x Gaussian transform (Goerg, 2015): y = mu + s u exp(delta u^2 / 2), u ~ N(0, 1)."""

    def __init__(self, mu: float = 0.0, s: float = 1.0, delta: float = 0.0):
        self.mu, self.s, self.delta = float(mu), float(s), float(delta)

    @staticmethod
    def _w_delta(z: np.ndarray, delta: float) -> np.ndarray:
        if delta < 1e-12:
            return z
        return np.sign(z) * np.sqrt(np.real(special.lambertw(delta * z * z)) / delta)

    def fit(self, y: np.ndarray) -> "LambertW":
        """Maximum likelihood of (mu, s, delta), delta >= 0."""
        y = np.asarray(y, dtype=float)

        def nll(p):
            mu, s, delta = p[0], np.exp(p[1]), np.exp(p[2])
            z = (y - mu) / s
            w = np.real(special.lambertw(delta * z * z))
            u = np.sign(z) * np.sqrt(w / delta)
            # du/dz = u / (z (1 + W(delta z^2))) = exp(-W / 2) / (1 + W)
            log_jac = -0.5 * w - np.log1p(w)
            return -(np.sum(-0.5 * u * u + log_jac) - len(y) * np.log(s))

        start = np.array([np.median(y), np.log(np.std(y) * 0.8), np.log(0.1)])
        res = optimize.minimize(nll, start, method="Nelder-Mead",
                                options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000, "maxfev": 40000})
        self.mu, self.s, self.delta = float(res.x[0]), float(np.exp(res.x[1])), float(np.exp(res.x[2]))
        return self

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Gaussianise: u = W_delta((y - mu) / s)."""
        return self._w_delta((np.asarray(y, dtype=float) - self.mu) / self.s, self.delta)

    def forward(self, u: np.ndarray) -> np.ndarray:
        """Restore heavy tails: y = mu + s u exp(delta u^2 / 2)."""
        u = np.asarray(u, dtype=float)
        return self.mu + self.s * u * np.exp(0.5 * self.delta * u * u)

    def params(self) -> dict:
        return {"mu": self.mu, "s": self.s, "delta": self.delta}


class Preprocessor:
    """Steps 2-4 of the pipeline of the paper: standardise, inverse Lambert W, standardise."""

    def fit(self, r: np.ndarray) -> "Preprocessor":
        r = np.asarray(r, dtype=float)
        self.m1, self.s1 = float(r.mean()), float(r.std())
        x = (r - self.m1) / self.s1
        self.lw = LambertW().fit(x)
        u = self.lw.inverse(x)
        self.m2, self.s2 = float(u.mean()), float(u.std())
        return self

    def transform(self, r: np.ndarray) -> np.ndarray:
        return (self.lw.inverse((np.asarray(r, dtype=float) - self.m1) / self.s1) - self.m2) / self.s2

    def inverse_transform(self, v: np.ndarray) -> np.ndarray:
        return self.m1 + self.s1 * self.lw.forward(np.asarray(v, dtype=float) * self.s2 + self.m2)

    def state(self) -> dict:
        return {"m1": self.m1, "s1": self.s1, "m2": self.m2, "s2": self.s2, "lambertw": self.lw.params()}

    @classmethod
    def from_state(cls, st: dict) -> "Preprocessor":
        p = cls()
        p.m1, p.s1, p.m2, p.s2 = st["m1"], st["s1"], st["m2"], st["s2"]
        p.lw = LambertW(**st["lambertw"])
        return p


# ----------------------------------------------------------------------------- networks
class TemporalBlock(nn.Module):
    """Definition B.1: two dilated causal convolutions, each followed by a PReLU (valid convolutions)."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, kernel: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_in, n_hidden, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(n_hidden, n_out, kernel, dilation=dilation)
        self.act1, self.act2 = nn.PReLU(), nn.PReLU()

    def forward(self, x):
        return self.act2(self.conv2(self.act1(self.conv1(x))))


class TCN(nn.Module):
    """TCN with skip connections (Definition 3.15, Table 3): the outputs of all temporal blocks, cropped to
    the final length, are summed and mapped by a 1x1 convolution to the output channels."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.blocks = nn.ModuleList([TemporalBlock(n_in if i == 0 else n_hidden, n_hidden, n_hidden, k, d)
                                     for i, (k, d) in enumerate(BLOCKS)])
        self.out = nn.Conv1d(n_hidden, n_out, 1)

    def forward(self, x):
        n = x.shape[-1] - (RFS - 1)  # output length of the valid convolutions
        skip = 0
        for b in self.blocks:  # the skips are summed as they are produced, in block order
            x = b(x)
            skip = skip + x[..., -n:]
        return self.out(skip)


# ----------------------------------------------------------------------------- scores of the paper
def acf(x: np.ndarray, lags: int) -> np.ndarray:
    """Autocorrelation at lags 1..lags of each row of x (n, T)."""
    x = np.atleast_2d(x)
    x = x - x.mean(axis=1, keepdims=True)
    den = (x * x).sum(axis=1)
    return np.stack([(x[:, k:] * x[:, :-k]).sum(axis=1) / den for k in range(1, lags + 1)], axis=1)


def leverage(x: np.ndarray, lags: int) -> np.ndarray:
    """Corr(r_{t+tau}^2, r_t) at lags 1..lags of each row of x (n, T)."""
    x = np.atleast_2d(x)
    out = []
    for k in range(1, lags + 1):
        a, b = x[:, :-k], x[:, k:] ** 2
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        out.append((a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1)))
    return np.stack(out, axis=1)


def dy_metric(h: np.ndarray, g: np.ndarray) -> float:
    """DY metric of Dragulescu and Yakovenko (2002): bins holding about five historical observations."""
    edges = np.quantile(h, np.linspace(0, 1, max(len(h) // 5, 2) + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ph = np.histogram(h, edges)[0] / len(h)
    pg = np.histogram(g, edges)[0] / len(g)
    eps = 1.0 / len(g)
    return float(np.sum(np.abs(np.log(ph + eps) - np.log(pg + eps))))


def paper_scores(r_hist: np.ndarray, r_gen: np.ndarray, lags: int = 250, horizons=(1, 5, 20, 100)) -> dict:
    """The metrics of Table 2 of the paper: EMD and DY of t-day log returns, ACF scores and leverage score.

    r_hist: historical log returns (T,); r_gen: generated paths of log returns (M, T~)."""
    from scipy.stats import wasserstein_distance

    out = {}
    ch, cg = np.concatenate([[0.0], np.cumsum(r_hist)]), np.concatenate(
        [np.zeros((len(r_gen), 1)), np.cumsum(r_gen, axis=1)], axis=1)
    for t in horizons:
        h, g = ch[t:] - ch[:-t], (cg[:, t:] - cg[:, :-t]).ravel()
        out[f"emd_{t}"] = float(wasserstein_distance(h, g))
        out[f"dy_{t}"] = dy_metric(h, g)
    for name, f in (("acf_id", lambda z: z), ("acf_abs", np.abs), ("acf_sq", np.square)):
        out[name] = float(np.linalg.norm(acf(f(r_hist), lags)[0] - acf(f(r_gen), lags).mean(axis=0)))
    out["leverage"] = float(np.linalg.norm(leverage(r_hist, lags)[0] - leverage(r_gen, lags).mean(axis=0)))
    return out


# ----------------------------------------------------------------------------- generator
class QuantGANGenerator(Generator):
    """Pure-TCN Quant GAN trained on one history of daily log returns; generates paths of any length."""

    def __init__(self, cfg: dict, target_hash: str, name: str = "qgan"):
        self.cfg = dict(cfg)
        self.target_hash = target_hash
        self.name = name
        self.dev = device(cfg.get("device", "auto"))
        self.net = TCN(cfg["noise_dim"], cfg["hidden"], 1).to(self.dev)
        self.pre: Preprocessor | None = None
        self.meta: dict = {}

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def _raw(self, z: np.ndarray, batch: int) -> np.ndarray:
        self.net.eval()
        out = []
        for start in range(0, len(z), batch):
            x = torch.from_numpy(z[start:start + batch]).to(self.dev)
            out.append(self.net(x).squeeze(1).cpu().numpy().astype(np.float64))
        return np.concatenate(out, axis=0)

    def sample_returns(self, n, T, rng, batch: int = 5000):
        z = rng.standard_normal((n, self.cfg["noise_dim"], T + RFS - 1)).astype(np.float32)
        out = self.pre.inverse_transform(self._raw(z, batch))
        if self.dev.type == "mps":  # the GPU shares the memory of the system: release the cached blocks
            torch.mps.empty_cache()
        return out

    # ---------------------------------------------------------------- training
    def fit(self, history: np.ndarray, seed: int, out_dir: str | Path) -> "QuantGANGenerator":
        c = self.cfg
        out_dir = Path(out_dir)
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        history = np.asarray(history, dtype=float)
        self.pre = Preprocessor().fit(history)
        x = self.pre.transform(history).astype(np.float32)
        idx = np.arange(len(x) - RFS + 1)[:, None] + np.arange(RFS)[None, :]
        windows = torch.from_numpy(x[idx][:, None, :])  # (n, 1, RFS), kept on the CPU
        gcpu = torch.Generator().manual_seed(int(seed))  # all random draws of training on the CPU
        torch.manual_seed(int(seed))
        self.net = TCN(c["noise_dim"], c["hidden"], 1).to(self.dev)
        disc = TCN(1, c["hidden"], 1).to(self.dev)
        opt_g = torch.optim.Adam(self.net.parameters(), lr=c["lr_g"], betas=tuple(c["betas"]))
        opt_d = torch.optim.Adam(disc.parameters(), lr=c["lr_d"], betas=tuple(c["betas"]))
        bce = nn.BCEWithLogitsLoss()
        B, gamma = c["batch"], c["r1_gamma"]
        ones, zeros = torch.ones(B, device=self.dev), torch.zeros(B, device=self.dev)
        z_eval = np.random.default_rng(seed).standard_normal(
            (c["select_paths"], c["noise_dim"], c["select_length"] + RFS - 1)).astype(np.float32)
        log, t0 = [], time.time()
        for step in range(1, c["steps"] + 1):
            self.net.train(); disc.train()
            real = windows[torch.randint(len(windows), (B,), generator=gcpu)].to(self.dev).requires_grad_(True)
            z = torch.randn(B, c["noise_dim"], 2 * RFS - 1, generator=gcpu).to(self.dev)
            fake = self.net(z)
            d_real, d_fake = disc(real).view(-1), disc(fake.detach()).view(-1)
            grad = torch.autograd.grad(d_real.sum(), real, create_graph=True)[0]
            loss_d = bce(d_real, ones) + bce(d_fake, zeros) + 0.5 * gamma * grad.pow(2).sum(dim=(1, 2)).mean()
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            loss_g = bce(disc(fake).view(-1), ones)  # non-saturating generator loss
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
            if step % c["checkpoint_every"] == 0:
                gen = self.pre.inverse_transform(self._raw(z_eval, 1000))
                sc = paper_scores(history, gen, lags=c["select_lags"], horizons=tuple(c["select_horizons"]))
                torch.save(self.net.state_dict(), out_dir / "checkpoints" / f"step{step:06d}.pt")
                log.append({"step": step, "loss_d": float(loss_d), "loss_g": float(loss_g), **sc,
                            "seconds": time.time() - t0})
        sel = select_checkpoint(log, c["select_metrics"])
        self.net.load_state_dict(torch.load(out_dir / "checkpoints" / f"step{sel:06d}.pt", map_location=self.dev, weights_only=True))
        torch.save(self.net.state_dict(), out_dir / "generator.pt")
        self.meta = {"name": self.name, "target_hash": self.target_hash, "cfg": c, "seed": int(seed),
                     "selected_step": int(sel), "preprocessing": self.pre.state(), "train_returns": len(history),
                     "device": str(self.dev), "train_seconds": time.time() - t0}
        (out_dir / "training_log.json").write_text(json.dumps(log, indent=1))
        (out_dir / "meta.json").write_text(json.dumps(self.meta, indent=1))
        return self

    @classmethod
    def load(cls, run_dir: str | Path, dev: str = "auto") -> "QuantGANGenerator":
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text())
        g = cls({**meta["cfg"], "device": dev}, meta["target_hash"], meta["name"])
        g.net.load_state_dict(torch.load(run_dir / "generator.pt", map_location=g.dev, weights_only=True))
        g.pre = Preprocessor.from_state(meta["preprocessing"])
        g.meta = meta
        return g

    def describe(self):
        return {**super().describe(), **{k: v for k, v in self.meta.items() if k != "cfg"}}


def select_checkpoint(log: list[dict], metrics: list[str]) -> int:
    """Checkpoint with the smallest mean rank over the selection metrics (computed on the training data)."""
    steps = np.array([r["step"] for r in log])
    ranks = np.mean([np.argsort(np.argsort([r[m] for r in log])) for m in metrics], axis=0)
    return int(steps[np.argmin(ranks)])
