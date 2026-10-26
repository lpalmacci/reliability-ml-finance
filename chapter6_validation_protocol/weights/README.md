# Trained Quant GANs of the thesis

Training the Quant GAN on the Apple GPU (MPS) is not reproducible bit for bit, even on the same machine with
the same seeds (see `REPRODUCE.md`, *Determinism*). The numbers of the chapter that involve the Quant GAN can
therefore be regenerated exactly only from the models trained for the thesis, which this folder holds.

`make weights` (or `uv run python scripts/install_weights.py`) checks the SHA-256 of every file against
`MANIFEST.json` and copies the models into `results/`. The training scripts skip a model whose `meta.json`
exists, so `make qgan` then uses these models without retraining. `make all` installs them first.

| Folder (same path under `results/`) | Models | Experiment |
|---|---|---|
| `q0/runs/seed0`–`seed2` | 3 | fidelity check Q0, starting settings |
| `q0/tune/<configuration>/<seed>` | 7 | tuning round Q0b: one run for each of four configurations, three runs of the chosen one |
| `runs/qgan/<process>/<regime>/seed0`–`seed4` | 40 | E13: four processes, ten years and rich data, five runs |
| `real/runs/qgan/<process>/<regime>/seed0`–`seed4` | 20 | E13 on the processes calibrated on the market (E10) |
| `real/runs_market/qgan/w<window>/seed0`–`seed2` | 27 | E11: three runs on each of the nine windows of the market |

Each model has the generator at the selected checkpoint (`generator.pt`), its metadata (configuration, seed,
selected step, preprocessing of the training returns) in `meta.json`, and the metrics of every checkpoint in
`training_log.json`. The intermediate checkpoints are not included.

With these models, on the platform of the thesis (Apple M2 Pro, macOS, the versions of `uv.lock`), the full study
followed by `make verify` reproduces all 8,473 numbers of the chapter, and its tables and validation cards, byte for
byte. To retrain the Quant GAN instead, run the steps of `make all` without `make weights`: the generators are
still found unfit on every process, but some numbers of the chapter change, and so does the outcome of one
hypothesis (`REPRODUCE.md`, *Determinism*).
