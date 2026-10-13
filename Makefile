# Reproduce the results of the chapter. `make quick` is a smoke run (about 20 minutes);
# `make all` is the full pre-registered run (about 4-5 hours on an 8-core laptop CPU).
UV ?= uv
PY = $(UV) run python -m
CFG ?= configs/full.yaml

.PHONY: test quick all train meta case thesis verify clean

test:
	$(UV) run pytest

train:
	$(PY) tavp.experiments.train --config $(CFG)

meta:
	$(PY) tavp.experiments.e1_null --config $(CFG)
	$(PY) tavp.experiments.e2_defects --config $(CFG)
	$(PY) tavp.experiments.e6_calibration --config $(CFG) --part meta

case:
	$(PY) tavp.experiments.e4_case --config $(CFG)
	$(PY) tavp.experiments.e5_compare --config $(CFG)
	$(PY) tavp.experiments.e6_calibration --config $(CFG) --part case
	$(PY) tavp.experiments.e7_robust --config $(CFG)

thesis:
	$(PY) tavp.report.build --config $(CFG)

all: test train meta case thesis

quick:
	$(MAKE) all CFG=configs/quick.yaml

verify:
	$(UV) run python scripts/verify.py

clean:
	rm -rf results/runs results/e* thesis_assets/tables/* thesis_assets/figures/* thesis_assets/cards/*
