# Reproduce the results of the chapter (see REPRODUCE.md). `make quick` is a smoke run;
# `make all` is the full study (about 17 hours on a 12-core laptop CPU).
UV ?= uv
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY = $(UV) run python -m
CFG ?= configs/full.yaml
XCFG ?= configs/exploratory.yaml
J ?=
JOBS = $(if $(J),--jobs $(J),)

.PHONY: test quick all train meta case exploratory real thesis verify verify-frozen freeze clean

test:
	$(UV) run pytest -q

train:
	$(PY) tavp.experiments.train --config $(CFG) $(JOBS)

meta:
	$(PY) tavp.experiments.e1_null --config $(CFG) $(JOBS)
	$(PY) tavp.experiments.e2_defects --config $(CFG) $(JOBS)
	$(PY) tavp.experiments.e12_fine --config $(CFG) $(JOBS)
	$(PY) tavp.experiments.e6_calibration --config $(CFG) --part meta $(JOBS)
	$(PY) tavp.experiments.l4_size --config $(CFG) $(JOBS)

case:
	$(PY) tavp.experiments.e4_case --config $(CFG) $(JOBS)
	$(PY) tavp.experiments.e4_facts --config $(CFG) $(JOBS)
	$(PY) tavp.experiments.e5_compare --config $(CFG) $(JOBS)
	$(PY) tavp.experiments.e6_calibration --config $(CFG) --part case $(JOBS)
	$(PY) tavp.experiments.e7_robust --config $(CFG) $(JOBS)
	$(PY) tavp.experiments.e8_attain --config $(CFG) $(JOBS)
	$(PY) tavp.experiments.e9_mr --config $(CFG) $(JOBS)

exploratory:
	$(PY) tavp.experiments.train --config $(XCFG) $(JOBS)
	$(PY) tavp.experiments.x1_wide --config $(XCFG) $(JOBS)
	$(PY) tavp.experiments.e4_facts --config $(XCFG) $(JOBS)

# Market-data analysis (D8, not pre-registered): data download, pseudo-real oracles (E10), L4 on the market (E11)
RCFG = configs/real.yaml
real:
	$(UV) run python scripts/get_french.py
	$(PY) tavp.experiments.e10_calibrate --config $(CFG)
	$(PY) tavp.experiments.train --config $(RCFG) $(JOBS)
	$(PY) tavp.experiments.e1_null --config $(RCFG) $(JOBS)
	$(PY) tavp.experiments.e4_case --config $(RCFG) $(JOBS)
	$(PY) tavp.experiments.e5_compare --config $(RCFG) $(JOBS)
	$(PY) tavp.experiments.e8_attain --config $(RCFG) $(JOBS)
	$(PY) tavp.experiments.e10_multistart --config $(RCFG) --protocol $(JOBS)
	$(PY) tavp.experiments.e10_multistart --config $(CFG) $(JOBS)
	$(PY) tavp.experiments.e11_market_l4 --config $(RCFG) $(JOBS)

thesis:
	$(PY) tavp.report.build --config $(CFG) $(JOBS)

all: test train meta case exploratory real thesis

quick:
	$(MAKE) test train meta case thesis CFG=configs/quick.yaml J=$(J)

verify:
	$(UV) run python scripts/verify.py

verify-frozen:
	$(UV) run python scripts/verify.py --frozen

freeze:
	$(UV) run python scripts/freeze.py

clean:
	rm -rf results/runs results/e* results/x1_wide results/quick thesis_assets/tables/* thesis_assets/figures/* thesis_assets/cards/*
