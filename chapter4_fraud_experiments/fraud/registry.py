"""The two datasets of Chapter 4, with the fingerprint and the counts that the loader must find.

The files are in data/ (see data/README.md for their sources and licences).
"""
from __future__ import annotations

from pathlib import Path

from fraud.ordinal_maps import CARCLAIMS_ORDINALS
from fraud.specs import DatasetSpec

# <chapter4_fraud_experiments>/fraud/registry.py -> <chapter4_fraud_experiments>
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"

DATASETS: dict[str, DatasetSpec] = {
    "carclaims": DatasetSpec(
        # Vehicle insurance claims, 1994-1996 (Phua, Alahakoon and Lee, 2004), Kaggle shivamb/vehicle-claim-fraud-detection.
        name="carclaims",
        path=DATA_DIR / "fraud_oracle.csv",
        target="FraudFound_P",
        positive_label=1,
        drop_cols=("PolicyNumber", "RepNumber"),        # identifiers
        ordinal_specs=CARCLAIMS_ORDINALS,
        loader_hook="carclaims",
        sha256="d7dab6086aa186fc82c607d50919067a1599cea45afb4965511fd7da9e935e09",
        n_rows=15_419,
        n_frauds=923,
        n_variables=30,
    ),
    "ulb": DatasetSpec(
        # Credit card transactions of September 2013 (Dal Pozzolo et al., 2015), Kaggle mlg-ulb/creditcardfraud.
        # Time counts the seconds from the start of the collection and is not a property of the transaction.
        name="ulb",
        path=DATA_DIR / "ulb_creditcard.csv",
        target="Class",
        positive_label=1,
        drop_cols=("Time",),
        loader_hook=None,
        sha256="76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89",
        n_rows=284_807,
        n_frauds=492,
        n_variables=29,
    ),
}


def list_datasets() -> list[str]:
    return list(DATASETS.keys())
