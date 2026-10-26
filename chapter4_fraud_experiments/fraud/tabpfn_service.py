"""TabPFN through the remote service of its developers (``tabpfn-client``).

- the model is requested explicitly, ``model_path="v3_default"`` (the base model of TabPFN 3), never
  "auto", with which the server would choose its own current default;
- ``random_state`` comes from the key of the member;
- the other parameters keep the defaults of the client;
- ``usage()`` reads the daily and monthly use of the service, so that the tokens of each call can be
  recorded as the difference before and after.

The token is read from the environment variable ``TABPFN_TOKEN`` or from a file ``.env`` in the root of
this folder (one line ``TABPFN_TOKEN=...``); it is never written in the code or in the logs. The service
may change, so that querying it again may not reproduce the saved scores: the replication starts from
the scores saved in results/tabpfn/.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from fraud.registry import ROOT

MODEL_PATH = "v3_default"
MODEL_VERSION = "v3"
ENV_FILE = ROOT / ".env"


def load_tabpfn_token() -> bool:
    """Put ``TABPFN_TOKEN`` from ``.env`` into the environment, if it is not there already."""
    os.environ.setdefault("TABPFN_CLIENT_CI_MODE", "1")
    if os.environ.get("TABPFN_TOKEN"):
        return True
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith("TABPFN_TOKEN=") and not line.startswith("#"):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    os.environ["TABPFN_TOKEN"] = value
                    return True
    return False


def _client():
    if not load_tabpfn_token():
        raise RuntimeError("TABPFN_TOKEN is missing (environment variable or .env in the chapter folder)")
    import tabpfn_client
    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    return tabpfn_client


def client_version() -> str:
    return _client().__version__


def build_tabpfn_classifier(*, seed: int) -> Any:
    """Classifier with ``model_path`` fixed and ``random_state`` from the key of the member."""
    tc = _client()
    return tc.TabPFNClassifier(model_path=MODEL_PATH, random_state=int(seed))


def usage() -> dict[str, Any]:
    """Use of the service: tokens of the day and of the month, limits, reset times."""
    _client()
    from tabpfn_client.client import ServiceClient
    from tabpfn_client.config import get_access_token
    u = ServiceClient.get_api_usage(get_access_token())
    return {k: u.get(k) for k in ("daily_tokens_used", "daily_token_limit", "monthly_tokens_used",
                                  "monthly_token_limit", "daily_reset_time", "monthly_reset_time")}


def estimate_cost(n_train: int, n_test: int, n_col: int) -> dict[str, Any]:
    """``estimate_cost`` of the client: it uses only the dimensions, uploads no data and uses no quota."""
    tc = _client()
    r = tc.estimate_cost(np.zeros((n_train, n_col)), np.zeros((n_test, n_col)), model_version=MODEL_VERSION)
    return {"estimated": int(r.estimated_cost), "pricing_version": r.pricing_version,
            "input": r.inputs.model_dump(mode="json")}


def score(model: Any, X_raw: pd.DataFrame) -> np.ndarray:
    """Probability of the positive class (the classes are np.unique(y) = [0, 1])."""
    p = np.asarray(model.predict_proba(X_raw))
    if list(model.classes_) != [0, 1]:
        raise RuntimeError(f"unexpected classes {model.classes_}")
    return p[:, 1]
