"""Preparation of the variables: fitted on the training part, applied to the other parts.

- numerical variables: median imputation, then a quantile transformation towards the normal law
  (monotone, so that trees are unaffected, and robust to the heavy-tailed amounts, which matters for the
  support vector machine and the perceptron);
- ordered categorical variables: their order (``OrdinalSpec``), unknown categories coded -1;
- nominal variables with at most two values: a single 0/1 column;
- other nominal variables: one indicator per level, with the levels below 1% of the training cases
  (and at least 5 occurrences) grouped into a common "infrequent" column, where unseen levels also go.

All transformers are fitted on the training part only, and the order of the columns is deterministic.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer

from fraud.specs import OrdinalSpec


class Preprocessor:
    """Build and fit a ColumnTransformer on the training part, and apply it to the other parts.

    Parameters
    ----------
    ordinal_specs : sequence of :class:`OrdinalSpec`
        Order of the ordered categorical variables.
    verbose : bool
        Print a one-line summary of the encoded variables after fitting.
    """

    def __init__(self, ordinal_specs: List[OrdinalSpec] | Tuple[OrdinalSpec, ...], *, verbose: bool = False) -> None:
        self._ordinal_specs = list(ordinal_specs)
        self._verbose = bool(verbose)
        self._ct: ColumnTransformer | None = None
        self._feature_info: Dict | None = None

    # ------------------------------------------------------------------ public API

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on ``X_train`` and return the transformed float32 array."""
        self._ct, self._feature_info = self._build_and_fit(X_train)
        if self._verbose:
            self._log_info()
        return self._ct.transform(X_train).astype(np.float32)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted transformer."""
        if self._ct is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit_transform(X_train) first.")
        return self._ct.transform(X).astype(np.float32)

    @property
    def is_fitted(self) -> bool:
        return self._ct is not None

    @property
    def feature_info(self) -> Dict:
        """Columns of each group: ``num_cols``, ``ordinal_cols``, ``binary_nom``, ``multi_nom``."""
        if self._feature_info is None:
            raise RuntimeError("Preprocessor is not fitted.")
        return self._feature_info

    @property
    def n_features_out(self) -> int:
        """Total number of encoded columns."""
        if self._ct is None:
            raise RuntimeError("Preprocessor is not fitted.")
        fi = self._feature_info
        assert fi is not None
        n_ohe = (sum(len(self._ct.named_transformers_["nom"].categories_[i]) - 1 for i in range(len(fi["multi_nom"])))
                 if fi["multi_nom"] else 0)
        return len(fi["num_cols"]) + len(fi["ordinal_cols"]) + len(fi["binary_nom"]) + n_ohe

    # ------------------------------------------------------------------ internal

    def _build_and_fit(self, X_train: pd.DataFrame) -> Tuple[ColumnTransformer, Dict]:
        ordinal_pairs = [(s.col, list(s.categories)) for s in self._ordinal_specs if s.col in X_train.columns]
        ordinal_cols = [c for c, _ in ordinal_pairs]
        ordinal_cats = [cats for _, cats in ordinal_pairs]

        all_cat_cols = list(dict.fromkeys(X_train.select_dtypes("object").columns))
        nominal_cols = [c for c in all_cat_cols if c not in ordinal_cols]
        binary_nom: List[str] = []
        multi_nom: List[str] = []
        for col in nominal_cols:
            (binary_nom if X_train[col].nunique() <= 2 else multi_nom).append(col)
        num_cols = list(dict.fromkeys(X_train.select_dtypes(exclude="object").columns))

        transformers = []
        if num_cols:
            n_q = int(min(1000, max(10, len(X_train) // 10)))
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", QuantileTransformer(n_quantiles=n_q, output_distribution="normal",
                                           subsample=200_000, random_state=0)),
            ]), num_cols))
        if ordinal_cols:
            transformers.append(("ord", OrdinalEncoder(categories=ordinal_cats, handle_unknown="use_encoded_value",
                                                       unknown_value=-1), ordinal_cols))
        if binary_nom:
            transformers.append(("bin_nom", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                                 binary_nom))
        if multi_nom:
            # no drop="first": with handle_unknown an unseen level would be indistinguishable from the dropped
            # one; rare levels (below 1% of the training part, at least 5 occurrences) share one column
            transformers.append(("nom", OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist",
                                                      min_frequency=max(5, int(0.01 * len(X_train)))), multi_nom))

        ct = ColumnTransformer(transformers=transformers, remainder="drop")
        ct.fit(X_train)
        feature_info = {"num_cols": num_cols, "ordinal_cols": ordinal_cols, "binary_nom": binary_nom,
                        "multi_nom": multi_nom}
        return ct, feature_info

    def _log_info(self) -> None:
        fi = self._feature_info
        assert fi is not None and self._ct is not None
        print(f"[PREPROCESSOR] num={len(fi['num_cols'])} | ordinal={len(fi['ordinal_cols'])} | "
              f"binary_nom={len(fi['binary_nom'])} | multi_nom={len(fi['multi_nom'])} | "
              f"total_processed_features={self.n_features_out}")
