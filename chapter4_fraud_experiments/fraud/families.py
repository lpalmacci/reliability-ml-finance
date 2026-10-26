"""The families of the members.

Every family exposes

- ``build(model_seed, auxiliary_seed, hyperparameters)``,
- ``fit(model, X_proc, X_raw, y, cat_cols, cat_idx)``,
- ``score(model, X_proc, X_raw)``: the raw score (probability or decision function),
- ``log_odds(model, X_proc, X_raw)``: the log-odds z on which each member is calibrated, p = sigma(a z + b):
  the margin of the model where it has one (LightGBM ``raw_score=True``, XGBoost ``output_margin=True``,
  CatBoost ``prediction_type="RawFormulaVal"``, ``decision_function`` of the logistic regression and of the
  support vector machine; for the perceptron, the input of its output unit, by a forward pass in float64);
  otherwise logit(clip(r, 1e-6, 1 - 1e-6)) of the raw probability r (random forest, extra trees, and TabPFN,
  whose scores come from ``query_tabpfn.py``).

Two paths for the variables: "processed" (``fraud.preprocessor``) for the families that need a numerical
matrix, "raw" for LightGBM and CatBoost, which receive the categorical variables in their original form.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import logit

from fraud.defaults import get_default_params, merge_params

#: clipping of the raw probability before the logit (fixed, not the one of the library)
EPS_LOGIT = 1e-6


def logit_of_probability(r: np.ndarray) -> np.ndarray:
    """Log-odds of a raw probability: logit(clip(r, 1e-6, 1 - 1e-6))."""
    return logit(np.clip(np.asarray(r, dtype=float), EPS_LOGIT, 1.0 - EPS_LOGIT))


def _raw_score(model: Any, X: np.ndarray) -> np.ndarray:
    """A one-dimensional score: the probability of class 1 if available, else the decision function."""
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X))
            if p.ndim == 2 and p.shape[1] == 2:
                return p[:, 1]
            return p.ravel()
        except Exception:  # noqa: BLE001  (SVC with probability=False raises here)
            pass
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    raise AttributeError(f"Model {type(model).__name__} has neither predict_proba nor decision_function.")


def _cast_lgbm_categoricals(X_train_vocab: pd.DataFrame, X_cal: pd.DataFrame, X_test: pd.DataFrame,
                            cat_cols: list[str], unk_token: str = "__UNK__"
                            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cast the categorical columns with a vocabulary from the training part only, plus an UNK level."""
    from pandas.api.types import CategoricalDtype

    Xtr, Xc, Xs = X_train_vocab.copy(), X_cal.copy(), X_test.copy()
    for col in cat_cols:
        train_cats = pd.Index(Xtr[col].astype(str).unique())
        cats = pd.Index(list(train_cats) + [unk_token]).unique()
        ctype = CategoricalDtype(categories=cats, ordered=False)
        Xtr[col] = Xtr[col].astype(str).astype(ctype)

        def _cast_split(s: pd.Series, _cats=cats, _ctype=ctype) -> pd.Series:
            s = s.astype(str)
            s = s.where(s.isin(_cats), unk_token)
            return s.astype(_ctype)

        Xc[col] = _cast_split(Xc[col])
        Xs[col] = _cast_split(Xs[col])
    return Xtr, Xc, Xs


class _Sklearn:
    """scikit-learn and XGBoost families, on the processed variables."""

    feature_path: Literal["processed", "raw"] = "processed"
    param_key: str = ""

    def __init__(self, name: str) -> None:
        self.name = name

    def _class(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def build(self, model_seed: int, auxiliary_seed: int = 0, hyperparameters: dict[str, Any] | None = None) -> Any:
        base = get_default_params(model_seed)[self.param_key]
        return self._class()(**merge_params(base, **(hyperparameters or {})))

    def fit(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame, y: np.ndarray, **_: Any) -> None:
        model.fit(X_proc, y)

    def score(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
        return _raw_score(model, X_proc)

    def log_odds(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
        """Random forest and extra trees: logit of the raw probability."""
        return logit_of_probability(np.asarray(model.predict_proba(X_proc))[:, 1])


class XGBFamily(_Sklearn):
    param_key = "xgb"

    def _class(self):
        from xgboost import XGBClassifier
        return XGBClassifier

    def fit(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame, y: np.ndarray, **_: Any) -> None:
        n0 = max(int(np.sum(y == 0)), 1)
        n1 = max(int(np.sum(y == 1)), 1)
        model.set_params(scale_pos_weight=n0 / n1)
        model.fit(X_proc, y, verbose=False)

    def log_odds(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict(X_proc, output_margin=True), dtype=float).ravel()


class LogRegFamily(_Sklearn):
    param_key = "logreg"

    def _class(self):
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression

    def log_odds(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.decision_function(X_proc), dtype=float).ravel()


class SVMRBFFamily(_Sklearn):
    param_key = "svm_rbf"

    def _class(self):
        from sklearn.svm import SVC
        return SVC

    def log_odds(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.decision_function(X_proc), dtype=float).ravel()


class _Forest(_Sklearn):
    """Forests: trained in parallel (each tree has its own seed, so the result is identical) but
    predicted on a single thread: with ``n_jobs=-1`` scikit-learn sums the predictions of the trees in
    the order in which the threads finish, and the result changes by up to 2e-16 between two calls."""

    def fit(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame, y: np.ndarray, **_: Any) -> None:
        model.fit(X_proc, y)
        model.set_params(n_jobs=1)


class RandomForestFamily(_Forest):
    param_key = "random_forest"

    def _class(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier


class ExtraTreesFamily(_Forest):
    param_key = "extra_trees"

    def _class(self):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier


class MLPFamily(_Sklearn):
    """Perceptron: ``MLPClassifier`` accepts no class weights, so its subsample is oversampled to 50/50
    (the minority class drawn with replacement until it equals the majority), with the auxiliary seed of
    the member; the calibration at the natural share of frauds restores the scale."""

    param_key = "mlp"

    def _class(self):
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier

    def build(self, model_seed: int, auxiliary_seed: int = 0, hyperparameters: dict[str, Any] | None = None) -> Any:
        model = super().build(model_seed, auxiliary_seed, hyperparameters)
        model._oversampling_seed = int(auxiliary_seed)
        return model

    def fit(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame, y: np.ndarray, **_: Any) -> None:
        from fraud.resampling import oversample_to_balanced
        y = np.asarray(y).astype(int)
        X = np.asarray(X_proc)
        idx = oversample_to_balanced(y, seed=model._oversampling_seed)
        model.fit(X[idx], y[idx])

    def log_odds(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
        """Margin: the input of the output unit before the logistic, by a forward pass on ``coefs_`` and
        ``intercepts_`` in float64. ``predict_proba`` saturates at 0 or 1 (in float32 for float32 inputs),
        and the logit of the clipped probability would give the saturated cases the same score."""
        activations = {"relu": lambda a: np.maximum(a, 0.0), "tanh": np.tanh,
                       "logistic": lambda a: 1.0 / (1.0 + np.exp(-a)), "identity": lambda a: a}
        if model.out_activation_ != "logistic" or model.coefs_[-1].shape[1] != 1:
            raise ValueError("the margin is defined for the binary perceptron with a logistic output")
        f = activations[model.activation]
        a = np.asarray(X_proc, dtype=np.float64)
        for W, b in zip(model.coefs_[:-1], model.intercepts_[:-1]):
            a = f(a @ np.asarray(W, np.float64) + np.asarray(b, np.float64))
        z = a @ np.asarray(model.coefs_[-1], np.float64) + np.asarray(model.intercepts_[-1], np.float64)
        return z[:, 0]


class CatBoostFamily:
    """CatBoost on the raw variables, with balanced class weights."""

    name = "CatBoost"
    feature_path: Literal["processed", "raw"] = "raw"

    def build(self, model_seed: int, auxiliary_seed: int = 0, hyperparameters: dict[str, Any] | None = None) -> Any:
        from catboost import CatBoostClassifier
        base = get_default_params(model_seed)["catboost"]
        return CatBoostClassifier(**merge_params(base, **(hyperparameters or {})))

    def fit(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame, y: np.ndarray, *,
            cat_cols: list[str] | None = None, cat_idx: list[int] | None = None, **_: Any) -> None:
        from catboost import Pool
        n0 = max(int(np.sum(y == 0)), 1)
        n1 = max(int(np.sum(y == 1)), 1)
        total = n0 + n1
        model.set_params(class_weights=[total / (2 * n0), total / (2 * n1)])
        model.fit(Pool(X_raw, y, cat_features=cat_idx or []))

    def score(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict_proba(X_raw))[:, 1]

    def log_odds(self, model: Any, X_proc: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict(X_raw, prediction_type="RawFormulaVal"), dtype=float).ravel()


class LightGBMFamily:
    """LightGBM on the raw variables, with the vocabulary of the categorical variables from the
    training part only.

    ``variant="s2"``: the defaults of the library (100 rounds, no class weights) with the constraint on
    the Hessian of the leaves; ``variant="s2_unconstrained"``: the same without the constraint;
    ``variant="s1"``: the base of the first construction with the hyperparameters drawn for the member,
    including ``num_boost_round``."""

    feature_path: Literal["processed", "raw"] = "raw"

    def __init__(self, variant: str) -> None:
        if variant not in ("s1", "s2", "s2_unconstrained"):
            raise ValueError(variant)
        self.variant = variant
        self.name = "LightGBM_constrained" if variant == "s1" else "LightGBM"

    def build(self, model_seed: int, auxiliary_seed: int = 0, hyperparameters: dict[str, Any] | None = None) -> dict[str, Any]:
        defaults = get_default_params(model_seed, unconstrained=self.variant == "s2_unconstrained")
        base = defaults["lgbm_s1" if self.variant == "s1" else "lgbm"]
        params = merge_params(base, **(hyperparameters or {}))
        num_boost_round = int(params.pop("num_boost_round", 100))
        return {"params": params, "num_boost_round": num_boost_round, "booster": None}

    def fit(self, model: dict[str, Any], X_proc: np.ndarray, X_raw: pd.DataFrame, y: np.ndarray, *,
            cat_cols: list[str] | None = None, **_: Any) -> None:
        import lightgbm as lgb
        cat_cols = cat_cols or []
        X_fit, _, _ = _cast_lgbm_categoricals(X_raw, X_raw.iloc[0:0], X_raw.iloc[0:0], cat_cols)
        dtrain = lgb.Dataset(X_fit, label=y, categorical_feature=cat_cols, free_raw_data=False)
        model["booster"] = lgb.train(params=model["params"], train_set=dtrain, num_boost_round=model["num_boost_round"])
        model["cat_cols"] = cat_cols
        model["X_vocab"] = X_raw

    def score(self, model: dict[str, Any], X_proc: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
        _, X_cast, _ = _cast_lgbm_categoricals(model["X_vocab"], X_raw, X_raw.iloc[0:0], model["cat_cols"])
        return model["booster"].predict(X_cast)

    def log_odds(self, model: dict[str, Any], X_proc: np.ndarray, X_raw: pd.DataFrame) -> np.ndarray:
        _, X_cast, _ = _cast_lgbm_categoricals(model["X_vocab"], X_raw, X_raw.iloc[0:0], model["cat_cols"])
        return np.asarray(model["booster"].predict(X_cast, raw_score=True), dtype=float)
