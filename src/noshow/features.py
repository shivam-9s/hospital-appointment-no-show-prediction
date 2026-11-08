"""Preprocessing building blocks shared by training and inference (one code path = no train/serve skew)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each numeric column to [q_low, q_high] learned on the TRAINING data (outlier handling)."""

    def __init__(self, q_low: float = 0.001, q_high: float = 0.999):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.low_ = X.quantile(self.q_low).to_numpy()
        self.high_ = X.quantile(self.q_high).to_numpy()
        return self

    def transform(self, X):
        return np.clip(pd.DataFrame(X).to_numpy(dtype=float), self.low_, self.high_)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns that depend only on a row's own raw values (used by the API for a single booking)."""
    out = df.copy()
    out["log_lead_days"] = np.log1p(out["lead_days"])
    out["same_day"] = (out["lead_days"] == 0).astype(int)
    out["has_history"] = (out["prior_appointments"] > 0).astype(int)
    out["prior_no_show_rate"] = np.where(
        out["prior_appointments"] > 0, out["prior_no_shows"] / out["prior_appointments"].clip(lower=1), np.nan
    )
    return out


def build_preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("winsor", Winsorizer()), ("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), config.NUMERIC_FEATURES),
            ("bin", "passthrough", config.BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=config.MIN_NEIGHBOURHOOD_COUNT,
                    sparse_output=False,
                ),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


class PlattCalibrator:
    """Smooth, monotone probability calibration: logistic regression on logit(raw score).

    Chosen over isotonic regression because isotonic is a step function: tiny input changes cause
    jumps of several probability points, which makes per-prediction explanations unstable.
    """

    def fit(self, score, y):
        from sklearn.linear_model import LogisticRegression

        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(score), y)
        return self

    @staticmethod
    def _logit(s):
        s = np.clip(np.asarray(s, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s)).reshape(-1, 1)

    def predict(self, score):
        return self.lr_.predict_proba(self._logit(score))[:, 1]
