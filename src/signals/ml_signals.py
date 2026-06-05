"""Phase 6 — ML signals (XGBoost + Random Forest ensemble).

Engineers a feature matrix from technical indicators, labels each bar by the
sign of the forward ``horizon``-bar return (up / down / sideways), and trains an
XGBoost classifier plus a Random Forest. A directional signal is only emitted
when **both** models agree; ensemble confidence is the mean class probability.

Honesty note: this is honest, look-ahead-free *structure* — labels use only
future returns that are dropped from training, and ``predict`` uses the latest
features. It is **not** a validated alpha model. The PyTorch LSTM from the spec
is intentionally deferred; ``ensemble_ready`` reflects whether the tree models
trained successfully. Walk-forward retraining is supported via ``retrain``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import ta

logger = logging.getLogger(__name__)

_LABELS = {0: "down", 1: "sideways", 2: "up"}


@dataclass
class MLPrediction:
    direction: str = "sideways"     # up / down / sideways
    agreement: bool = False         # XGB and RF agree on direction
    confidence: float = 0.0         # mean class probability across models
    xgb_direction: str = "sideways"
    rf_direction: str = "sideways"
    top_features: list = field(default_factory=list)
    trained: bool = False


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """~30 technical features per bar (all backward-looking)."""
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)
    f["ret1"] = c.pct_change()
    f["ret5"] = c.pct_change(5)
    f["ret10"] = c.pct_change(10)
    for n in (8, 21, 50, 200):
        f[f"ema_dist{n}"] = c / c.ewm(span=n, adjust=False).mean() - 1
    for n in (7, 14, 21):
        f[f"rsi{n}"] = ta.momentum.RSIIndicator(c, n).rsi()
    macd = ta.trend.MACD(c)
    f["macd"] = macd.macd()
    f["macd_hist"] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(h, l, c)
    f["stoch_k"] = stoch.stoch()
    f["stoch_d"] = stoch.stoch_signal()
    f["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()
    f["cci"] = ta.trend.CCIIndicator(h, l, c).cci()
    f["roc"] = ta.momentum.ROCIndicator(c).roc()
    f["adx"] = ta.trend.ADXIndicator(h, l, c).adx()
    atr = ta.volatility.AverageTrueRange(h, l, c).average_true_range()
    f["atr_pct"] = atr / c
    bb = ta.volatility.BollingerBands(c)
    f["bb_pct"] = bb.bollinger_pband()
    f["bb_width"] = bb.bollinger_wband()
    f["mfi"] = ta.volume.MFIIndicator(h, l, c, v).money_flow_index()
    f["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v).chaikin_money_flow()
    f["obv_chg"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume().pct_change(5)
    f["vol_ratio"] = v / v.rolling(20).mean()
    f["hvol"] = np.log(c / c.shift(1)).rolling(20).std()
    return f


def make_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.01) -> pd.Series:
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    labels = pd.Series(1, index=df.index, dtype=int)  # sideways
    labels[fwd > threshold] = 2   # up
    labels[fwd < -threshold] = 0  # down
    return labels


class MLEnsemble:
    def __init__(self, horizon: int = 5, threshold: float = 0.01):
        self.horizon = horizon
        self.threshold = threshold
        self._xgb = None
        self._rf = None
        self._features: list[str] = []
        self.trained = False

    @property
    def ensemble_ready(self) -> bool:
        return self.trained and self._xgb is not None and self._rf is not None

    def train(self, df: pd.DataFrame) -> bool:
        """Fit XGBoost + Random Forest on the labelled history. Returns success."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from xgboost import XGBClassifier

            feats = build_features(df)
            labels = make_labels(df, self.horizon, self.threshold)
            data = feats.copy()
            data["_label"] = labels
            # Drop the unlabelled forward tail and any NaN warmup rows.
            data = data.iloc[: -self.horizon].replace([np.inf, -np.inf], np.nan).dropna()
            if len(data) < 100:
                logger.info("ML: insufficient clean rows to train (%d)", len(data))
                return False

            self._features = [c for c in data.columns if c != "_label"]
            X, y = data[self._features].values, data["_label"].values
            if len(np.unique(y)) < 2:
                return False

            self._xgb = XGBClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                tree_method="hist", verbosity=0,
            )
            self._rf = RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=20,
                n_jobs=-1, random_state=7,
            )
            self._xgb.fit(X, y)
            self._rf.fit(X, y)
            self.trained = True
            return True
        except Exception:
            logger.exception("MLEnsemble.train failed")
            return False

    def predict(self, df: pd.DataFrame) -> MLPrediction:
        """Predict the next-``horizon`` direction from the latest bar."""
        if not self.ensemble_ready:
            return MLPrediction(trained=False)
        try:
            feats = build_features(df)[self._features]
            row = feats.replace([np.inf, -np.inf], np.nan).iloc[[-1]]
            if row.isna().any(axis=1).iloc[0]:
                row = row.fillna(0.0)
            X = row.values

            xgb_proba = self._xgb.predict_proba(X)[0]
            rf_proba = self._rf.predict_proba(X)[0]
            xgb_cls = int(np.argmax(xgb_proba))
            rf_cls = int(np.argmax(rf_proba))

            agreement = xgb_cls == rf_cls
            mean_conf = float((xgb_proba.max() + rf_proba.max()) / 2)
            importances = sorted(
                zip(self._features, self._xgb.feature_importances_),
                key=lambda kv: kv[1], reverse=True,
            )[:5]
            return MLPrediction(
                direction=_LABELS[xgb_cls] if agreement else "sideways",
                agreement=agreement,
                confidence=mean_conf if agreement else 0.0,
                xgb_direction=_LABELS[xgb_cls],
                rf_direction=_LABELS[rf_cls],
                top_features=[name for name, _ in importances],
                trained=True,
            )
        except Exception:
            logger.exception("MLEnsemble.predict failed")
            return MLPrediction(trained=True)

    # Walk-forward retrain hook (called by the live loop every N days).
    def retrain(self, df: pd.DataFrame) -> bool:
        return self.train(df)
