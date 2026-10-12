"""Шаг 13: улучшение лучшей модели через калибровку, блендинг и stacking.

Ключевая методологическая правка: главный кандидат `FeatureStack_LogReg`
обучается поверх 25 исходных факторов и 11 прогнозов моделей. Это позволяет
после улучшения интерпретировать вклад и рыночных признаков, и model signals.
"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import softmax
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    BLEND_MIN_TRAIN,
    BLEND_OPTUNA_TRIALS,
    DATA_PROCESSED,
    DATA_RAW,
    FIGURES,
    RANDOM_STATE,
    ROLLING_AUC_WINDOW,
    SOFTMAX_BLEND_TEMPERATURE,
    TABLES,
    THRESHOLD_GRID_HIGH,
    THRESHOLD_GRID_LOW,
    THRESHOLD_GRID_SIZE,
)
from src.utils import (  # noqa: E402
    diebold_mariano as _shared_dm,
    expanding_walk_forward_indices,
    holm_bonferroni as _shared_holm,
    max_drawdown as _shared_max_drawdown,
    per_fold_log_loss as _shared_per_fold_log_loss,
    rolling_auc as _shared_rolling_auc,
    setup_plot_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_13")

SUBFOLDER = "step_13"
MODEL_PRED_COLS = [
    "LogisticRegression", "SVM", "RandomForest", "XGBoost", "CatBoost",
    "LSTM", "TFT", "TabTransformer", "VariantA", "VariantB", "VariantC",
]


def _ensure_dirs() -> None:
    os.makedirs(os.path.join(FIGURES, SUBFOLDER), exist_ok=True)
    os.makedirs(os.path.join(TABLES, SUBFOLDER), exist_ok=True)
    os.makedirs(DATA_PROCESSED, exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> None:
    _ensure_dirs()
    base = os.path.join(FIGURES, SUBFOLDER, name)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_future_returns(index: pd.DatetimeIndex) -> pd.Series:
    imoex = pd.read_csv(os.path.join(DATA_RAW, "imoex.csv"), parse_dates=["date"])
    close = imoex.sort_values("date").set_index("date")["close"]
    monthly_ret = close.resample("ME").last().pct_change()
    return monthly_ret.shift(-1).reindex(index).fillna(0)


def align_prediction_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    baseline = pd.read_parquet(os.path.join(DATA_PROCESSED, "baseline_predictions.parquet"))
    regime = pd.read_parquet(os.path.join(DATA_PROCESSED, "regime_predictions.parquet"))
    preds = pd.concat([baseline, regime], axis=1)
    y_raw = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_target.parquet"))["y_next"].dropna().astype(int)
    y = y_raw.reindex(preds.index)
    features = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_features.parquet")).reindex(preds.index).fillna(0)
    regimes = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_regimes.parquet"))["regime"].astype(int).reindex(preds.index)
    returns = load_future_returns(preds.index)
    preds = preds.reindex(columns=MODEL_PRED_COLS).clip(0, 1)
    walk = pd.read_parquet(os.path.join(DATA_PROCESSED, "walk_forward_regimes_n2.parquet"))
    return preds, features, y, returns, regimes, walk


def _resolve_threshold(threshold: float | pd.Series, index: pd.Index) -> pd.Series:
    """Приводит порог к Series, выровненной по index.

    Если threshold скаляр — возвращает константную серию. Если Series —
    переиндексирует, отсутствующие значения заполняет 0.5 (нейтральный порог).
    """
    if isinstance(threshold, pd.Series):
        return threshold.reindex(index).fillna(0.5)
    return pd.Series(float(threshold), index=index)


def classification_metrics(
    y: pd.Series,
    pred: pd.Series,
    threshold: float | pd.Series = 0.5,
) -> dict[str, float]:
    """Метрики классификации; threshold может быть скаляр или Series (walk-forward)."""
    df = pd.concat([y.rename("y"), pred.rename("pred")], axis=1, join="inner").dropna()
    y_true = df["y"].astype(int)
    p = df["pred"].clip(1e-6, 1 - 1e-6)
    thr_series = _resolve_threshold(threshold, p.index)
    cls = (p >= thr_series).astype(int)
    auc = roc_auc_score(y_true, p) if y_true.nunique() > 1 else np.nan
    return {
        "AUC-ROC": float(auc),
        "Accuracy": float(accuracy_score(y_true, cls)),
        "F1 macro": float(f1_score(y_true, cls, average="macro", zero_division=0)),
        "Precision(y=1)": float(precision_score(y_true, cls, zero_division=0)),
        "Log-Loss": float(log_loss(y_true, p, labels=[0, 1])),
    }


def strategy_returns(
    pred: pd.Series,
    future_returns: pd.Series,
    threshold: float | pd.Series = 0.5,
) -> pd.Series:
    """Доходность стратегии; threshold — скаляр или Series (walk-forward)."""
    aligned = pd.concat(
        [pred.rename("pred"), future_returns.rename("ret")],
        axis=1, join="inner",
    ).fillna(0)
    thr_series = _resolve_threshold(threshold, aligned.index)
    return aligned["ret"].where(aligned["pred"] >= thr_series, 0.0)


def max_drawdown(returns: pd.Series) -> float:
    """Тонкая обёртка над utils.max_drawdown — для обратной совместимости тестов."""
    return _shared_max_drawdown(returns)


def rolling_auc_samples(y: pd.Series, pred: pd.Series, window: int = ROLLING_AUC_WINDOW) -> pd.Series:
    """Тонкая обёртка над utils.rolling_auc — для обратной совместимости тестов."""
    return _shared_rolling_auc(y, pred, window)


def time_series_isotonic_calibration(y: pd.Series, pred: pd.Series, min_train: int = BLEND_MIN_TRAIN) -> pd.Series:
    df = pd.concat([y.rename("y"), pred.rename("pred")], axis=1, join="inner").dropna()
    out = pd.Series(df["pred"].values, index=df.index, name=f"{pred.name or 'pred'}_calibrated")
    for train_slice, i in expanding_walk_forward_indices(len(df), min_train):
        train = df.iloc[train_slice]
        if train["y"].nunique() < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(train["pred"], train["y"])
        out.iloc[i] = float(iso.predict([df["pred"].iloc[i]])[0])
    return out.clip(0, 1)


def _threshold_grid() -> np.ndarray:
    """Сетка порогов из config (для F1 и доходности стратегии)."""
    return np.linspace(THRESHOLD_GRID_LOW, THRESHOLD_GRID_HIGH, THRESHOLD_GRID_SIZE)


def tune_threshold_walk_forward(
    y: pd.Series,
    pred: pd.Series,
    future_returns: pd.Series | None = None,
    objective: str = "f1_macro",
    min_train: int = BLEND_MIN_TRAIN,
) -> tuple[pd.Series, pd.Series]:
    """Walk-forward подбор бинарного порога (F1 / доходность стратегии).

    Для каждого индекса i ≥ min_train:
        1. На срезе [:i] (включительно) перебирается сетка порогов
           THRESHOLD_GRID_SIZE точек в [THRESHOLD_GRID_LOW, THRESHOLD_GRID_HIGH].
        2. Для каждого порога считается F1-macro или итоговая доходность стратегии
           по тем же [:i] OOF-данным.
        3. Лучший порог сохраняется в threshold[i] и применяется к pred[i] →
           signal[i] = int(pred[i] ≥ threshold[i]).

    Args:
        y: целевая 0/1 серия (out-of-fold target).
        pred: out-of-fold вероятности модели.
        future_returns: следующие месячные доходности — нужны для objective="strategy_return".
        objective: "f1_macro" или "strategy_return".
        min_train: минимум прошлых OOF-точек.

    Returns:
        Кортеж (signal_series, threshold_series) одинаковой длины с pred.
        До min_train — по умолчанию порог 0.5, signal = (pred ≥ 0.5).
    """
    if objective == "strategy_return" and future_returns is None:
        raise ValueError("future_returns required for strategy_return objective")

    df = pd.concat([y.rename("y"), pred.rename("pred")], axis=1, join="inner").dropna()
    if future_returns is not None:
        df = df.join(future_returns.rename("ret"), how="left").fillna({"ret": 0.0})

    grid = _threshold_grid()
    thresholds = pd.Series(0.5, index=df.index, name=f"threshold_{objective}")
    signals = pd.Series((df["pred"] >= 0.5).astype(int), index=df.index, name=f"signal_{objective}")

    for train_slice, i in expanding_walk_forward_indices(len(df), min_train):
        train = df.iloc[train_slice]
        if train["y"].nunique() < 2:
            continue
        best_thr, best_score = 0.5, -np.inf
        for thr in grid:
            cls = (train["pred"] >= thr).astype(int)
            if objective == "strategy_return":
                strat = train["ret"].where(cls == 1, 0.0)
                score = float((1 + strat).prod() - 1)
            else:
                score = float(f1_score(train["y"], cls, average="macro", zero_division=0))
            if score > best_score:
                best_thr, best_score = float(thr), score
        thresholds.iloc[i] = best_thr
        signals.iloc[i] = int(df["pred"].iloc[i] >= best_thr)

    return signals, thresholds


def softmax_auc_blend(
    preds: pd.DataFrame,
    y: pd.Series,
    min_train: int = BLEND_MIN_TRAIN,
    temperature: float = SOFTMAX_BLEND_TEMPERATURE,
    return_weights_per_row: bool = False,
) -> tuple[pd.Series, dict[str, float]] | tuple[pd.Series, dict[str, float], pd.DataFrame]:
    """Walk-forward AUC-soft blending без look-ahead.

    Для каждого i ≥ min_train: AUC каждой модели считается на [:i] OOF-точках,
    softmax(AUC * τ) даёт веса, веса применяются к строке i.

    Возвращает OOF-предсказание и словарь *последних* (для финальной строки)
    весов — для логирования и графиков. При return_weights_per_row=True
    дополнительно возвращает DataFrame walk-forward весов (index=preds.index,
    columns=preds.columns) — нужен для интерпретации SHAP в шаге 14.
    """
    y_aligned = y.reindex(preds.index)
    out = pd.Series(preds.mean(axis=1).values, index=preds.index, name="Blend_softmax_auc")
    last_weights = np.ones(preds.shape[1]) / preds.shape[1]
    weights_per_row = pd.DataFrame(
        np.tile(last_weights, (len(preds), 1)),
        index=preds.index, columns=preds.columns,
    )

    for train_slice, i in expanding_walk_forward_indices(len(preds), min_train):
        y_train = y_aligned.iloc[train_slice]
        if y_train.dropna().nunique() < 2:
            weights_per_row.iloc[i] = last_weights
            continue
        aucs = []
        for col in preds.columns:
            p_train = preds[col].iloc[train_slice]
            aucs.append(float(roc_auc_score(y_train, p_train)))
        w = softmax(np.asarray(aucs) * temperature)
        out.iloc[i] = float(np.dot(preds.iloc[i].values, w))
        weights_per_row.iloc[i] = w
        last_weights = w

    weights_dict = dict(zip(preds.columns, map(float, last_weights), strict=False))
    if return_weights_per_row:
        return out.clip(0, 1), weights_dict, weights_per_row
    return out.clip(0, 1), weights_dict


def _optuna_search_weights(
    preds_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Случайный поиск весов на симплексе (Dirichlet) с максимизацией AUC.

    Используется как поиск весов внутри одного шага walk-forward. Не делает
    look-ahead: видит только y_train / preds_train.
    """
    n_models = preds_train.shape[1]
    best_w = np.ones(n_models) / n_models
    best_auc = -np.inf
    if y_train.dropna().nunique() < 2:
        return best_w
    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(n_models))
        blended = preds_train.values @ w
        try:
            auc = float(roc_auc_score(y_train.values, blended))
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_w = w
    return best_w


def optuna_weight_blend(
    preds: pd.DataFrame,
    y: pd.Series,
    n_trials: int = BLEND_OPTUNA_TRIALS,
    min_train: int = BLEND_MIN_TRAIN,
    return_weights_per_row: bool = False,
) -> tuple[pd.Series, dict[str, float]] | tuple[pd.Series, dict[str, float], pd.DataFrame]:
    """Walk-forward random-search blending на симплексе без look-ahead.

    Для каждого i ≥ min_train веса подбираются на [:i] (random search на
    симплексе с фиксированным seed), потом применяются к строке i.

    Возвращает OOF-предсказание и словарь *последних* весов (для финальной
    строки). При return_weights_per_row=True дополнительно возвращает
    DataFrame walk-forward весов для каждой строки — нужен для шага 14.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y_aligned = y.reindex(preds.index)
    out = pd.Series(preds.mean(axis=1).values, index=preds.index, name="Blend_optuna")
    last_weights = np.ones(preds.shape[1]) / preds.shape[1]
    weights_per_row = pd.DataFrame(
        np.tile(last_weights, (len(preds), 1)),
        index=preds.index, columns=preds.columns,
    )

    for train_slice, i in expanding_walk_forward_indices(len(preds), min_train):
        y_train = y_aligned.iloc[train_slice]
        preds_train = preds.iloc[train_slice]
        w = _optuna_search_weights(preds_train, y_train, n_trials, rng)
        out.iloc[i] = float(np.dot(preds.iloc[i].values, w))
        weights_per_row.iloc[i] = w
        last_weights = w

    weights_dict = dict(zip(preds.columns, map(float, last_weights), strict=False))
    if return_weights_per_row:
        return out.clip(0, 1), weights_dict, weights_per_row
    return out.clip(0, 1), weights_dict


def _stacking_design(preds: pd.DataFrame, features: pd.DataFrame | None = None) -> pd.DataFrame:
    if features is None:
        return preds.copy()
    feature_cols = [f"feat__{c}" for c in features.columns]
    pred_cols = [f"model__{c}" for c in preds.columns]
    Xf = features.copy()
    Xf.columns = feature_cols
    Xp = preds.copy()
    Xp.columns = pred_cols
    return pd.concat([Xf, Xp], axis=1).fillna(0)


def stacking_oof(
    preds: pd.DataFrame,
    y: pd.Series,
    features: pd.DataFrame | None = None,
    min_train: int = BLEND_MIN_TRAIN,
) -> tuple[pd.Series, pd.DataFrame]:
    X = _stacking_design(preds, features)
    y = y.reindex(X.index).astype(int)
    out = pd.Series(preds.mean(axis=1).values, index=X.index, name="Stacking_LogReg")
    coef_rows = []
    last_model: LogisticRegression | None = None
    last_scaler: StandardScaler | None = None
    for i in range(min_train, len(X)):
        y_train = y.iloc[:i]
        if y_train.nunique() < 2:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X.iloc[:i])
        model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000)
        model.fit(X_train, y_train)
        out.iloc[i] = float(model.predict_proba(scaler.transform(X.iloc[[i]]))[0, 1])
        last_model = model
        last_scaler = scaler
    if last_model is None:
        coef = np.zeros(X.shape[1])
    else:
        coef = last_model.coef_[0] / np.maximum(last_scaler.scale_, 1e-12)
    for feature, value in zip(X.columns, coef, strict=False):
        if feature.startswith("feat__"):
            source = "market_feature"
            raw_feature = feature.replace("feat__", "", 1)
        elif feature.startswith("model__"):
            source = "model_prediction"
            raw_feature = feature.replace("model__", "", 1)
        else:
            source = "model_prediction"
            raw_feature = feature
        coef_rows.append({
            "feature": raw_feature,
            "design_feature": feature,
            "source": source,
            "coefficient": float(value),
            "abs_coefficient": float(abs(value)),
        })
    return out.clip(0, 1), pd.DataFrame(coef_rows).sort_values("abs_coefficient", ascending=False)


def regime_hard_switch(preds: pd.DataFrame, regimes: pd.Series) -> pd.Series:
    regimes = regimes.reindex(preds.index).astype(int)
    return pd.Series(np.where(regimes == 0, preds["VariantA"], preds["VariantC"]), index=preds.index, name="RegimeSwitch_Hard").clip(0, 1)


def align_walk_forward_soft_probs(walk_df: pd.DataFrame) -> pd.DataFrame:
    df = walk_df.copy()
    if "regime_prob_0" not in df or "regime_prob_1" not in df:
        raise ValueError("walk_df must contain regime_prob_0/regime_prob_1")
    # In this project codebook after stage 09: 0=bear, 1=bull. Preserve but normalize.
    total = (df["regime_prob_0"] + df["regime_prob_1"]).replace(0, np.nan)
    df["p_bear"] = (df["regime_prob_0"] / total).fillna(0.5)
    df["p_bull"] = (df["regime_prob_1"] / total).fillna(0.5)
    return df


def regime_soft_switch(preds: pd.DataFrame, walk_probs: pd.DataFrame) -> pd.Series:
    probs = align_walk_forward_soft_probs(walk_probs).reindex(preds.index)
    p_bear = probs["p_bear"].fillna(0.5)
    p_bull = probs["p_bull"].fillna(0.5)
    return (p_bear * preds["VariantA"] + p_bull * preds["VariantC"]).rename("RegimeSwitch_Soft").clip(0, 1)


def _threshold_label(thr: float | pd.Series) -> float:
    """Скалярная метка порога для T24: среднее, если walk-forward Series, иначе сам скаляр."""
    if isinstance(thr, pd.Series):
        valid = thr.dropna()
        return float(valid.mean()) if not valid.empty else 0.5
    return float(thr)


def evaluate_systems(
    predictions: pd.DataFrame,
    y: pd.Series,
    future_returns: pd.Series,
    thresholds: dict[str, float | pd.Series] | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or {}
    rows = []
    for name in predictions.columns:
        thr = thresholds.get(name, 0.5)
        metrics = classification_metrics(y.reindex(predictions.index), predictions[name], threshold=thr)
        strat = strategy_returns(predictions[name], future_returns, threshold=thr)
        metrics.update({
            "Система": name,
            "Threshold": _threshold_label(thr),
            "Доходность стратегии": float((1 + strat).prod() - 1),
            "Max drawdown": max_drawdown(strat),
        })
        rows.append(metrics)
    buy_hold = future_returns.reindex(predictions.index).fillna(0)
    rows.append({
        "Система": "BuyAndHold",
        "AUC-ROC": np.nan,
        "Accuracy": np.nan,
        "F1 macro": np.nan,
        "Precision(y=1)": np.nan,
        "Log-Loss": np.nan,
        "Threshold": np.nan,
        "Доходность стратегии": float((1 + buy_hold).prod() - 1),
        "Max drawdown": max_drawdown(buy_hold),
    })
    cols = ["Система", "AUC-ROC", "Accuracy", "F1 macro", "Precision(y=1)", "Log-Loss", "Threshold", "Доходность стратегии", "Max drawdown"]
    return pd.DataFrame(rows)[cols].sort_values("AUC-ROC", ascending=False, na_position="last")


def diebold_mariano(loss_a: pd.Series, loss_b: pd.Series, h: int = 1) -> tuple[float, float]:
    """Тонкая обёртка над utils.diebold_mariano — для обратной совместимости."""
    return _shared_dm(loss_a, loss_b, h=h)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Тонкая обёртка над utils.holm_bonferroni — для обратной совместимости."""
    return _shared_holm(p_values)


def pairwise_tests(
    predictions: pd.DataFrame,
    y: pd.Series,
    best_name: str,
    h: int = 1,
) -> pd.DataFrame:
    """Парные сравнения улучшенных систем через Diebold-Mariano по log-loss.

    Заменяет старый t-test на overlapping rolling AUC. DM считается на
    per-fold (per-month) log-loss и не требует i.i.d. предположения о
    перекрывающихся окнах.
    """
    best_loss = _shared_per_fold_log_loss(y, predictions[best_name])
    rows = []
    raw_p_values: list[float] = []
    for col in predictions.columns:
        if col == best_name:
            continue
        other_loss = _shared_per_fold_log_loss(y, predictions[col])
        dm_stat, p_value = _shared_dm(best_loss, other_loss, h=h)
        rows.append({
            "best": best_name,
            "other": col,
            "dm_stat": dm_stat,
            "p_value_raw": p_value,
        })
        raw_p_values.append(p_value)
    p_holm = _shared_holm(raw_p_values)
    for row, p_adj in zip(rows, p_holm, strict=True):
        row["p_value_holm"] = p_adj
        row["significant_holm_5pct"] = bool(p_adj < 0.05)
    return pd.DataFrame(rows)


def build_improved_predictions() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    dict[str, float | pd.Series],
    dict[str, dict[str, float]],
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    """Собирает улучшенные системы строго walk-forward.

    Все слои улучшения (калибровка, подбор порога, soft/optuna блендинг,
    stacking, FeatureStack) используют только прошлые OOF-данные через
    expanding-window. Исходные blend_inputs — VariantA/VariantC/SVM —
    являются OOF-предсказаниями из шагов 8/9.
    """
    preds, features, y, future_returns, regimes, walk = align_prediction_frame()
    candidates = pd.DataFrame(index=preds.index)
    candidates["VariantA"] = preds["VariantA"]
    candidates["VariantC"] = preds["VariantC"]
    candidates["SVM"] = preds["SVM"]
    candidates["VariantA_calibrated"] = time_series_isotonic_calibration(
        y, preds["VariantA"], min_train=BLEND_MIN_TRAIN,
    )

    # Walk-forward подбор порога: signal — 0/1 OOF-сигнал, threshold — серия порогов
    signal_f1, threshold_f1 = tune_threshold_walk_forward(
        y, preds["VariantA"], objective="f1_macro", min_train=BLEND_MIN_TRAIN,
    )
    signal_ret, threshold_ret = tune_threshold_walk_forward(
        y, preds["VariantA"], future_returns, objective="strategy_return",
        min_train=BLEND_MIN_TRAIN,
    )

    blend_inputs = preds[["VariantA", "VariantC", "SVM"]]
    softmax_blend, softmax_weights, softmax_weights_per_row = softmax_auc_blend(
        blend_inputs, y, min_train=BLEND_MIN_TRAIN, return_weights_per_row=True,
    )
    candidates["Blend_softmax_auc"] = softmax_blend
    optuna_blend, optuna_weights, optuna_weights_per_row = optuna_weight_blend(
        blend_inputs, y, n_trials=BLEND_OPTUNA_TRIALS, min_train=BLEND_MIN_TRAIN,
        return_weights_per_row=True,
    )
    candidates["Blend_optuna"] = optuna_blend
    candidates["Stacking_LogReg"], stacking_coef = stacking_oof(
        preds, y, features=None, min_train=BLEND_MIN_TRAIN,
    )
    feature_stack, feature_coef = stacking_oof(
        preds, y, features=features, min_train=BLEND_MIN_TRAIN,
    )
    candidates["FeatureStack_LogReg"] = feature_stack.rename("FeatureStack_LogReg")
    candidates["RegimeSwitch_Hard"] = regime_hard_switch(preds, regimes)
    candidates["RegimeSwitch_Soft"] = regime_soft_switch(preds, walk)

    # threshold-tuned кандидаты: вероятность остаётся VariantA, классификатор —
    # walk-forward порог из соответствующей серии. Для оценки в evaluate_systems
    # эта серия передаётся через словарь thresholds.
    candidates["VariantA_threshold_f1"] = preds["VariantA"]
    candidates["VariantA_threshold_return"] = preds["VariantA"]
    thresholds: dict[str, float | pd.Series] = {
        "VariantA_threshold_f1": threshold_f1,
        "VariantA_threshold_return": threshold_ret,
    }

    weights = {"softmax": softmax_weights, "optuna": optuna_weights}
    weights_per_row = {
        "softmax": softmax_weights_per_row,
        "optuna": optuna_weights_per_row,
    }
    coef = pd.concat([
        stacking_coef.assign(model="Stacking_LogReg"),
        feature_coef.assign(model="FeatureStack_LogReg"),
    ], ignore_index=True)
    return (
        candidates.clip(0, 1),
        features,
        y,
        future_returns,
        preds,
        thresholds,
        weights,
        coef,
        weights_per_row,
    )


def fig_77_auc_bar(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    plot_df = table.dropna(subset=["AUC-ROC"]).sort_values("AUC-ROC", ascending=False)
    sns.barplot(data=plot_df, x="AUC-ROC", y="Система", ax=ax, color="#4C78A8")
    ax.set_title("AUC-ROC улучшенных систем")
    save_figure(fig, "fig_77_auc_bar_improvements")


def fig_78_cumulative_returns(predictions: pd.DataFrame, y: pd.Series, future_returns: pd.Series, best_name: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in [best_name, "VariantA"]:
        if name in predictions:
            equity = (1 + strategy_returns(predictions[name], future_returns)).cumprod()
            ax.plot(equity.index, equity.values, label=name)
    ax.plot((1 + future_returns.reindex(predictions.index).fillna(0)).cumprod(), label="BuyAndHold")
    ax.set_title("Кумулятивная доходность: лучшая система vs VariantA vs buy-and-hold")
    ax.set_ylabel("Индекс капитала")
    ax.legend()
    save_figure(fig, "fig_78_cumulative_returns_improvements")


def fig_79_roc(predictions: pd.DataFrame, y: pd.Series, best_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in [best_name, "VariantA", "FeatureStack_LogReg", "RegimeSwitch_Soft"]:
        if name in predictions:
            fpr, tpr, _ = roc_curve(y.reindex(predictions.index), predictions[name])
            ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_title("ROC-кривые улучшенных систем")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    save_figure(fig, "fig_79_roc_curves_improvements")


def fig_80_calibration(predictions: pd.DataFrame, y: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in ["VariantA", "VariantA_calibrated"]:
        prob_true, prob_pred = calibration_curve(y.reindex(predictions.index), predictions[name], n_bins=6, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black")
    ax.set_title("Калибровка вероятностей до и после IsotonicRegression")
    ax.set_xlabel("Средняя предсказанная вероятность")
    ax.set_ylabel("Фактическая доля y=1")
    ax.legend()
    save_figure(fig, "fig_80_calibration_before_after")


def fig_81_coefficients(coef: pd.DataFrame) -> None:
    top = coef.loc[coef["model"] == "FeatureStack_LogReg"].sort_values("abs_coefficient", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, x="abs_coefficient", y="feature", ax=ax, color="#72B7B2")
    ax.set_title("Вклад признаков в FeatureStack_LogReg (|коэффициент|)")
    ax.set_xlabel("|коэффициент|")
    ax.set_ylabel("")
    save_figure(fig, "fig_81_stacking_model_weights")


def fig_82_switch_probs(preds: pd.DataFrame, walk: pd.DataFrame) -> None:
    probs = align_walk_forward_soft_probs(walk).reindex(preds.index)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(probs.index, probs["p_bear"], label="P(медвежий)")
    ax.plot(probs.index, probs["p_bull"], label="P(бычий)")
    ax.set_title("Walk-forward вероятности режимов для мягкого переключения моделей")
    ax.set_ylabel("Вероятность")
    ax.legend()
    save_figure(fig, "fig_82_regime_switching_probs")


def main() -> None:
    setup_plot_style()
    (
        predictions,
        features,
        y,
        future_returns,
        base_preds,
        thresholds,
        _weights,
        coef,
        weights_per_row,
    ) = build_improved_predictions()
    table = evaluate_systems(predictions, y, future_returns, thresholds=thresholds)
    best_name = table.dropna(subset=["AUC-ROC"]).iloc[0]["Система"]
    pairwise = pairwise_tests(predictions, y, best_name)
    _ensure_dirs()
    predictions.to_parquet(os.path.join(DATA_PROCESSED, "improved_model_predictions.parquet"))
    features.to_parquet(os.path.join(DATA_PROCESSED, "improved_model_features.parquet"))
    # Сохраняем walk-forward веса бленда отдельным файлом — нужны для шага 14
    # (интерпретация SHAP победителя без повторного in-sample подбора).
    weights_per_row["optuna"].to_parquet(
        os.path.join(DATA_PROCESSED, "improved_blend_weights_optuna.parquet"),
    )
    weights_per_row["softmax"].to_parquet(
        os.path.join(DATA_PROCESSED, "improved_blend_weights_softmax.parquet"),
    )
    coef.to_csv(os.path.join(TABLES, SUBFOLDER, "T26_feature_stack_coefficients.csv"), index=False, encoding="utf-8-sig")
    table.to_csv(os.path.join(TABLES, SUBFOLDER, "T24_model_improvements.csv"), index=False, encoding="utf-8-sig")
    pairwise.to_csv(os.path.join(TABLES, SUBFOLDER, "T25_improvement_pairwise_tests.csv"), index=False, encoding="utf-8-sig")
    fig_77_auc_bar(table)
    fig_78_cumulative_returns(predictions, y, future_returns, str(best_name))
    fig_79_roc(predictions, y, str(best_name))
    fig_80_calibration(predictions, y)
    fig_81_coefficients(coef)
    fig_82_switch_probs(base_preds, pd.read_parquet(os.path.join(DATA_PROCESSED, "walk_forward_regimes_n2.parquet")))
    log.info("Лучшая улучшенная система по AUC: %s", best_name)
    log.info("Walk-forward пороги (mean): %s",
             {k: float(v.mean()) if isinstance(v, pd.Series) else v for k, v in thresholds.items()})
    log.info("Сохранены T24/T25 и fig_77-82")


if __name__ == "__main__":
    main()
