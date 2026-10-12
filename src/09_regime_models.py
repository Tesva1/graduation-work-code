"""Шаг 8: режимно-адаптивные модели (варианты A, B, C).

Цель — получить три варианта прогнозов, в которых ML-модели обучаются с учётом
скрытых рыночных режимов из HMM (шаг 7), не нарушая принцип отсутствия
look-ahead в *обучении ML*.

Архитектура:
* Метки для разделения train-подвыборок — глобальный HMM
  (`data/processed/monthly_regimes.parquet`, 132 точки, 0=медвежий, 1=бычий).
  Используется только как фильтр строк, режим НЕ подаётся в признаки.
  Мягкий look-ahead в структуре HMM признан допустимым (Hamilton 1989).
* Для варианта B на тестовом окне используются walk-forward вероятности
  из `walk_forward_regimes_n2.parquet` (без look-ahead, кодировка инвертирована —
  выравнивается в коде).

Варианты:
* A. SVM(RBF) на каждый режим. Если |train_sub| < HMM_MIN_REGIME_SIZE —
     fallback на baseline-SVM, обученный на полном train.
* B. SVM на каждый режим + мягкое смешивание walk-forward вероятностями.
* C. Из 5 классических моделей (LogReg, SVM, RF, XGBoost, CatBoost) выбирается
     лучшая для каждого режима по внутренней time-series CV (3 фолда, gap=1).

Вход:  monthly_features.parquet, monthly_target.parquet, monthly_regimes.parquet,
       walk_forward_regimes_n2.parquet, data/raw/imoex.csv.
Выход: data/processed/regime_predictions.parquet (VariantA/B/C),
       tables/step_09/T20_best_per_regime.csv.

Запуск: python src/09_regime_models.py
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    DATA_PROCESSED,
    HMM_MIN_REGIME_SIZE,
    REGIME_INNER_CV_GAP,
    REGIME_INNER_CV_N_SPLITS,
    REGIME_VARIANT_A_MODEL,
    REGIME_VARIANT_C_MODELS,
)
from src.utils import save_table  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_09")

SUBFOLDER = "step_09"
FALLBACK_TAG = "fallback"  # Сигнал в run_variant_c, что обучение режимной модели не состоялось


def _load_baseline_module():
    """Динамически импортирует src/08_baseline_models.py (имя начинается с цифры)."""
    module_path = os.path.join(PROJECT_ROOT, "src", "08_baseline_models.py")
    spec = importlib.util.spec_from_file_location("baseline_models", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_baseline = _load_baseline_module()
walk_forward_splits = _baseline.walk_forward_splits
make_estimator = _baseline.make_estimator
default_params = _baseline.default_params

# Sanity: все модели для варианта C должны быть известны базовому модулю
for _name in REGIME_VARIANT_C_MODELS:
    _baseline.default_params(_name)  # Поднимет KeyError, если 08 рефакторился


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Загружает X, y, returns, regimes_global, walk_softprobs (сырой).

    Returns:
        Кортеж (X, y, returns, regimes_global, walk_softprobs).
    """
    X, y, returns = _baseline.load_data()

    # Согласованно с baseline (08): NaN в первых ~12 строках (ret_12m) заполняем 0.
    # См. 08_baseline_models.py: X.loc[train_idx].fillna(0).
    X = X.fillna(0)

    reg_path = os.path.join(DATA_PROCESSED, "monthly_regimes.parquet")
    regimes_global = pd.read_parquet(reg_path)["regime"].astype(int)
    regimes_global = regimes_global.reindex(y.index)

    walk_path = os.path.join(DATA_PROCESSED, "walk_forward_regimes_n2.parquet")
    walk_softprobs = pd.read_parquet(walk_path)

    log.info(
        "Загружено: X=%s, y=%s, regimes_global=%s (распределение: %s), walk_softprobs=%s",
        X.shape, y.shape, regimes_global.shape,
        regimes_global.value_counts().to_dict(), walk_softprobs.shape,
    )
    return X, y, returns, regimes_global, walk_softprobs


def align_walk_forward_softprobs(walk_df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """Выравнивает walk-forward soft probs с кодировкой monthly_regimes.

    monthly_regimes: 0=медвежий, 1=бычий. В walk_forward_regimes_n2 метки могут
    быть инвертированы. «Бычий» определяется по средней next-month доходности:
    тот режим, где она выше — бычий. При инверсии меняем местами 0 и 1
    (regime_hard, regime_prob_0/1).

    Args:
        walk_df: Сырой DataFrame walk-forward вероятностей.
        returns: Месячные доходности IMOEX.

    Returns:
        DataFrame с теми же колонками, но 0=медвежий, 1=бычий.
    """
    if walk_df.empty:
        return walk_df.copy()

    df = walk_df.copy()
    next_ret = returns.shift(-1).reindex(df.index)

    means: dict[int, float] = {}
    for r in (0, 1):
        mask = df["regime_hard"] == r
        means[r] = float(next_ret[mask].mean(skipna=True)) if mask.sum() else float("nan")

    if np.isnan(means.get(0, np.nan)) or np.isnan(means.get(1, np.nan)):
        log.warning("align_walk_forward_softprobs: NaN в средних %s — swap пропущен", means)
        bullish_in_raw = 1
    else:
        bullish_in_raw = 0 if means[0] > means[1] else 1

    if bullish_in_raw == 1:
        log.info("walk-forward labels уже выровнены (raw[1]=%.4f > raw[0]=%.4f)",
                 means[1], means[0])
    else:
        log.info("walk-forward labels инвертированы (raw[0]=%.4f > raw[1]=%.4f) — swap",
                 means[0], means[1])
        df["regime_hard"] = 1 - df["regime_hard"]
        prob0 = df["regime_prob_0"].copy()
        df["regime_prob_0"] = df["regime_prob_1"]
        df["regime_prob_1"] = prob0

    return df


def _build_pipeline(estimator: Any) -> Pipeline:
    """StandardScaler + классификатор."""
    return Pipeline(steps=[("scaler", StandardScaler()), ("clf", estimator)])


def fit_regime_submodel(
    model_name: str, X_tr_regime: pd.DataFrame, y_tr_regime: pd.Series,
) -> Pipeline:
    """Обучает один режимный подмодель на (X, y) данного режима."""
    estimator = make_estimator(model_name, default_params(model_name))
    pipe = _build_pipeline(estimator)
    pipe.fit(X_tr_regime.values, y_tr_regime.values)
    return pipe


def _inner_cv_auc(model_name: str, X_arr: np.ndarray, y_arr: np.ndarray) -> float:
    """Среднее AUC по time-series CV для одной модели. NaN если все фолды вырождены."""
    n = len(y_arr)
    n_splits = min(REGIME_INNER_CV_N_SPLITS, max(2, n // 8))
    if n_splits < REGIME_INNER_CV_N_SPLITS:
        log.debug(
            "Inner CV downgraded to %d splits (n=%d, max=%d)",
            n_splits, n, REGIME_INNER_CV_N_SPLITS,
        )
    if n_splits < 2 or n < 10:
        return float("nan")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=REGIME_INNER_CV_GAP)
    aucs: list[float] = []
    for tr_idx, va_idx in tscv.split(X_arr):
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue
        try:
            pipe = _build_pipeline(make_estimator(model_name, default_params(model_name)))
            pipe.fit(X_arr[tr_idx], y_tr)
            probs = pipe.predict_proba(X_arr[va_idx])[:, 1]
            aucs.append(float(roc_auc_score(y_va, probs)))
        except Exception as exc:
            log.debug("inner CV: %s упал: %s", model_name, exc)
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def select_best_for_regime(
    X_tr_regime: pd.DataFrame, y_tr_regime: pd.Series,
) -> tuple[str, Pipeline | None]:
    """Выбирает лучшую из 5 моделей по внутренней CV.

    Если |подвыборки| < HMM_MIN_REGIME_SIZE или один класс — возвращает
    ("fallback", None). Если все CV-AUC = NaN — берёт LogisticRegression
    как устойчивый дефолт.

    Args:
        X_tr_regime: Признаки подвыборки режима.
        y_tr_regime: Метки подвыборки режима.

    Returns:
        Кортеж (model_name, pipeline | None). Pipeline уже обучен на полной
        подвыборке, или None при fallback.
    """
    n = len(y_tr_regime)
    if n < HMM_MIN_REGIME_SIZE or y_tr_regime.nunique() < 2:
        return FALLBACK_TAG, None

    X_arr = X_tr_regime.values
    y_arr = y_tr_regime.values.astype(int)

    scores: dict[str, float] = {}
    for name in REGIME_VARIANT_C_MODELS:
        try:
            scores[name] = _inner_cv_auc(name, X_arr, y_arr)
        except Exception as exc:
            log.warning("select_best_for_regime: %s упал: %s", name, exc)
            scores[name] = float("nan")

    valid = {k: v for k, v in scores.items() if not np.isnan(v)}
    if not valid:
        log.warning("select_best_for_regime: все CV вернули NaN (n=%d) — LogReg по умолчанию", n)
        best_name = "LogisticRegression"
    else:
        best_name = max(valid, key=valid.get)

    pipe = fit_regime_submodel(best_name, X_tr_regime, y_tr_regime)
    return best_name, pipe


def _train_regime_models(
    X: pd.DataFrame, y: pd.Series, regimes_global: pd.Series,
    train_idx: pd.DatetimeIndex, model_name: str, baseline_pipe: Pipeline,
) -> tuple[dict[int, Pipeline], int]:
    """Внутренний хелпер: обучает SVM-подмодели по режимам с fallback.

    Возвращает {regime: pipeline} и число fallback-срабатываний (0/1/2).
    """
    models: dict[int, Pipeline] = {}
    fallback_count = 0
    for r in (0, 1):
        mask = regimes_global.loc[train_idx] == r
        X_sub = X.loc[train_idx][mask]
        y_sub = y.loc[train_idx][mask]
        if len(y_sub) < HMM_MIN_REGIME_SIZE or y_sub.nunique() < 2:
            models[r] = baseline_pipe
            fallback_count += 1
        else:
            models[r] = fit_regime_submodel(model_name, X_sub, y_sub)
    return models, fallback_count


def run_variant_a(
    X: pd.DataFrame, y: pd.Series, regimes_global: pd.Series,
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
) -> pd.Series:
    """Вариант A: SVM(RBF) на каждом режиме + fallback на baseline SVM.

    Для теста t берётся регим из regimes_global[t] и применяется
    соответствующая модель (либо baseline при fallback).
    """
    preds: dict[pd.Timestamp, float] = {}
    fallback_total = 0

    for train_idx, test_idx in splits:
        baseline_pipe = fit_regime_submodel(
            REGIME_VARIANT_A_MODEL, X.loc[train_idx], y.loc[train_idx],
        )
        models, fb = _train_regime_models(
            X, y, regimes_global, train_idx, REGIME_VARIANT_A_MODEL, baseline_pipe,
        )
        fallback_total += fb

        for t in test_idx:
            r_t = int(regimes_global.loc[t]) if pd.notna(regimes_global.loc[t]) else -1
            pipe = models.get(r_t, baseline_pipe)
            preds[t] = float(pipe.predict_proba(X.loc[[t]].values)[0, 1])

    log.info("Variant A: %d fallback из %d (фолдов × режимов)",
             fallback_total, len(splits) * 2)
    return pd.Series(preds, name="VariantA").sort_index()


def run_variant_b(
    X: pd.DataFrame, y: pd.Series, regimes_global: pd.Series,
    walk_softprobs_aligned: pd.DataFrame,
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
) -> pd.Series:
    """Вариант B: SVM на каждый режим + мягкое смешивание walk-forward вер-ями.

    pred[t] = p_walk[t,0] * model_0(X[t]) + p_walk[t,1] * model_1(X[t]).
    Если walk-forward soft probs нет/NaN для t — равномерное смешивание (0.5/0.5).
    """
    preds: dict[pd.Timestamp, float] = {}
    fallback_total = 0

    for train_idx, test_idx in splits:
        baseline_pipe = fit_regime_submodel(
            REGIME_VARIANT_A_MODEL, X.loc[train_idx], y.loc[train_idx],
        )
        models, fb = _train_regime_models(
            X, y, regimes_global, train_idx, REGIME_VARIANT_A_MODEL, baseline_pipe,
        )
        fallback_total += fb

        for t in test_idx:
            x_row = X.loc[[t]].values
            p0_model = float(models[0].predict_proba(x_row)[0, 1])
            p1_model = float(models[1].predict_proba(x_row)[0, 1])

            if t in walk_softprobs_aligned.index:
                p0 = walk_softprobs_aligned.loc[t, "regime_prob_0"]
                p1 = walk_softprobs_aligned.loc[t, "regime_prob_1"]
                if pd.isna(p0) or pd.isna(p1):
                    p0, p1 = 0.5, 0.5
            else:
                log.debug(
                    "Variant B: no walk-forward soft probs for %s — using (0.5, 0.5)", t,
                )
                p0, p1 = 0.5, 0.5
            preds[t] = float(p0) * p0_model + float(p1) * p1_model

    log.info("Variant B: %d fallback из %d (фолдов × режимов)",
             fallback_total, len(splits) * 2)
    return pd.Series(preds, name="VariantB").sort_index()


def run_variant_c(
    X: pd.DataFrame, y: pd.Series, regimes_global: pd.Series,
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
) -> tuple[pd.Series, pd.DataFrame]:
    """Вариант C: best-of-5 классических моделей на режим (inner CV).

    Returns:
        Кортеж (preds, choices). choices: DataFrame[fold_test_date →
        regime_0_best, regime_1_best].
    """
    preds: dict[pd.Timestamp, float] = {}
    choice_rows: list[dict[str, Any]] = []
    fallback_total = 0

    for train_idx, test_idx in splits:
        baseline_pipe = fit_regime_submodel(
            REGIME_VARIANT_A_MODEL, X.loc[train_idx], y.loc[train_idx],
        )
        models: dict[int, Pipeline] = {}
        names: dict[int, str] = {}
        for r in (0, 1):
            mask = regimes_global.loc[train_idx] == r
            X_sub = X.loc[train_idx][mask]
            y_sub = y.loc[train_idx][mask]
            name, pipe = select_best_for_regime(X_sub, y_sub)
            if name == FALLBACK_TAG or pipe is None:
                models[r] = baseline_pipe
                names[r] = FALLBACK_TAG
                fallback_total += 1
            else:
                models[r] = pipe
                names[r] = name

        for t in test_idx:
            r_t = int(regimes_global.loc[t]) if pd.notna(regimes_global.loc[t]) else -1
            pipe = models.get(r_t, baseline_pipe)
            preds[t] = float(pipe.predict_proba(X.loc[[t]].values)[0, 1])

        choice_rows.append({
            "fold_test_date": test_idx[0],
            "regime_0_best": names.get(0, FALLBACK_TAG),
            "regime_1_best": names.get(1, FALLBACK_TAG),
        })

    log.info("Variant C: %d fallback из %d (фолдов × режимов)",
             fallback_total, len(splits) * 2)
    pred_series = pd.Series(preds, name="VariantC").sort_index()
    choices = pd.DataFrame(choice_rows).set_index("fold_test_date").sort_index()
    return pred_series, choices


def _quick_auc(pred: pd.Series, y: pd.Series) -> float:
    """Быстрый AUC для логирования."""
    aligned_y = y.reindex(pred.index).dropna().astype(int)
    aligned_p = pred.reindex(aligned_y.index).dropna()
    if aligned_p.empty or aligned_y.nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(aligned_y, aligned_p))
    except Exception:
        return float("nan")


def main() -> None:
    """Полный pipeline режимных моделей: A, B, C → regime_predictions.parquet."""
    X, y, returns, reg_global, walk_softprobs = load_data()
    walk_aligned = align_walk_forward_softprobs(walk_softprobs, returns)

    splits = walk_forward_splits(X, y)
    log.info("walk_forward_splits: %d фолдов", len(splits))

    log.info("=== Variant A (SVM по режимам + fallback) ===")
    pred_a = run_variant_a(X, y, reg_global, splits)
    log.info("=== Variant B (вероятностный ансамбль SVM) ===")
    pred_b = run_variant_b(X, y, reg_global, walk_aligned, splits)
    log.info("=== Variant C (best-of-5 inner CV) ===")
    pred_c, c_choices = run_variant_c(X, y, reg_global, splits)

    out_df = pd.concat([pred_a, pred_b, pred_c], axis=1)
    out_df.columns = ["VariantA", "VariantB", "VariantC"]
    out_path = os.path.join(DATA_PROCESSED, "regime_predictions.parquet")
    out_df.to_parquet(out_path)
    log.info("Сохранено: %s, shape=%s", out_path, out_df.shape)

    save_table(c_choices, "T20_best_per_regime", subfolder=SUBFOLDER)
    log.info("Сохранено: tables/%s/T20_best_per_regime.csv (rows=%d)",
             SUBFOLDER, len(c_choices))

    if not c_choices.empty:
        for r_col in ("regime_0_best", "regime_1_best"):
            counts = c_choices[r_col].value_counts().to_dict()
            log.info("Variant C: распределение по %s: %s", r_col, counts)

    log.info("Variant A AUC (test): %.4f", _quick_auc(pred_a, y))
    log.info("Variant B AUC (test): %.4f", _quick_auc(pred_b, y))
    log.info("Variant C AUC (test): %.4f", _quick_auc(pred_c, y))
    log.info("Готово. Метрики и графики — в скрипте 09b_regime_metrics_figures.py.")


if __name__ == "__main__":
    main()
