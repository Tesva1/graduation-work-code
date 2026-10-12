"""Шаг 10: SHAP-анализ лучшей режимной модели Variant A (walk-forward).

Цель — интерпретировать режимную модель Variant A: SVM(RBF), обученный
отдельно для каждого рыночного режима. Так как SVM не является деревом,
используется shap.KernelExplainer.

Методология (СТРОГИЙ walk-forward):
* Для каждого тестового месяца t из regime_predictions.parquet:
    1. Определяется cutoff: train_end_period = t.period - (GAP_MONTHS + 1).
    2. Per-regime SVM обучается ТОЛЬКО на месячных строках с периодом
       ≤ train_end_period. Тестовая строка t в обучении не участвует.
    3. KernelExplainer берёт фон из train-строк того же режима
       (детерминированный sample с фиксированным seed).
    4. SHAP-значения вычисляются для X[t] моделью режима t.
* Этим вариант соответствует архитектуре Variant A в src/09_regime_models.py
  (Hamilton 1989 «soft look-ahead» для меток HMM, но строгий no-look-ahead
  в обучении ML и SHAP).
* Сохраняются: shap_values_variant_a.parquet, T23, fig_56--fig_68.

Запуск:
    python src/10_shap_analysis.py

Переменные окружения для ускоренной отладки:
    SHAP_MAX_EVAL_ROWS=20      ограничить число тестовых строк
    SHAP_BACKGROUND_SIZE=12    уменьшить фон KernelExplainer
    SHAP_NSAMPLES=100          уменьшить число samples KernelExplainer
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Callable
from typing import Any

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (  # noqa: E402
    DATA_PROCESSED,
    FIGURES,
    GAP_MONTHS,
    HMM_MIN_REGIME_SIZE,
    RANDOM_STATE,
    SHAP_BACKGROUND_SIZE,
    SHAP_NSAMPLES,
    TABLES,
    TEST_END,
    TEST_START,
)
from src.feature_utils import FEATURE_LABELS  # noqa: E402
from src.utils import setup_plot_style  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_10")

SUBFOLDER = "step_10"
SHAP_VALUES_FILE = os.path.join(DATA_PROCESSED, "shap_values_variant_a.parquet")
SHAP_META_FILE = os.path.join(DATA_PROCESSED, "shap_values_variant_a_meta.parquet")
T23_FILE = os.path.join(TABLES, "T23_shap_top_features.csv")

REGIME_LABELS = {
    "global": "Глобально",
    0: "Медвежий режим",
    1: "Бычий режим",
    2: "Режим 2",
    3: "Режим 3",
}

FIG_NAMES = {
    56: "fig_56_global_shap_bar",
    57: "fig_57_global_shap_beeswarm",
    58: "fig_58_shap_by_regime_three_panel",
    59: "fig_59_shap_regime_heatmap",
    60: "fig_60_shap_beeswarm_bear",
    61: "fig_61_shap_beeswarm_bull",
    62: "fig_62_rolling_shap_top5",
    63: "fig_63_rolling_shap_with_regime_background",
    64: "fig_64_dependence_urals",
    65: "fig_65_dependence_volatility",
    66: "fig_66_dependence_key_rate",
    67: "fig_67_waterfall_covid_2020",
    68: "fig_68_waterfall_growth_2021",
}


def feature_label(feature: str) -> str:
    """Возвращает русскую подпись признака."""
    return FEATURE_LABELS.get(feature, feature)


def _ensure_dirs() -> None:
    os.makedirs(os.path.join(FIGURES, SUBFOLDER), exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(DATA_PROCESSED, exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> None:
    """Сохраняет фигуру одновременно в PNG и PDF."""
    _ensure_dirs()
    base = os.path.join(FIGURES, SUBFOLDER, name)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex]:
    """Загружает признаки, target, режимы и индекс тестового периода."""
    X = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_features.parquet"))
    y_df = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_target.parquet"))
    regimes_df = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_regimes.parquet"))

    y = y_df["y_next"].dropna().astype(int)
    X = X.reindex(y.index).fillna(0)
    regimes = regimes_df["regime"].reindex(y.index).astype(int)

    pred_path = os.path.join(DATA_PROCESSED, "regime_predictions.parquet")
    if os.path.exists(pred_path):
        pred = pd.read_parquet(pred_path)
        test_index = pred.index.intersection(X.index)
    else:
        periods = X.index.to_period("M")
        test_periods = pd.period_range(TEST_START, TEST_END, freq="M")
        test_index = X.index[periods.isin(test_periods)]

    max_rows = os.getenv("SHAP_MAX_EVAL_ROWS")
    if max_rows:
        test_index = test_index[: int(max_rows)]
        log.warning("SHAP_MAX_EVAL_ROWS=%s: тестовый период ограничен", max_rows)

    log.info(
        "Загружено: X=%s, y=%s, regimes=%s, test=%d (%s..%s)",
        X.shape,
        y.shape,
        regimes.value_counts().to_dict(),
        len(test_index),
        test_index.min().date() if len(test_index) else None,
        test_index.max().date() if len(test_index) else None,
    )
    return X, y, regimes, pd.DatetimeIndex(test_index)


def _make_svm_pipeline() -> Pipeline:
    """Создаёт SVM(RBF) Variant A с дефолтами step 08/09: C=1, gamma=scale."""
    estimator = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def fit_per_regime_svms(
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
) -> dict[int, Pipeline]:
    """Legacy: обучает отдельную SVM для каждого режима на ВСЕХ monthly данных.

    ВНИМАНИЕ: эта функция — full-sample fit. Для дипломного pipeline она
    оставлена ТОЛЬКО для unit-тестов и быстрых smoke-проверок. Production-путь
    использует `fit_per_regime_svms_for_train` с walk-forward отсечением
    тестовых данных.

    Если в режиме только один класс, используется fallback: SVM на полной выборке.
    """
    X_aligned = X.reindex(y.index).fillna(0)
    regimes_aligned = regimes.reindex(y.index).astype(int)

    full_model = _make_svm_pipeline()
    full_model.fit(X_aligned.values, y.values.astype(int))

    models: dict[int, Pipeline] = {}
    for regime in sorted(regimes_aligned.dropna().unique()):
        r = int(regime)
        mask = regimes_aligned == r
        X_r = X_aligned.loc[mask]
        y_r = y.loc[X_r.index]
        if len(y_r) < 2 or y_r.nunique() < 2:
            log.warning(
                "Режим %s: n=%d, классов=%d; fallback на full SVM",
                r,
                len(y_r),
                y_r.nunique(),
            )
            models[r] = full_model
            continue
        model = _make_svm_pipeline()
        model.fit(X_r.values, y_r.values.astype(int))
        models[r] = model
        log.info("SVM режима %s обучена: n=%d, balance=%s", r, len(y_r), y_r.value_counts().to_dict())
    return models


def fit_per_regime_svms_for_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    regimes_train: pd.Series,
    fallback_model: Pipeline | None = None,
) -> dict[int, Pipeline]:
    """Walk-forward версия: обучает per-regime SVM ТОЛЬКО на train-данных.

    Args:
        X_train: Признаки train-периода (без gap и без тестовой строки).
        y_train: Целевая на тех же индексах.
        regimes_train: Метки режимов на тех же индексах.
        fallback_model: Готовая модель на полном train (если в режиме < HMM_MIN_REGIME_SIZE
            строк или один класс — используется fallback). Если None — обучается здесь.

    Returns:
        Словарь {regime: Pipeline}. Все модели обучены без участия test-данных.
    """
    X_aligned = X_train.fillna(0)
    regimes_aligned = regimes_train.astype(int)

    if fallback_model is None:
        fallback_model = _make_svm_pipeline()
        fallback_model.fit(X_aligned.values, y_train.values.astype(int))

    models: dict[int, Pipeline] = {}
    for regime in sorted(regimes_aligned.dropna().unique()):
        r = int(regime)
        mask = regimes_aligned == r
        X_r = X_aligned.loc[mask]
        y_r = y_train.loc[X_r.index]
        if len(y_r) < HMM_MIN_REGIME_SIZE or y_r.nunique() < 2:
            log.debug(
                "Режим %s: n=%d, классов=%d → fallback на baseline SVM",
                r, len(y_r), y_r.nunique(),
            )
            models[r] = fallback_model
            continue
        model = _make_svm_pipeline()
        model.fit(X_r.values, y_r.values.astype(int))
        models[r] = model
    return models


def _background_for_regime(
    X: pd.DataFrame,
    regimes: pd.Series,
    regime: int,
    background_size: int,
) -> pd.DataFrame:
    """Детерминированно выбирает фон KernelExplainer для режима.

    Для walk-forward вызовов передаётся уже отфильтрованный X_train —
    фон берётся только из train-периода соответствующего режима.
    """
    X_r = X.loc[regimes == regime]
    if X_r.empty:
        X_r = X
    if len(X_r) > background_size:
        X_r = X_r.sample(background_size, random_state=RANDOM_STATE).sort_index()
    return X_r.fillna(0)


def _train_idx_for_test_month(
    monthly_index: pd.DatetimeIndex,
    test_month: pd.Timestamp,
    gap: int,
) -> pd.DatetimeIndex:
    """Возвращает train-индексы, согласованные с walk_forward_splits шага 8.

    train_end_period = period(test_month) - (gap + 1).
    train_idx = все строки monthly_index с периодом ≤ train_end_period.
    """
    test_period = pd.Period(test_month, freq="M")
    train_end_period = test_period - (gap + 1)
    monthly_periods = monthly_index.to_period("M")
    return monthly_index[monthly_periods <= train_end_period]


def _coerce_shap_matrix(raw: Any, n_rows: int, n_features: int) -> np.ndarray:
    """Нормализует разные форматы shap_values в матрицу (n_rows, n_features)."""
    if isinstance(raw, list):
        raw = raw[-1]
    arr = np.asarray(raw)
    if arr.ndim == 3:
        # SHAP >= 0.45 иногда возвращает (n, features, outputs).
        if arr.shape[0] == n_rows and arr.shape[1] == n_features:
            arr = arr[:, :, -1]
        elif arr.shape[0] == 2:
            arr = arr[-1]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape != (n_rows, n_features):
        raise ValueError(f"Unexpected SHAP shape {arr.shape}, expected {(n_rows, n_features)}")
    return arr.astype(float)


def compute_shap_test_period(
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    *,
    models: dict[int, Pipeline] | None = None,
    test_index: pd.DatetimeIndex | None = None,
    background_size: int = SHAP_BACKGROUND_SIZE,
    nsamples: int = SHAP_NSAMPLES,
    explainer_factory: Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray], Any] | None = None,
) -> pd.DataFrame:
    """LEGACY: считает SHAP values Variant A на тестовом периоде с full-fit моделями.

    Используется только в unit-тестах, где FakeExplainer обходит дорогой
    KernelExplainer. Production-путь — `compute_shap_test_period_walk_forward`.
    """
    X_clean = X.reindex(y.index).fillna(0)
    regimes_aligned = regimes.reindex(y.index).astype(int)
    if models is None:
        models = fit_per_regime_svms(X_clean, y, regimes_aligned)

    if test_index is None:
        periods = X_clean.index.to_period("M")
        test_periods = pd.period_range(TEST_START, TEST_END, freq="M")
        test_index = X_clean.index[periods.isin(test_periods)]
    test_index = pd.DatetimeIndex(test_index).intersection(X_clean.index)

    if explainer_factory is None:
        import shap

        explainer_factory = shap.KernelExplainer

    out = pd.DataFrame(index=test_index, columns=X_clean.columns, dtype=float)
    for regime in sorted(regimes_aligned.loc[test_index].dropna().unique()):
        r = int(regime)
        if r not in models:
            raise KeyError(f"Нет SVM-модели для режима {r}")

        idx_r = test_index[regimes_aligned.loc[test_index] == r]
        if len(idx_r) == 0:
            continue
        model = models[r]
        background = _background_for_regime(X_clean, regimes_aligned, r, background_size)
        X_eval = X_clean.loc[idx_r]

        def model_fn(values: np.ndarray, model: Pipeline = model) -> np.ndarray:
            values = np.asarray(values)
            return model.predict_proba(values)[:, 1]

        log.info(
            "Kernel SHAP: regime=%s, test_rows=%d, background=%d, features=%d, nsamples=%d",
            r,
            len(X_eval),
            len(background),
            X_eval.shape[1],
            nsamples,
        )
        explainer = explainer_factory(model_fn, background.values)
        np.random.seed(RANDOM_STATE)  # детерминизм пермутаций KernelExplainer
        raw_values = explainer.shap_values(X_eval.values, nsamples=nsamples)
        values = _coerce_shap_matrix(raw_values, len(X_eval), X_eval.shape[1])
        out.loc[idx_r, :] = values

    if out.isna().any().any():
        missing = int(out.isna().sum().sum())
        raise ValueError(f"SHAP содержит NaN после расчёта: {missing}")
    return out


def compute_shap_test_period_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    *,
    test_index: pd.DatetimeIndex,
    gap: int = GAP_MONTHS,
    background_size: int = SHAP_BACKGROUND_SIZE,
    nsamples: int = SHAP_NSAMPLES,
    explainer_factory: Callable[[Callable[[np.ndarray], np.ndarray], np.ndarray], Any] | None = None,
) -> pd.DataFrame:
    """Walk-forward SHAP Variant A — без look-ahead.

    Для каждого тестового месяца t:
        1. Train-индексы: monthly_periods ≤ period(t) - (gap+1).
        2. Per-regime SVM обучается только на train-индексах того же режима.
        3. KernelExplainer берёт фон из train-строк режима t.
        4. SHAP вычисляется для X[t] моделью режима регима в t.

    Архитектура совпадает с walk_forward_splits шага 8 (idx test ровно один
    месяц, gap=1 → между train и test ровно один пропущенный месяц).

    Args:
        X: Monthly features (вся история, включая будущее).
        y: Target y_next.
        regimes: Метки режимов (используются как фильтр строк, не как фича).
        test_index: Индекс тестовых месяцев — как правило индекс
            regime_predictions.parquet.
        gap: Зазор в месяцах между train и test (по умолчанию config.GAP_MONTHS).
        background_size: Размер фона KernelExplainer.
        nsamples: nsamples KernelExplainer.shap_values.
        explainer_factory: Инъекция для тестов.

    Returns:
        DataFrame SHAP values; индекс — test_index, колонки — X.columns.
    """
    X_clean = X.reindex(y.index).fillna(0)
    regimes_aligned = regimes.reindex(y.index).astype(int)
    test_index = pd.DatetimeIndex(test_index).intersection(X_clean.index).sort_values()

    if explainer_factory is None:
        import shap

        explainer_factory = shap.KernelExplainer

    out = pd.DataFrame(index=test_index, columns=X_clean.columns, dtype=float)
    monthly_index = X_clean.index

    for t in test_index:
        train_idx = _train_idx_for_test_month(monthly_index, t, gap)
        if len(train_idx) < HMM_MIN_REGIME_SIZE:
            log.warning(
                "compute_shap_walk_forward: для %s train слишком мал (n=%d); строка пропущена",
                t.date(), len(train_idx),
            )
            continue

        X_train = X_clean.loc[train_idx]
        y_train = y.loc[train_idx]
        reg_train = regimes_aligned.loc[train_idx]

        # Fallback baseline на всём train того же фолда — не зависит от режима
        fallback = _make_svm_pipeline()
        fallback.fit(X_train.values, y_train.values.astype(int))

        models = fit_per_regime_svms_for_train(
            X_train, y_train, reg_train, fallback_model=fallback,
        )

        r_t = int(regimes_aligned.loc[t]) if pd.notna(regimes_aligned.loc[t]) else -1
        model = models.get(r_t, fallback)

        background = _background_for_regime(
            X_train, reg_train, r_t, background_size,
        )
        if background.empty:
            background = X_train.tail(min(len(X_train), background_size))

        def model_fn(values: np.ndarray, model: Pipeline = model) -> np.ndarray:
            values = np.asarray(values)
            return model.predict_proba(values)[:, 1]

        explainer = explainer_factory(model_fn, background.values)
        np.random.seed(RANDOM_STATE)  # детерминизм пермутаций KernelExplainer
        raw_values = explainer.shap_values(X_clean.loc[[t]].values, nsamples=nsamples)
        values = _coerce_shap_matrix(raw_values, 1, X_clean.shape[1])
        out.loc[t, :] = values[0]

    if out.isna().any().any():
        n_dropped = int(out.isna().any(axis=1).sum())
        out = out.dropna(how="any")
        log.warning("compute_shap_walk_forward: %d тестовых строк пропущено из-за слишком короткого train", n_dropped)
    return out


def save_shap_outputs(shap_values: pd.DataFrame, regimes: pd.Series) -> None:
    """Сохраняет SHAP values и metadata parquet."""
    _ensure_dirs()
    shap_values.to_parquet(SHAP_VALUES_FILE)
    meta = pd.DataFrame({"regime": regimes.reindex(shap_values.index).astype(int)})
    meta.to_parquet(SHAP_META_FILE)
    log.info("Сохранено: %s, shape=%s", SHAP_VALUES_FILE, shap_values.shape)
    log.info("Сохранено: %s, shape=%s", SHAP_META_FILE, meta.shape)


def _importance(shap_values: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP importance."""
    if shap_values.empty:
        return pd.Series(dtype=float)
    return shap_values.abs().mean(axis=0).sort_values(ascending=False)


def compute_t23(
    shap_values: pd.DataFrame,
    regimes: pd.Series,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Формирует T23: топ-N факторов глобально и по каждому режиму."""
    regimes_aligned = regimes.reindex(shap_values.index).astype(int)
    rows: list[dict[str, object]] = []

    blocks: list[tuple[str, pd.DataFrame]] = [("global", shap_values)]
    for regime in sorted(regimes_aligned.dropna().unique()):
        r = int(regime)
        blocks.append((f"regime_{r}", shap_values.loc[regimes_aligned == r]))

    for block, values in blocks:
        imp = _importance(values).head(top_n)
        for rank, (feature, value) in enumerate(imp.items(), start=1):
            rows.append(
                {
                    "block": block,
                    "rank": rank,
                    "feature": feature,
                    "feature_ru": feature_label(feature),
                    "mean_abs_shap": float(value),
                    "n_obs": int(len(values)),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(
            columns=["rank", "feature", "feature_ru", "mean_abs_shap", "n_obs"],
            index=pd.MultiIndex.from_arrays([[], []], names=["block", "rank"]),
        )
    table = table.set_index(["block", "rank"]).sort_index()
    table["rank"] = table.index.get_level_values("rank")
    return table[["rank", "feature", "feature_ru", "mean_abs_shap", "n_obs"]]


def save_t23(table: pd.DataFrame) -> None:
    _ensure_dirs()
    if isinstance(table.index, pd.MultiIndex) and "block" in table.index.names:
        csv_table = table.reset_index(level="block").reset_index(drop=True)
    else:
        csv_table = table
    csv_table.to_csv(T23_FILE, index=False, encoding="utf-8-sig")
    log.info("Сохранено: %s, shape=%s", T23_FILE, table.shape)


def rolling_shap_importance(
    shap_values: pd.DataFrame,
    *,
    window: int = 12,
    top_k: int = 5,
) -> pd.DataFrame:
    """Возвращает top-k mean|SHAP| в скользящем окне."""
    rows: list[dict[str, object]] = []
    if len(shap_values) < window:
        return pd.DataFrame(columns=["date", "feature", "importance", "rank"])

    for end in range(window, len(shap_values) + 1):
        window_values = shap_values.iloc[end - window : end]
        date = shap_values.index[end - 1]
        imp = _importance(window_values).head(top_k)
        for rank, (feature, value) in enumerate(imp.items(), start=1):
            rows.append(
                {
                    "date": date,
                    "feature": feature,
                    "importance": float(value),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def _top_features(shap_values: pd.DataFrame, n: int = 10) -> list[str]:
    return list(_importance(shap_values).head(n).index)


def _plot_importance_bar(
    ax: plt.Axes,
    shap_values: pd.DataFrame,
    title: str,
    *,
    top_n: int = 10,
) -> None:
    imp = _importance(shap_values).head(top_n).sort_values()
    labels = [feature_label(f) for f in imp.index]
    ax.barh(labels, imp.values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Среднее |SHAP|")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)


def fig_56_global_bar(shap_values: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    _plot_importance_bar(ax, shap_values, "Глобальная важность факторов Variant A", top_n=15)
    save_figure(fig, FIG_NAMES[56])


def _save_summary_plot(
    shap_values: pd.DataFrame,
    X_test: pd.DataFrame,
    filename: str,
    title: str,
    *,
    max_display: int = 15,
) -> None:
    import shap

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values.values,
        features=X_test.loc[shap_values.index],
        feature_names=[feature_label(c) for c in shap_values.columns],
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(title, y=1.02)
    save_figure(fig, filename)


def fig_57_global_beeswarm(shap_values: pd.DataFrame, X_test: pd.DataFrame) -> None:
    _save_summary_plot(
        shap_values,
        X_test,
        FIG_NAMES[57],
        "Глобальное распределение SHAP-значений Variant A",
    )


def fig_58_three_panel(shap_values: pd.DataFrame, regimes: pd.Series) -> None:
    regimes_aligned = regimes.reindex(shap_values.index).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharex=False)
    blocks = [("Глобально", shap_values)]
    for r in sorted(regimes_aligned.unique())[:2]:
        blocks.append((REGIME_LABELS.get(int(r), f"Режим {r}"), shap_values.loc[regimes_aligned == r]))
    while len(blocks) < 3:
        blocks.append(("Нет данных", shap_values.iloc[0:0]))

    for ax, (title, values) in zip(axes, blocks, strict=False):
        if values.empty:
            ax.axis("off")
            ax.set_title(title)
        else:
            _plot_importance_bar(ax, values, title, top_n=10)
    fig.suptitle("SHAP-важность факторов: глобально и по режимам", y=1.02)
    save_figure(fig, FIG_NAMES[58])


def fig_59_heatmap(shap_values: pd.DataFrame, regimes: pd.Series) -> None:
    regimes_aligned = regimes.reindex(shap_values.index).astype(int)
    cols: dict[str, pd.Series] = {"Глобально": _importance(shap_values)}
    for r in sorted(regimes_aligned.unique()):
        cols[REGIME_LABELS.get(int(r), f"Режим {r}")] = _importance(
            shap_values.loc[regimes_aligned == r]
        )
    heat = pd.DataFrame(cols).fillna(0)
    top = heat["Глобально"].sort_values(ascending=False).head(15).index
    heat = heat.loc[top]
    heat.index = [feature_label(f) for f in heat.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Mean |SHAP|"})
    ax.set_title("Карта важности факторов по режимам")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, FIG_NAMES[59])


def fig_60_61_regime_beeswarms(
    shap_values: pd.DataFrame,
    X_test: pd.DataFrame,
    regimes: pd.Series,
) -> None:
    regimes_aligned = regimes.reindex(shap_values.index).astype(int)
    ordered = sorted(regimes_aligned.unique())
    targets = [(60, ordered[0] if len(ordered) > 0 else None), (61, ordered[1] if len(ordered) > 1 else None)]
    for fig_no, regime in targets:
        filename = FIG_NAMES[fig_no]
        if regime is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axis("off")
            ax.set_title("Нет данных для режима")
            save_figure(fig, filename)
            continue
        mask = regimes_aligned == regime
        values = shap_values.loc[mask]
        if len(values) < 2:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axis("off")
            ax.set_title(f"{REGIME_LABELS.get(int(regime), f'Режим {regime}')}: недостаточно данных")
            save_figure(fig, filename)
            continue
        _save_summary_plot(
            values,
            X_test,
            filename,
            f"SHAP-распределение: {REGIME_LABELS.get(int(regime), f'Режим {regime}')}",
            max_display=12,
        )


def fig_62_rolling_top5(rolling: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    if rolling.empty:
        ax.text(0.5, 0.5, "Недостаточно данных для 12-месячного окна", ha="center", va="center")
        ax.axis("off")
    else:
        top_features = (
            rolling.groupby("feature")["importance"].mean().sort_values(ascending=False).head(5).index
        )
        for feature in top_features:
            part = rolling[rolling["feature"] == feature].sort_values("date")
            ax.plot(part["date"], part["importance"], marker="o", label=feature_label(feature))
        ax.set_title("Скользящая SHAP-важность топ-5 факторов (12 месяцев)")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Mean |SHAP|")
        ax.legend(loc="best")
    save_figure(fig, FIG_NAMES[62])


def _regime_spans(regimes: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Группирует последовательные месяцы одного режима в интервалы."""
    if regimes.empty:
        return []
    regimes = regimes.sort_index().astype(int)
    spans: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start = regimes.index[0]
    prev = regimes.index[0]
    current = int(regimes.iloc[0])
    for date, value in regimes.iloc[1:].items():
        value = int(value)
        if value != current:
            spans.append((start, prev, current))
            start = date
            current = value
        prev = date
    spans.append((start, prev, current))
    return spans


def fig_63_rolling_with_regime_bg(rolling: pd.DataFrame, regimes: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {0: "#E45756", 1: "#54A24B", 2: "#F2CF5B", 3: "#72B7B2"}
    regimes_sorted = regimes.sort_index().astype(int)
    for start, end, r in _regime_spans(regimes_sorted):
        ax.axvspan(start, end + pd.offsets.MonthEnd(1), color=colors.get(r, "0.8"), alpha=0.12)

    if rolling.empty:
        ax.text(0.5, 0.5, "Недостаточно данных для 12-месячного окна", ha="center", va="center")
    else:
        top_features = (
            rolling.groupby("feature")["importance"].mean().sort_values(ascending=False).head(3).index
        )
        for feature in top_features:
            part = rolling[rolling["feature"] == feature].sort_values("date")
            ax.plot(part["date"], part["importance"], marker="o", label=feature_label(feature))
        ax.legend(loc="best")
    ax.set_title("Скользящая SHAP-важность с фоном рыночных режимов")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Mean |SHAP|")
    save_figure(fig, FIG_NAMES[63])


def _dependence_feature_candidates(kind: str, columns: list[str]) -> list[str]:
    if kind == "urals":
        return ["urals_ret_1m", "urals_ret_3m", "urals_brent_spread"]
    if kind == "vol":
        return ["imoex_vol_20d", "imoex_vol_60d", "imoex_garch_vol", "imoex_vol_ratio_20_60"]
    if kind == "key_rate":
        return ["key_rate_level", "key_rate_change"]
    return columns


def _pick_existing(candidates: list[str], columns: pd.Index) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _dependence_plot(
    shap_values: pd.DataFrame,
    X_test: pd.DataFrame,
    regimes: pd.Series,
    feature: str | None,
    fig_no: int,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    if feature is None or feature not in shap_values.columns:
        ax.text(0.5, 0.5, "Признак отсутствует в данных", ha="center", va="center")
        ax.axis("off")
        save_figure(fig, FIG_NAMES[fig_no])
        return

    regimes_aligned = regimes.reindex(shap_values.index).astype(int)
    plot_df = pd.DataFrame(
        {
            "x": X_test.loc[shap_values.index, feature],
            "shap": shap_values[feature],
            "regime": regimes_aligned,
        }
    )
    sns.scatterplot(data=plot_df, x="x", y="shap", hue="regime", palette="colorblind", ax=ax, s=55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(feature_label(feature))
    ax.set_ylabel("SHAP-значение")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, [REGIME_LABELS.get(int(float(v)), f"Режим {v}") for v in labels], title="")
    save_figure(fig, FIG_NAMES[fig_no])


def fig_64_66_dependence(shap_values: pd.DataFrame, X_test: pd.DataFrame, regimes: pd.Series) -> None:
    columns = list(shap_values.columns)
    _dependence_plot(
        shap_values,
        X_test,
        regimes,
        _pick_existing(_dependence_feature_candidates("urals", columns), shap_values.columns),
        64,
        "SHAP dependence: нефтяной фактор Urals",
    )
    _dependence_plot(
        shap_values,
        X_test,
        regimes,
        _pick_existing(_dependence_feature_candidates("vol", columns), shap_values.columns),
        65,
        "SHAP dependence: волатильность IMOEX",
    )
    _dependence_plot(
        shap_values,
        X_test,
        regimes,
        _pick_existing(_dependence_feature_candidates("key_rate", columns), shap_values.columns),
        66,
        "SHAP dependence: ключевая ставка ЦБ",
    )


def _nearest_index(index: pd.DatetimeIndex, target: str) -> pd.Timestamp | None:
    if len(index) == 0:
        return None
    target_ts = pd.Timestamp(target)
    diffs = np.abs((index - target_ts).days)
    return index[int(np.argmin(diffs))]


def _waterfall_bar(
    shap_values: pd.DataFrame,
    X_test: pd.DataFrame,
    date: pd.Timestamp | None,
    fig_no: int,
    title: str,
    *,
    top_n: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    if date is None or date not in shap_values.index:
        ax.text(0.5, 0.5, "Дата отсутствует в SHAP-выборке", ha="center", va="center")
        ax.axis("off")
        save_figure(fig, FIG_NAMES[fig_no])
        return

    row = shap_values.loc[date].sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)
    row = row.sort_values()
    colors = ["#E45756" if v < 0 else "#54A24B" for v in row.values]
    ax.barh([feature_label(f) for f in row.index], row.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{title}: {date.strftime('%Y-%m')}")
    ax.set_xlabel("SHAP-значение")
    ax.set_ylabel("")
    # Кратко показываем факт значения признака в подписи оси, не перегружая бар.
    for i, (feature, value) in enumerate(row.items()):
        raw_value = X_test.loc[date, feature]
        ax.text(
            value,
            i,
            f"  x={raw_value:.3g}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )
    save_figure(fig, FIG_NAMES[fig_no])


def fig_67_68_waterfalls(shap_values: pd.DataFrame, X_test: pd.DataFrame) -> None:
    covid = _nearest_index(shap_values.index, "2020-03-31")
    growth = _nearest_index(shap_values.index, "2021-03-31")
    _waterfall_bar(shap_values, X_test, covid, 67, "Локальное объяснение прогноза (COVID-19)")
    _waterfall_bar(shap_values, X_test, growth, 68, "Локальное объяснение прогноза (рост рынка)")


def generate_all_figures(
    shap_values: pd.DataFrame,
    X: pd.DataFrame,
    regimes: pd.Series,
) -> None:
    """Создаёт fig_56--fig_68 в figures/step_10."""
    X_test = X.loc[shap_values.index].fillna(0)
    fig_56_global_bar(shap_values)
    fig_57_global_beeswarm(shap_values, X_test)
    fig_58_three_panel(shap_values, regimes)
    fig_59_heatmap(shap_values, regimes)
    fig_60_61_regime_beeswarms(shap_values, X_test, regimes)
    rolling = rolling_shap_importance(shap_values, window=12, top_k=5)
    fig_62_rolling_top5(rolling)
    fig_63_rolling_with_regime_bg(rolling, regimes.reindex(shap_values.index))
    fig_64_66_dependence(shap_values, X_test, regimes)
    fig_67_68_waterfalls(shap_values, X_test)
    log.info("Сохранены fig_56--fig_68 в %s", os.path.join(FIGURES, SUBFOLDER))


def validate_outputs(shap_values: pd.DataFrame, t23: pd.DataFrame, regimes: pd.Series) -> None:
    """Быстрая валидация результатов SHAP-анализа."""
    if shap_values.empty:
        raise ValueError("SHAP values пустые")
    if not np.isfinite(shap_values.values).all():
        raise ValueError("SHAP values содержат NaN/inf")
    if shap_values.shape[1] != 25:
        log.warning("Ожидалось 25 признаков, получено %d", shap_values.shape[1])
    if t23.empty:
        raise ValueError("T23 пустая")
    counts = regimes.reindex(shap_values.index).value_counts().to_dict()
    if len(counts) < 2:
        log.warning("В тестовом SHAP периоде обнаружен только один режим: %s", counts)
    log.info(
        "Валидация: shap=%s, regimes_test=%s, T23=%s",
        shap_values.shape,
        counts,
        t23.shape,
    )


def main() -> None:
    """Полный запуск walk-forward SHAP для Variant A."""
    setup_plot_style()
    X, y, regimes, test_index = load_data()
    if len(test_index) == 0:
        raise ValueError("Тестовый индекс пуст: проверьте regime_predictions.parquet или TEST_START/TEST_END")

    background_size = int(os.getenv("SHAP_BACKGROUND_SIZE", str(SHAP_BACKGROUND_SIZE)))
    nsamples = int(os.getenv("SHAP_NSAMPLES", str(SHAP_NSAMPLES)))
    log.info(
        "Walk-forward Kernel SHAP: background_size=%d, nsamples=%d, n_test=%d, gap=%d",
        background_size, nsamples, len(test_index), GAP_MONTHS,
    )

    shap_values = compute_shap_test_period_walk_forward(
        X,
        y,
        regimes,
        test_index=test_index,
        gap=GAP_MONTHS,
        background_size=background_size,
        nsamples=nsamples,
    )
    save_shap_outputs(shap_values, regimes)

    t23 = compute_t23(shap_values, regimes, top_n=10)
    save_t23(t23)
    generate_all_figures(shap_values, X, regimes)
    validate_outputs(shap_values, t23, regimes)
    log.info("Готово: walk-forward SHAP Variant A завершён.")


if __name__ == "__main__":
    main()
