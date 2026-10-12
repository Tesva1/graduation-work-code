"""Шаг 8 (оценка): метрики и визуализации режимных моделей.

Сравнивает 11 систем (8 baseline + 3 регимных варианта A/B/C) на едином walk-forward
test-периоде Feb 2020 — Nov 2025 (70 фолдов). Готовит таблицы T18, T19, T21, T22
и фигуры fig_44–fig_55. T20 уже сохранён скриптом 09_regime_models.py.

Вход:
    data/processed/baseline_predictions.parquet  — 70 × 8
    data/processed/regime_predictions.parquet    — 70 × 3
    data/processed/monthly_target.parquet        — y_next per month
    data/processed/monthly_regimes.parquet       — глобальные HMM-режимы
    data/raw/imoex.csv                           — для buy-and-hold
    tables/step_09/T20_best_per_regime.csv       — для fig_54

Выход:
    tables/step_09/T18_all_systems.csv
    tables/step_09/T19_metrics_by_regime.csv
    tables/step_09/T21_annual_returns.csv
    tables/step_09/T22_pairwise_tests.csv
    figures/step_09/fig_44–fig_55 (PNG)

Запуск: python src/09b_regime_metrics_figures.py
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_PROCESSED, DATA_RAW, FIGURES, TABLES  # noqa: E402
from src.utils import save_fig, save_table, setup_plot_style  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("step_09b")

SUBFOLDER = "step_09"
REGIME_VARIANTS = ["VariantA", "VariantB", "VariantC"]
BASELINE_ORDER = [
    "LogisticRegression",
    "SVM",
    "RandomForest",
    "XGBoost",
    "CatBoost",
    "LSTM",
    "TFT",
    "TabTransformer",
]
ALL_SYSTEMS = BASELINE_ORDER + REGIME_VARIANTS

# Метки человекочитаемых названий режимов на русском (для глобальной HMM из шага 6).
# В monthly_regimes.parquet 0 = медвежий (39 точек), 1 = бычий (93 точки).
REGIME_LABELS_RU = {0: "медвежий", 1: "бычий"}


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка модуля 08 для повторного использования compute_metrics / align
# ──────────────────────────────────────────────────────────────────────────────


def _load_baseline_module():
    """Загружает src/08_baseline_models.py через importlib.util.

    Имя файла начинается с цифры → стандартный import не работает.
    """
    path = os.path.join(PROJECT_ROOT, "src", "08_baseline_models.py")
    spec = importlib.util.spec_from_file_location("_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_baseline = None


def _b():
    """Lazy-loaded baseline module (compute_metrics, align_strategy_returns)."""
    global _baseline
    if _baseline is None:
        _baseline = _load_baseline_module()
    return _baseline


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────


def align_strategy_returns_local(
    signal_index: pd.DatetimeIndex,
    imoex_ret: pd.Series,
) -> pd.Series:
    """Локальная копия baseline.align_strategy_returns без зависимости от модуля.

    Сигнал на конец месяца t применяется к доходности IMOEX за месяц t+1.
    """
    future_ret = imoex_ret.shift(-1)
    return future_ret.reindex(signal_index)


def load_all_predictions() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Загружает все 11 систем + таргет + IMOEX returns + глобальные режимы.

    Returns:
        preds:   DataFrame (70, 11) — 8 baseline + 3 регимных варианта.
        y:       Series  истинных меток (132 наблюдений, индекс конец месяца).
        returns: Series  месячных доходностей IMOEX.
        regimes: Series  глобальных HMM-режимов (132 точки).
    """
    baseline_path = os.path.join(DATA_PROCESSED, "baseline_predictions.parquet")
    regime_path = os.path.join(DATA_PROCESSED, "regime_predictions.parquet")
    target_path = os.path.join(DATA_PROCESSED, "monthly_target.parquet")
    regimes_path = os.path.join(DATA_PROCESSED, "monthly_regimes.parquet")

    bl = pd.read_parquet(baseline_path)
    rg = pd.read_parquet(regime_path)
    tgt = pd.read_parquet(target_path)
    reg = pd.read_parquet(regimes_path)

    if not bl.index.equals(rg.index):
        rg = rg.reindex(bl.index)

    cols_baseline = [c for c in BASELINE_ORDER if c in bl.columns]
    cols_regime = [c for c in REGIME_VARIANTS if c in rg.columns]
    preds = pd.concat([bl[cols_baseline], rg[cols_regime]], axis=1)

    # Последняя строка y_next NaN (из shift(-1) — нет данных для t+1).
    y = tgt["y_next"].dropna().astype(int)

    imoex_path = os.path.join(DATA_RAW, "imoex.csv")
    imoex = pd.read_csv(imoex_path, parse_dates=["date"])
    imoex = imoex.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
    returns = imoex["close"].resample("ME").last().pct_change()

    regimes = reg["regime"].astype(int)

    log.info(
        "Загружено: preds %s, y %s, returns %s, regimes %s",
        preds.shape, y.shape, returns.shape, regimes.shape,
    )
    return preds, y, returns, regimes


# ──────────────────────────────────────────────────────────────────────────────
# Метрики
# ──────────────────────────────────────────────────────────────────────────────


def _system_metrics(probs: pd.Series, y: pd.Series, returns: pd.Series) -> dict:
    """Вычисляет 6 метрик для одной системы (один столбец вероятностей)."""
    probs = probs.dropna()
    if probs.empty:
        return {
            "AUC-ROC": np.nan,
            "Accuracy": np.nan,
            "F1 macro": np.nan,
            "Precision(y=1)": np.nan,
            "Log-Loss": np.nan,
            "Доходность стратегии": np.nan,
        }
    y_true = y.loc[probs.index].astype(int)
    y_pred = (probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, probs) if y_true.nunique() > 1 else np.nan
    except Exception:
        auc = np.nan

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)

    try:
        ll = log_loss(y_true, probs)
    except Exception:
        ll = np.nan

    fut = align_strategy_returns_local(probs.index, returns)
    strategy_ret = fut * y_pred.values
    cum_return = float((1 + strategy_ret.fillna(0)).prod() - 1)

    return {
        "AUC-ROC": round(float(auc) if not np.isnan(auc) else np.nan, 4),
        "Accuracy": round(float(acc), 4),
        "F1 macro": round(float(f1), 4),
        "Precision(y=1)": round(float(prec), 4),
        "Log-Loss": round(float(ll) if not np.isnan(ll) else np.nan, 4),
        "Доходность стратегии": round(cum_return, 4),
    }


def compute_t18(preds_df: pd.DataFrame, y: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """T18 — 11×6 метрик: все системы (8 baseline + A/B/C)."""
    rows = []
    for sys_name in preds_df.columns:
        m = _system_metrics(preds_df[sys_name], y, returns)
        m["Система"] = sys_name
        rows.append(m)
    df = pd.DataFrame(rows).set_index("Система")
    cols = ["AUC-ROC", "Accuracy", "F1 macro", "Precision(y=1)", "Log-Loss", "Доходность стратегии"]
    return df[cols]


def compute_t19(
    preds_df: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """T19 — метрики 3 регимных вариантов отдельно по бычьему/медвежьему режиму.

    Returns:
        DataFrame с MultiIndex (Система, Режим) × 6 метрик.
    """
    rows = []
    for variant in [c for c in REGIME_VARIANTS if c in preds_df.columns]:
        probs = preds_df[variant].dropna()
        for r_val in sorted(regimes.dropna().unique()):
            r_label = REGIME_LABELS_RU.get(int(r_val), f"режим {int(r_val)}")
            mask = regimes.loc[probs.index].astype(int) == int(r_val)
            sub_idx = probs.index[mask.values]
            if len(sub_idx) < 2:
                m = {
                    "AUC-ROC": np.nan,
                    "Accuracy": np.nan,
                    "F1 macro": np.nan,
                    "Precision(y=1)": np.nan,
                    "Log-Loss": np.nan,
                    "Доходность стратегии": np.nan,
                }
            else:
                m = _system_metrics(probs.loc[sub_idx], y, returns)
            m["Система"] = variant
            m["Режим"] = r_label
            m["N мес."] = int(len(sub_idx))
            rows.append(m)
    df = pd.DataFrame(rows).set_index(["Система", "Режим"])
    cols = ["AUC-ROC", "Accuracy", "F1 macro", "Precision(y=1)", "Log-Loss", "Доходность стратегии", "N мес."]
    return df[cols]


def compute_t21(preds_df: pd.DataFrame, y: pd.Series, returns: pd.Series) -> pd.DataFrame:
    """T21 — годовые накопленные доходности по каждой системе + buy-and-hold IMOEX.

    Returns:
        DataFrame (систем + 1 строка B&H) × (годы test-периода + Итого).
    """
    rows: list[dict] = []
    common_idx = preds_df.dropna(how="all").index
    bah_ret = align_strategy_returns_local(common_idx, returns)
    years = sorted(set(common_idx.year.tolist()))

    for sys_name in preds_df.columns:
        probs = preds_df[sys_name].reindex(common_idx)
        y_pred = (probs >= 0.5).astype(int)
        strat_ret = bah_ret * y_pred.values
        row = {"Система": sys_name}
        for yr in years:
            mask = strat_ret.index.year == yr
            yr_ret = (1 + strat_ret.loc[mask].fillna(0)).prod() - 1
            row[str(yr)] = round(float(yr_ret), 4)
        total = (1 + strat_ret.fillna(0)).prod() - 1
        row["Итого"] = round(float(total), 4)
        rows.append(row)

    # Buy-and-hold IMOEX как baseline для сравнения.
    bah_row = {"Система": "Buy & Hold IMOEX"}
    for yr in years:
        mask = bah_ret.index.year == yr
        yr_ret = (1 + bah_ret.loc[mask].fillna(0)).prod() - 1
        bah_row[str(yr)] = round(float(yr_ret), 4)
    bah_total = (1 + bah_ret.fillna(0)).prod() - 1
    bah_row["Итого"] = round(float(bah_total), 4)
    rows.append(bah_row)

    df = pd.DataFrame(rows).set_index("Система")
    return df


def compute_t22(preds_df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """T22 — pairwise t-test: лучший baseline vs каждый регимный вариант.

    Сравнение по per-fold accuracy (бинарная корректность предсказания на каждом
    тестовом месяце). Для t-test связных выборок: для каждого фолда — 1 если
    pred == y_true, иначе 0. Двусторонний тест.

    Returns:
        DataFrame (3 регимных варианта × {Baseline-best, t-stat, p-value, Значимо}).
    """
    baseline_cols = [c for c in BASELINE_ORDER if c in preds_df.columns]
    if not baseline_cols:
        return pd.DataFrame()

    # Лучший baseline по AUC-ROC
    aucs = {}
    for c in baseline_cols:
        probs = preds_df[c].dropna()
        if probs.empty:
            continue
        y_true = y.loc[probs.index].astype(int)
        try:
            aucs[c] = roc_auc_score(y_true, probs) if y_true.nunique() > 1 else np.nan
        except Exception:
            aucs[c] = np.nan
    best_base = max(aucs, key=lambda k: aucs[k] if not np.isnan(aucs[k]) else -1)

    base_probs = preds_df[best_base].dropna()
    base_pred = (base_probs >= 0.5).astype(int)
    base_y = y.loc[base_probs.index].astype(int)
    base_correct = (base_pred.values == base_y.values).astype(int)

    rows = []
    for variant in [v for v in REGIME_VARIANTS if v in preds_df.columns]:
        v_probs = preds_df[variant].dropna()
        common = base_probs.index.intersection(v_probs.index)
        if len(common) < 3:
            rows.append({
                "Регимный вариант": variant,
                "Baseline-best": best_base,
                "t-stat": np.nan,
                "p-value": np.nan,
                "Значимо (α=0.05)": "—",
            })
            continue
        v_pred = (v_probs.loc[common] >= 0.5).astype(int)
        v_y = y.loc[common].astype(int)
        v_correct = (v_pred.values == v_y.values).astype(int)
        b_correct = (base_pred.loc[common].values == base_y.loc[common].values).astype(int)
        try:
            t_stat, p_val = ttest_rel(v_correct, b_correct)
        except Exception:
            t_stat, p_val = np.nan, np.nan
        rows.append({
            "Регимный вариант": variant,
            "Baseline-best": best_base,
            "t-stat": round(float(t_stat), 4) if not np.isnan(t_stat) else np.nan,
            "p-value": round(float(p_val), 4) if not np.isnan(p_val) else np.nan,
            "Значимо (α=0.05)": "да" if (not np.isnan(p_val) and p_val < 0.05) else "нет",
        })
    return pd.DataFrame(rows).set_index("Регимный вариант")


# ──────────────────────────────────────────────────────────────────────────────
# Графики
# ──────────────────────────────────────────────────────────────────────────────


def _is_regime_variant(name: str) -> bool:
    return name in REGIME_VARIANTS


def fig_44_auc_bar(t18: pd.DataFrame) -> None:
    """fig_44 — AUC-ROC столбчатая диаграмма всех 10 (11) систем. Победитель выделен."""
    auc = t18["AUC-ROC"].dropna()
    winner = auc.idxmax()

    colors = []
    for s in auc.index:
        if s == winner:
            colors.append("#d62728")  # красный — победитель
        elif _is_regime_variant(s):
            colors.append("#2ca02c")  # зелёный — регимные
        else:
            colors.append("#1f77b4")  # синий — baseline

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(auc.index, auc.values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Случайный классификатор")
    for bar, val in zip(bars, auc.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("AUC-ROC всех систем: 8 базовых + 3 регимных варианта (walk-forward 2020–2025)")
    ax.set_xlabel("Система")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.3, max(0.85, auc.max() + 0.05))
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    save_fig(fig, "fig_44_auc_bar_all_systems", subfolder=SUBFOLDER)


def fig_45_metrics_heatmap(t18: pd.DataFrame) -> None:
    """fig_45 — тепловая карта метрик 11 систем × 6 показателей."""
    cols = ["AUC-ROC", "Accuracy", "F1 macro", "Precision(y=1)", "Log-Loss", "Доходность стратегии"]
    data = t18[cols].astype(float).copy()
    # Нормировка по столбцу для цвета (для Log-Loss инвертируем).
    norm = data.copy()
    for c in cols:
        col = data[c]
        if c == "Log-Loss":
            col = -col  # меньше = лучше → инвертируем для согласованной цветовой шкалы
        if col.max() == col.min():
            norm[c] = 0.5
        else:
            norm[c] = (col - col.min()) / (col.max() - col.min())

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(
        norm,
        annot=data.round(3),
        fmt="",
        cmap="YlGnBu",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Нормированная значимость (выше = лучше)"},
    )
    ax.set_title("Сравнение метрик 11 систем (8 базовых + варианты A/B/C)")
    ax.set_xlabel("Метрика")
    ax.set_ylabel("Система")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    save_fig(fig, "fig_45_metrics_heatmap_11x6", subfolder=SUBFOLDER)


def fig_46_top_cumulative(
    preds_df: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    t18: pd.DataFrame,
) -> None:
    """fig_46 — накопленная доходность: лучший baseline vs лучший регимный vs B&H."""
    baseline_cols = [c for c in BASELINE_ORDER if c in t18.index]
    regime_cols = [c for c in REGIME_VARIANTS if c in t18.index]
    best_base = t18.loc[baseline_cols, "AUC-ROC"].idxmax()
    best_reg = t18.loc[regime_cols, "AUC-ROC"].idxmax()

    common_idx = preds_df.dropna(how="all").index
    bah = align_strategy_returns_local(common_idx, returns)
    bah_cum = (1 + bah.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(14, 7))
    for sys_name, color, ls in [
        (best_base, "#1f77b4", "-"),
        (best_reg, "#d62728", "-"),
    ]:
        probs = preds_df[sys_name].reindex(common_idx)
        y_pred = (probs >= 0.5).astype(int)
        strat = align_strategy_returns_local(probs.index, returns) * y_pred.values
        cum = (1 + strat.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=f"{sys_name}", linewidth=2.0, color=color, linestyle=ls)

    ax.plot(bah_cum.index, bah_cum.values, "k--", linewidth=2.0, label="Пассивный портфель (B&H)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(f"Накопленная доходность: лучший baseline ({best_base}) vs лучший регимный ({best_reg}) vs B&H")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Накопленная доходность (база = 1.0)")
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "fig_46_cumulative_top_systems", subfolder=SUBFOLDER)


def fig_47_all_cumulative(
    preds_df: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
) -> None:
    """fig_47 — накопленная доходность всех 11 систем + B&H."""
    common_idx = preds_df.dropna(how="all").index
    bah = align_strategy_returns_local(common_idx, returns)
    bah_cum = (1 + bah.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(14, 7))
    for sys_name in preds_df.columns:
        probs = preds_df[sys_name].reindex(common_idx)
        if probs.dropna().empty:
            continue
        y_pred = (probs >= 0.5).astype(int)
        strat = align_strategy_returns_local(probs.index, returns) * y_pred.values
        cum = (1 + strat.fillna(0)).cumprod()
        if _is_regime_variant(sys_name):
            ax.plot(cum.index, cum.values, label=sys_name, linewidth=2.2, alpha=1.0)
        else:
            ax.plot(cum.index, cum.values, label=sys_name, linewidth=1.0, alpha=0.55)

    ax.plot(bah_cum.index, bah_cum.values, "k--", linewidth=2.0, label="Пассивный портфель (B&H)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Накопленная доходность всех 11 систем (регимные — жирные, baseline — полупрозрачные)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Накопленная доходность (база = 1.0)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "fig_47_cumulative_all_systems", subfolder=SUBFOLDER)


def fig_48_auc_rolling(
    preds_df: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
) -> None:
    """fig_48 — скользящий 12м AUC-ROC, фон по глобальным режимам (бычий/медвежий)."""
    window = 12
    fig, ax = plt.subplots(figsize=(14, 7))

    # Подложка-режимы. Берём только тестовый период.
    test_idx = preds_df.dropna(how="all").index
    if len(test_idx):
        regs = regimes.reindex(test_idx).fillna(method="ffill")
        # Раскраска по горизонтальным полосам (axvspan) — каждый месяц.
        prev_r: Optional[int] = None
        seg_start = test_idx[0]
        for ts in test_idx:
            r = int(regs.loc[ts]) if pd.notna(regs.loc[ts]) else -1
            if prev_r is None:
                prev_r = r
                seg_start = ts
                continue
            if r != prev_r:
                color = {0: "#fde0e0", 1: "#e0f5e0"}.get(prev_r, "#eeeeee")
                ax.axvspan(seg_start, ts, color=color, alpha=0.45, zorder=0)
                seg_start = ts
                prev_r = r
        # Закрыть последний сегмент.
        if prev_r is not None:
            color = {0: "#fde0e0", 1: "#e0f5e0"}.get(prev_r, "#eeeeee")
            ax.axvspan(seg_start, test_idx[-1], color=color, alpha=0.45, zorder=0)

    # Скользящий AUC для каждой системы.
    for sys_name in preds_df.columns:
        probs = preds_df[sys_name].dropna().sort_index()
        if probs.empty:
            continue
        y_true = y.loc[probs.index].astype(int)
        rolling: dict[pd.Timestamp, float] = {}
        dates = probs.index.tolist()
        for i in range(window - 1, len(dates)):
            wd = dates[i - window + 1 : i + 1]
            p_w = probs.loc[wd]
            y_w = y_true.loc[wd]
            if y_w.nunique() < 2:
                continue
            try:
                rolling[dates[i]] = roc_auc_score(y_w, p_w)
            except Exception:
                pass
        if not rolling:
            continue
        s = pd.Series(rolling)
        if _is_regime_variant(sys_name):
            ax.plot(s.index, s.values, label=sys_name, linewidth=2.2, zorder=3)
        else:
            ax.plot(s.index, s.values, label=sys_name, linewidth=0.9, alpha=0.6, zorder=2)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_title("AUC-ROC во времени (скользящее окно 12 мес.); фон — глобальный HMM-режим")
    ax.set_xlabel("Дата")
    ax.set_ylabel("AUC-ROC")
    # Легенда о цветах режимов
    from matplotlib.patches import Patch
    handles, labels_ = ax.get_legend_handles_labels()
    handles = handles + [
        Patch(facecolor="#e0f5e0", alpha=0.6, label="Бычий режим"),
        Patch(facecolor="#fde0e0", alpha=0.6, label="Медвежий режим"),
    ]
    labels_ = labels_ + ["Бычий режим", "Медвежий режим"]
    ax.legend(handles, labels_, fontsize=8, ncol=2, loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "fig_48_auc_rolling_with_regime_bg", subfolder=SUBFOLDER)


def fig_49_accuracy_by_regime(
    preds_df: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
) -> None:
    """fig_49 — Accuracy по режимам, грызуный bar (11 систем × 2 режима)."""
    common_idx = preds_df.dropna(how="all").index
    regs = regimes.reindex(common_idx).astype("Int64")
    unique_r = sorted([int(v) for v in regs.dropna().unique()])

    rows = []
    for sys_name in preds_df.columns:
        probs = preds_df[sys_name].dropna()
        y_pred = (probs >= 0.5).astype(int)
        y_true = y.loc[probs.index].astype(int)
        for r in unique_r:
            mask = regs.loc[probs.index].astype(int) == r
            if mask.sum() < 1:
                rows.append({"Система": sys_name, "Режим": REGIME_LABELS_RU.get(r, str(r)), "Accuracy": np.nan})
            else:
                acc = accuracy_score(y_true[mask.values], y_pred[mask.values])
                rows.append({"Система": sys_name, "Режим": REGIME_LABELS_RU.get(r, str(r)), "Accuracy": acc})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot = df.pivot(index="Система", columns="Режим", values="Accuracy").reindex(preds_df.columns)
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.6, width=0.75)
    ax.set_title("Accuracy по режимам: 11 систем × 2 режима (глобальный HMM)")
    ax.set_xlabel("Система")
    ax.set_ylabel("Accuracy")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    plt.xticks(rotation=25, ha="right")
    ax.legend(title="Режим")
    fig.tight_layout()
    save_fig(fig, "fig_49_accuracy_by_regime", subfolder=SUBFOLDER)


def fig_50_auc_gain_over_time(preds_df: pd.DataFrame, y: pd.Series) -> None:
    """fig_50 — Δ AUC: VariantA − лучший baseline (по AUC), скользящее 12м окно."""
    window = 12
    baseline_cols = [c for c in BASELINE_ORDER if c in preds_df.columns]

    # Глобально лучший baseline для опоры (по полному test-периоду).
    aucs = {}
    for c in baseline_cols:
        probs = preds_df[c].dropna()
        y_true = y.loc[probs.index].astype(int)
        try:
            aucs[c] = roc_auc_score(y_true, probs) if y_true.nunique() > 1 else -1
        except Exception:
            aucs[c] = -1
    best_base = max(aucs, key=lambda k: aucs[k])

    base_probs = preds_df[best_base].dropna().sort_index()
    base_y = y.loc[base_probs.index].astype(int)

    fig, ax = plt.subplots(figsize=(14, 6))

    for variant in [v for v in REGIME_VARIANTS if v in preds_df.columns]:
        v_probs = preds_df[variant].dropna().sort_index()
        common = base_probs.index.intersection(v_probs.index)
        if len(common) < window:
            continue
        delta_series: dict[pd.Timestamp, float] = {}
        dates = list(common)
        for i in range(window - 1, len(dates)):
            wd = dates[i - window + 1 : i + 1]
            yw = y.loc[wd].astype(int)
            if yw.nunique() < 2:
                continue
            try:
                a_v = roc_auc_score(yw, v_probs.loc[wd])
                a_b = roc_auc_score(yw, base_probs.loc[wd])
                delta_series[dates[i]] = a_v - a_b
            except Exception:
                pass
        if not delta_series:
            continue
        s = pd.Series(delta_series)
        ax.plot(s.index, s.values, label=f"Δ AUC ({variant} − {best_base})", linewidth=1.8)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(f"Прирост AUC регимных вариантов над лучшим baseline ({best_base}), 12-мес. окно")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Δ AUC")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "fig_50_auc_gain_over_time", subfolder=SUBFOLDER)


def fig_51_confusion_matrices(
    preds_df: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    t18: pd.DataFrame,
) -> None:
    """fig_51 — 4×3 grid: топ-4 системы × {общее, режим 0, режим 1}."""
    top4 = t18["AUC-ROC"].nlargest(4).index.tolist()
    regs = regimes.reindex(preds_df.index).astype("Int64")
    unique_r = sorted([int(v) for v in regs.dropna().unique()])

    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    for i, sys_name in enumerate(top4):
        probs = preds_df[sys_name].dropna()
        y_true_all = y.loc[probs.index].astype(int)
        y_pred_all = (probs >= 0.5).astype(int)

        # Общее
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
        ax = axes[i, 0]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Падение", "Рост"], yticklabels=["Падение", "Рост"],
                    cbar=False, linewidths=0.5)
        ax.set_title(f"{sys_name} — общее")
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истинное")

        # По режимам
        for j, r in enumerate(unique_r[:2]):
            mask = (regs.loc[probs.index].astype(int) == r).values
            ax_r = axes[i, j + 1]
            if mask.sum() < 1:
                ax_r.text(0.5, 0.5, "Нет данных", ha="center", va="center")
                ax_r.set_title(f"{sys_name} — {REGIME_LABELS_RU.get(r, str(r))}")
                ax_r.axis("off")
                continue
            cm_r = confusion_matrix(y_true_all[mask], y_pred_all[mask], labels=[0, 1])
            sns.heatmap(cm_r, annot=True, fmt="d", cmap="Blues", ax=ax_r,
                        xticklabels=["Падение", "Рост"], yticklabels=["Падение", "Рост"],
                        cbar=False, linewidths=0.5)
            ax_r.set_title(f"{sys_name} — {REGIME_LABELS_RU.get(r, str(r))} режим")
            ax_r.set_xlabel("Предсказано")
            ax_r.set_ylabel("Истинное")
    fig.suptitle("Матрицы ошибок топ-4 систем: общее + по режимам", fontsize=14)
    fig.tight_layout()
    save_fig(fig, "fig_51_confusion_matrices_grid", subfolder=SUBFOLDER)


def fig_52_drawdown(
    preds_df: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
) -> None:
    """fig_52 — кривые просадки топ-3 систем по AUC."""
    # Считаем accuracy и AUC по системе чтобы выбрать топ-3.
    aucs = {}
    for c in preds_df.columns:
        probs = preds_df[c].dropna()
        y_true = y.loc[probs.index].astype(int)
        try:
            aucs[c] = roc_auc_score(y_true, probs) if y_true.nunique() > 1 else -1
        except Exception:
            aucs[c] = -1
    top3 = sorted(aucs, key=lambda k: aucs[k], reverse=True)[:3]

    common_idx = preds_df.dropna(how="all").index
    fig, ax = plt.subplots(figsize=(14, 6))
    for sys_name in top3:
        probs = preds_df[sys_name].reindex(common_idx)
        y_pred = (probs >= 0.5).astype(int)
        strat = align_strategy_returns_local(probs.index, returns) * y_pred.values
        cum = (1 + strat.fillna(0)).cumprod()
        peak = cum.cummax()
        dd = cum / peak - 1.0
        ax.plot(dd.index, dd.values * 100.0, label=sys_name, linewidth=1.6)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Просадка торговых стратегий топ-3 систем (% от пика)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Просадка, %")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "fig_52_drawdown_top3", subfolder=SUBFOLDER)


def fig_53_monthly_returns_by_regime(
    preds_df: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    regimes: pd.Series,
    winner_name: str,
) -> None:
    """fig_53 — месячные доходности стратегии победителя, столбцы окрашены по режимам."""
    if winner_name not in preds_df.columns:
        log.warning("fig_53: победитель %s не в preds_df", winner_name)
        return

    probs = preds_df[winner_name].dropna()
    y_pred = (probs >= 0.5).astype(int)
    fut = align_strategy_returns_local(probs.index, returns)
    strat = (fut * y_pred.values).fillna(0.0)

    regs = regimes.reindex(probs.index).astype("Int64")
    color_map = {0: "#d62728", 1: "#2ca02c"}  # медвежий красный, бычий зелёный
    bar_colors = [color_map.get(int(r), "#888888") if pd.notna(r) else "#888888" for r in regs.values]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(strat.index, strat.values * 100, color=bar_colors, edgecolor="none", width=22)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Месячные доходности стратегии «{winner_name}», окраска по глобальному HMM-режиму")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Месячная доходность, %")
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#2ca02c", label="Бычий режим"),
        Patch(facecolor="#d62728", label="Медвежий режим"),
    ]
    ax.legend(handles=handles)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "fig_53_monthly_returns_by_regime", subfolder=SUBFOLDER)


def fig_54_best_model_per_regime_over_time() -> None:
    """fig_54 — кто побеждал в каждом режиме на каждом фолде (из T20).

    Stacked bar на основе T20_best_per_regime.csv: ось X — фолды (тестовые месяцы),
    раскраска показывает, какой baseline был выбран per-regime в данном фолде.
    """
    t20_path = os.path.join(TABLES, SUBFOLDER, "T20_best_per_regime.csv")
    if not os.path.exists(t20_path):
        log.warning("fig_54: T20 не найден по %s — пропуск", t20_path)
        return
    t20 = pd.read_csv(t20_path)
    if "fold_test_date" not in t20.columns:
        log.warning("fig_54: ожидается колонка fold_test_date в T20")
        return
    t20["fold_test_date"] = pd.to_datetime(t20["fold_test_date"])

    # Список регимов в T20: regime_0_best, regime_1_best, ...
    regime_cols = [c for c in t20.columns if c.startswith("regime_") and c.endswith("_best")]
    if not regime_cols:
        log.warning("fig_54: не найдено колонок regime_*_best в T20")
        return

    fig, axes = plt.subplots(len(regime_cols), 1, figsize=(14, 3.5 * len(regime_cols)),
                             sharex=True, squeeze=False)

    # Цвета для моделей.
    all_models = pd.unique(pd.concat([t20[c] for c in regime_cols]))
    palette = sns.color_palette("tab10", n_colors=max(3, len(all_models)))
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(sorted(all_models, key=str))}

    for ax_i, rc in enumerate(regime_cols):
        ax = axes[ax_i, 0]
        regime_idx = int(rc.split("_")[1])
        label_ru = REGIME_LABELS_RU.get(regime_idx, f"режим {regime_idx}")
        # Каждая ячейка — вертикальная полоска цвета модели.
        ys = np.ones(len(t20))
        bar_colors = [color_map[m] for m in t20[rc].values]
        ax.bar(t20["fold_test_date"], ys, color=bar_colors, width=22, edgecolor="none")
        ax.set_title(f"Лучшая модель в режиме «{label_ru}» по фолдам walk-forward")
        ax.set_yticks([])
        ax.set_xlabel("Дата фолда")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Общая легенда.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=color_map[m], label=str(m)) for m in sorted(all_models, key=str)]
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(handles)),
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig_54_best_model_per_regime_over_time", subfolder=SUBFOLDER)


def fig_55_auc_delta_heatmap(
    preds_df: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
) -> None:
    """fig_55 — Δ AUC heatmap: (вариант − лучший baseline) × (режим).

    Колонки: бычий, медвежий, общее. Строки: VariantA / VariantB / VariantC.
    Цвет — диверг. RdBu_r (красный = регимный лучше).
    """
    baseline_cols = [c for c in BASELINE_ORDER if c in preds_df.columns]
    regime_cols = [c for c in REGIME_VARIANTS if c in preds_df.columns]
    regs = regimes.reindex(preds_df.index).astype("Int64")
    unique_r = sorted([int(v) for v in regs.dropna().unique()])

    # AUC общая для каждой системы.
    def auc_for(sys_name: str, mask: Optional[pd.Series] = None) -> float:
        probs = preds_df[sys_name].dropna()
        if mask is not None:
            mm = mask.reindex(probs.index).fillna(False).astype(bool)
            probs = probs[mm.values]
        if probs.empty:
            return np.nan
        y_true = y.loc[probs.index].astype(int)
        if y_true.nunique() < 2:
            return np.nan
        try:
            return float(roc_auc_score(y_true, probs))
        except Exception:
            return np.nan

    cols_labels = []
    masks = []
    for r in unique_r:
        cols_labels.append(REGIME_LABELS_RU.get(r, f"режим {r}"))
        masks.append((regs == r))
    cols_labels.append("общее")
    masks.append(None)

    # Лучший baseline в каждом срезе (берём максимум baseline AUC в этом срезе).
    base_aucs_per_slice = []
    for mask in masks:
        per_base = {b: auc_for(b, mask) for b in baseline_cols}
        if all(np.isnan(v) for v in per_base.values()):
            base_aucs_per_slice.append((np.nan, "—"))
        else:
            best_b = max(per_base, key=lambda k: per_base[k] if not np.isnan(per_base[k]) else -1)
            base_aucs_per_slice.append((per_base[best_b], best_b))

    delta = np.zeros((len(regime_cols), len(cols_labels)))
    annot = np.empty(delta.shape, dtype=object)
    for i, variant in enumerate(regime_cols):
        for j, mask in enumerate(masks):
            v_auc = auc_for(variant, mask)
            b_auc, b_name = base_aucs_per_slice[j]
            if np.isnan(v_auc) or np.isnan(b_auc):
                delta[i, j] = 0.0
                annot[i, j] = "—"
            else:
                d = v_auc - b_auc
                delta[i, j] = d
                annot[i, j] = f"{d:+.3f}\nvs {b_name}"

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        delta,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0.0,
        ax=ax,
        xticklabels=cols_labels,
        yticklabels=regime_cols,
        linewidths=0.5,
        cbar_kws={"label": "Δ AUC (регимный − лучший baseline)"},
    )
    ax.set_title("Прирост AUC регимных вариантов над лучшим baseline по срезам режимов")
    ax.set_xlabel("Срез / режим")
    ax.set_ylabel("Регимный вариант")
    fig.tight_layout()
    save_fig(fig, "fig_55_auc_delta_heatmap", subfolder=SUBFOLDER)


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    setup_plot_style()
    preds, y, returns, regimes = load_all_predictions()

    log.info("Считаем T18…")
    t18 = compute_t18(preds, y, returns)
    save_table(t18, "T18_all_systems", subfolder=SUBFOLDER)

    log.info("Считаем T19…")
    t19 = compute_t19(preds, y, returns, regimes)
    save_table(t19, "T19_metrics_by_regime", subfolder=SUBFOLDER)

    log.info("Считаем T21…")
    t21 = compute_t21(preds, y, returns)
    save_table(t21, "T21_annual_returns", subfolder=SUBFOLDER)

    log.info("Считаем T22…")
    t22 = compute_t22(preds, y)
    save_table(t22, "T22_pairwise_tests", subfolder=SUBFOLDER)

    log.info("Строим 12 графиков fig_44–fig_55…")
    fig_44_auc_bar(t18)
    fig_45_metrics_heatmap(t18)
    fig_46_top_cumulative(preds, y, returns, t18)
    fig_47_all_cumulative(preds, y, returns)
    fig_48_auc_rolling(preds, y, regimes)
    fig_49_accuracy_by_regime(preds, y, regimes)
    fig_50_auc_gain_over_time(preds, y)
    fig_51_confusion_matrices(preds, y, regimes, t18)
    fig_52_drawdown(preds, y, returns)
    winner = t18["AUC-ROC"].idxmax()
    fig_53_monthly_returns_by_regime(preds, y, returns, regimes, winner)
    fig_54_best_model_per_regime_over_time()
    fig_55_auc_delta_heatmap(preds, y, regimes)

    log.info(
        "B.3 готов. Победитель T18: %s, AUC=%.4f",
        winner, float(t18.loc[winner, "AUC-ROC"]),
    )


if __name__ == "__main__":
    main()
