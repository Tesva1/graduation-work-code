"""Шаг 2: построение 25 месячных признаков и целевой переменной.

Вход:  data/raw/{imoex,brent,urals,usdrub,key_rate,rgbi}.csv
Выход: data/processed/monthly_features.parquet  (≈132 × 25)
       data/processed/monthly_target.parquet    (≈132 × 1)
       figures/step_03/fig_06..fig_09 (PNG+PDF)
       tables/T1_descriptive_stats.csv, T2_vif.csv

Запуск: python src/03_feature_engineering.py
"""
from __future__ import annotations

import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_RAW, DATA_PROCESSED,
    RANDOM_STATE, START_DATE, END_DATE,
    VOLATILITY_WINDOWS, RSI_PERIOD, SMA_SHORT, SMA_LONG,
)
from src import feature_utils as fu
from src.utils import setup_plot_style, save_fig, save_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_03")


def load_raw_data() -> dict[str, pd.DataFrame]:
    """Читает 6 CSV из data/raw/, парсит даты, сортирует, обрезает по START/END."""
    spec = {
        "imoex":    ("imoex.csv",    "close"),
        "brent":    ("brent.csv",    "close"),
        "urals":    ("urals.csv",    "close"),
        "usdrub":   ("usdrub.csv",   "close"),
        "rgbi":     ("rgbi.csv",     "close"),
        "key_rate": ("key_rate.csv", "rate"),
    }
    out = {}
    for name, (fname, price_col) in spec.items():
        df = pd.read_csv(os.path.join(DATA_RAW, fname), parse_dates=["date"])
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        df = df.loc[START_DATE:END_DATE]
        if price_col not in df.columns:
            raise ValueError(f"{fname}: нет столбца {price_col}")
        out[name] = df
        log.info(f"{name}: {len(df)} строк, {df.index.min().date()} → {df.index.max().date()}")
    return out


def build_monthly_grid(imoex: pd.DataFrame) -> pd.DatetimeIndex:
    """Сетка из последних торговых дней каждого месяца IMOEX."""
    return imoex.resample("ME").last().dropna().index


def align_to_monthly(daily: pd.Series, month_ends: pd.DatetimeIndex) -> pd.Series:
    """Для каждого month_end берёт последнее доступное дневное значение ≤ этой даты.
    Обрабатывает нерабочие дни и смешанную дневно-месячную частоту urals 2025.
    """
    daily_sorted = daily.sort_index()
    out = {}
    for t in month_ends:
        sub = daily_sorted.loc[:t]
        out[t] = float(sub.iloc[-1]) if len(sub) else np.nan
    return pd.Series(out, dtype=float)


def compute_features(raw: dict, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Возвращает DataFrame (len(month_ends), 25) — ровно 25 признаков исследования."""
    imoex_close = raw["imoex"]["close"]
    imoex_volume = raw["imoex"]["volume"]
    brent_close = raw["brent"]["close"]
    urals_close = raw["urals"]["close"]
    usdrub_close = raw["usdrub"]["close"]
    rgbi_close = raw["rgbi"]["close"]
    key_rate = raw["key_rate"]["rate"]

    imoex_daily_ret = imoex_close.pct_change()

    imoex_m = align_to_monthly(imoex_close, month_ends)
    urals_m = align_to_monthly(urals_close, month_ends)
    brent_m = align_to_monthly(brent_close, month_ends)
    usdrub_m = align_to_monthly(usdrub_close, month_ends)
    rgbi_m = align_to_monthly(rgbi_close, month_ends)
    key_rate_m = align_to_monthly(key_rate, month_ends)

    df = pd.DataFrame(index=month_ends)

    # 1–5: ценовая динамика IMOEX
    df["imoex_ret_1m"]  = fu.monthly_simple_return(imoex_m, 1)
    df["imoex_ret_3m"]  = fu.monthly_simple_return(imoex_m, 3)
    df["imoex_ret_6m"]  = fu.monthly_simple_return(imoex_m, 6)
    df["imoex_ret_12m"] = fu.monthly_simple_return(imoex_m, 12)
    df["imoex_ret_lastweek"] = pd.Series(
        [fu.last_week_return(imoex_close, t) for t in month_ends], index=month_ends
    )

    # 6–11: волатильность и риск
    vol_20d = fu.annualised_volatility(imoex_daily_ret, 20)
    vol_60d = fu.annualised_volatility(imoex_daily_ret, 60)
    df["imoex_vol_20d"] = vol_20d.reindex(month_ends, method="ffill")
    df["imoex_vol_60d"] = vol_60d.reindex(month_ends, method="ffill")
    df["imoex_vol_ratio_20_60"] = df["imoex_vol_20d"] / df["imoex_vol_60d"]
    df["imoex_max_drawdown_intramonth"] = pd.Series(
        [fu.intramonth_max_drawdown(imoex_close, t) for t in month_ends], index=month_ends
    )
    df["imoex_skew_20d"] = imoex_daily_ret.rolling(20).skew().reindex(month_ends, method="ffill")
    log.info("Считаем GARCH (это займёт ~2–3 минуты)...")
    df["imoex_garch_vol"] = fu.garch_conditional_volatility_at(
        imoex_daily_ret.dropna(), month_ends, window=252, random_state=RANDOM_STATE
    )

    # 12–13: объём и ликвидность
    vol_avg_20 = imoex_volume.rolling(20).mean()
    vol_avg_60 = imoex_volume.rolling(60).mean()
    df["imoex_volume_ratio_20_60"] = (vol_avg_20 / vol_avg_60).reindex(month_ends, method="ffill")
    daily_turnover = imoex_close * imoex_volume
    df["imoex_log_turnover"] = pd.Series(
        [float(np.log1p(daily_turnover.loc[:t].tail(21).mean())) for t in month_ends],
        index=month_ends,
    )

    # 14–16: технические
    df["imoex_rsi_14"] = fu.rsi(imoex_close, RSI_PERIOD).reindex(month_ends, method="ffill")
    df["imoex_bollinger_pos"] = fu.bollinger_position(imoex_close, SMA_SHORT).reindex(month_ends, method="ffill")
    sma_s = imoex_close.rolling(SMA_SHORT).mean()
    sma_l = imoex_close.rolling(SMA_LONG).mean()
    df["imoex_sma_ratio"] = (sma_s / sma_l - 1).reindex(month_ends, method="ffill")

    # 17–23: макро
    df["urals_ret_1m"] = fu.monthly_simple_return(urals_m, 1)
    df["urals_ret_3m"] = fu.monthly_simple_return(urals_m, 3)
    df["urals_brent_spread"] = (urals_m - brent_m) / brent_m
    df["usdrub_change"] = fu.monthly_simple_return(usdrub_m, 1)
    df["key_rate_level"] = key_rate_m
    df["key_rate_change"] = key_rate_m.diff()
    df["rgbi_ret_1m"] = fu.monthly_simple_return(rgbi_m, 1)

    # 24–25: производные
    df["urals_x_usdrub"] = df["urals_ret_1m"] * df["usdrub_change"]
    df["imoex_excess_return"] = df["imoex_ret_1m"] - df["key_rate_level"] / 100 / 12

    if df.shape[1] != 25:
        raise AssertionError(f"Ожидалось 25 признаков, получено {df.shape[1]}: {list(df.columns)}")
    return df


def compute_target(imoex_monthly: pd.Series) -> pd.Series:
    """y_{t+1} = 1, если IMOEX вырос за следующий месяц, иначе 0."""
    next_ret = imoex_monthly.pct_change().shift(-1)
    y = (next_ret > 0).astype("Int64")
    y[next_ret.isna()] = pd.NA
    return y.rename("y_next")


def validate(features: pd.DataFrame, target: pd.Series) -> None:
    """Жёсткие проверки признаков и целевой переменной."""
    assert features.shape[1] == 25, f"shape={features.shape}, ожидалось (*, 25)"
    assert 120 <= len(features) <= 140, f"строк {len(features)}, ожидалось ≈132"

    arr = features.to_numpy(dtype=float, na_value=np.nan)
    assert not np.isinf(arr).any(), "обнаружены inf в признаках"

    nan_per_row = features.isna().sum(axis=1)
    n_full_rows = (nan_per_row == 0).sum()
    assert n_full_rows >= len(features) - 15, \
        f"полностью заполненных строк {n_full_rows}, ожидалось ≥ {len(features)-15}"

    common = features.join(target.rename("y_next").astype(float)).dropna()
    corrs = common.corr()["y_next"].drop("y_next").abs()
    if (corrs > 0.5).any():
        bad = corrs[corrs > 0.5].to_dict()
        raise AssertionError(f"подозрение на leakage: |corr|>0.5 для {bad}")

    y_clean = target.dropna()
    pos_rate = float((y_clean == 1).mean())
    assert 0.35 <= pos_rate <= 0.65, f"баланс y=1: {pos_rate:.2%} вне [35%, 65%]"
    log.info(f"Валидация пройдена. y=1: {pos_rate:.1%}, max|corr|={corrs.max():.3f}")


def save_outputs(features: pd.DataFrame, target: pd.Series) -> None:
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    features.to_parquet(os.path.join(DATA_PROCESSED, "monthly_features.parquet"))
    target.to_frame().to_parquet(os.path.join(DATA_PROCESSED, "monthly_target.parquet"))
    log.info("Сохранено: monthly_features.parquet, monthly_target.parquet")



def fig_06_corr_heatmap(features: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = features.corr()
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, annot=False, cbar_kws={"shrink": 0.7, "label": "Коэффициент корреляции"})
    ax.set_title("Корреляционная матрица 25 месячных признаков", pad=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_06_corr_heatmap", subfolder="step_03")
    log.info("fig_06 сохранён")


def fig_07_feature_histograms(features: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    for col in features.columns:
        label = fu.FEATURE_LABELS.get(col, col)
        ylabel = fu.FEATURE_YLABEL.get(col, "Значение")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(features[col].dropna(), bins=25, alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.set_title(label)
        ax.set_xlabel(ylabel)
        ax.set_ylabel("Количество месяцев")
        fig.tight_layout()
        save_fig(fig, f"fig_07_{col}", subfolder="step_03")
    log.info("fig_07 сохранён (25 отдельных PNG)")


def fig_08_target_balance(target: pd.Series) -> None:
    import matplotlib.pyplot as plt
    y_int = target.dropna().astype(int)
    counts = y_int.value_counts().sort_index()
    counts.index = ["Падение (y=0)", "Рост (y=1)"]
    df_y = y_int.to_frame("y")
    df_y["year"] = df_y.index.year
    yearly = df_y.groupby("year")["y"].mean()

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    counts.plot.bar(ax=ax1, color=["#D65F5F", "#4878D0"], edgecolor="black", rot=0)
    ax1.set_title("Баланс классов целевой переменной")
    ax1.set_ylabel("Количество месяцев")
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 0.5, str(int(v)), ha="center", fontsize=11)
    fig1.tight_layout()
    save_fig(fig1, "fig_08a_class_balance", subfolder="step_03")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    yearly.plot(kind="bar", ax=ax2, color="#4878D0", edgecolor="black")
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="50%")
    ax2.set_title("Доля месяцев роста по годам")
    ax2.set_ylabel("Доля y=1")
    ax2.set_xlabel("Год")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)
    fig2.tight_layout()
    save_fig(fig2, "fig_08b_yearly_balance", subfolder="step_03")
    log.info("fig_08 сохранён (2 отдельных PNG)")


def fig_09_feature_timelines(features: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    cols = [
        "imoex_ret_1m", "imoex_vol_20d", "imoex_garch_vol",
        "urals_ret_1m", "usdrub_change", "key_rate_level",
        "imoex_rsi_14", "urals_brent_spread", "imoex_excess_return",
    ]
    for col in cols:
        label = fu.FEATURE_LABELS.get(col, col)
        ylabel = fu.FEATURE_YLABEL.get(col, "Значение")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(features.index, features[col], linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Дата")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        save_fig(fig, f"fig_09_{col}", subfolder="step_03")
    log.info("fig_09 сохранён (9 отдельных PNG)")


def table_T1_descriptive(features: pd.DataFrame) -> pd.DataFrame:
    desc = features.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    desc["skew"] = features.skew()
    desc["kurtosis"] = features.kurtosis()
    desc["n_missing"] = features.isna().sum()
    desc = desc.round(4)
    save_table(desc, "T1_descriptive_stats")
    log.info(f"T1 сохранена: {desc.shape}")
    return desc


def table_T2_vif(features: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = features.dropna().to_numpy(dtype=float)
    vif = pd.DataFrame({
        "feature": features.columns,
        "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])],
    }).sort_values("VIF", ascending=False).round(2)
    save_table(vif, "T2_vif")
    log.info(f"T2 сохранена, top-3:\n{vif.head(3).to_string(index=False)}")
    return vif


if __name__ == "__main__":
    setup_plot_style()
    raw = load_raw_data()
    month_ends = build_monthly_grid(raw["imoex"])
    log.info(f"Месяцев: {len(month_ends)}, {month_ends[0].date()} → {month_ends[-1].date()}")

    features = compute_features(raw, month_ends)
    imoex_m = align_to_monthly(raw["imoex"]["close"], month_ends)
    target = compute_target(imoex_m)
    log.info(f"features: {features.shape}, target: {target.shape}")
    log.info(f"NaN по столбцам:\n{features.isna().sum().to_string()}")

    validate(features, target)
    save_outputs(features, target)

    fig_06_corr_heatmap(features)
    fig_07_feature_histograms(features)
    fig_08_target_balance(target)
    fig_09_feature_timelines(features)

    T1 = table_T1_descriptive(features)
    T2 = table_T2_vif(features)
