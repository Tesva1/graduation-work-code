"""Шаг 10: перекрёстная проверка VAR/FEVD и SHAP.

Сравнивает долю дисперсии IMOEX из FEVD с SHAP-важностью макро-факторов
Variant A. Это проверка H5: согласуются ли эконометрический и ML-подход.
"""
from __future__ import annotations

import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import FIGURES, TABLES  # noqa: E402
from src.utils import setup_plot_style  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_10b")

SUBFOLDER = "step_10"

FACTOR_LABELS = {
    "urals": "Нефть Urals",
    "usdrub": "USD/RUB",
    "key_rate": "Ключевая ставка",
    "volatility": "Волатильность IMOEX",
}

FEVD_MAP = {
    "urals": "Доходность Urals",
    "usdrub": "Изменение USD/RUB",
    "key_rate": "Изменение ставки ЦБ",
    "volatility": "Волатильность IMOEX",
}

SHAP_GROUPS = {
    "urals": ["urals_ret_1m", "urals_ret_3m", "urals_brent_spread", "urals_x_usdrub"],
    "usdrub": ["usdrub_change", "urals_x_usdrub"],
    "key_rate": ["key_rate_level", "key_rate_change"],
    "volatility": ["imoex_vol_20d", "imoex_vol_60d", "imoex_vol_ratio_20_60", "imoex_garch_vol"],
}


def _ensure_dirs() -> None:
    os.makedirs(os.path.join(FIGURES, SUBFOLDER), exist_ok=True)
    os.makedirs(os.path.join(TABLES, SUBFOLDER), exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> None:
    _ensure_dirs()
    base = os.path.join(FIGURES, SUBFOLDER, name)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    fevd = pd.read_csv(os.path.join(TABLES, "T9_fevd.csv"))
    t23 = pd.read_csv(os.path.join(TABLES, "T23_shap_top_features.csv"))
    return fevd, t23


def build_fevd_macro_importance(fevd: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Extract FEVD macro importance for one horizon."""
    if "Горизонт" not in fevd.columns:
        raise ValueError("FEVD table must contain column 'Горизонт'")
    row_df = fevd.loc[fevd["Горизонт"].astype(int) == int(horizon)]
    if row_df.empty:
        row_df = fevd.tail(1)
        log.warning("Горизонт %s не найден в FEVD; используется последняя строка", horizon)
    row = row_df.iloc[0]
    records = []
    for factor, col in FEVD_MAP.items():
        if col not in fevd.columns:
            raise ValueError(f"FEVD column missing: {col}")
        records.append({
            "factor": factor,
            "factor_ru": FACTOR_LABELS[factor],
            "fevd_importance": float(row[col]),
        })
    return pd.DataFrame(records)


def build_shap_macro_importance(t23: pd.DataFrame, block: str = "global") -> pd.DataFrame:
    """Aggregate SHAP top-feature table into macro-factor groups."""
    required = {"block", "feature", "mean_abs_shap"}
    missing = required.difference(t23.columns)
    if missing:
        raise ValueError(f"T23 missing columns: {sorted(missing)}")
    block_df = t23.loc[t23["block"] == block].copy()
    records = []
    for factor, features in SHAP_GROUPS.items():
        value = float(block_df.loc[block_df["feature"].isin(features), "mean_abs_shap"].sum())
        records.append({
            "factor": factor,
            "factor_ru": FACTOR_LABELS[factor],
            "shap_importance": value,
        })
    return pd.DataFrame(records)


def build_var_shap_alignment(
    fevd: pd.DataFrame,
    t23: pd.DataFrame,
    horizon: int = 12,
    block: str = "global",
) -> pd.DataFrame:
    fevd_imp = build_fevd_macro_importance(fevd, horizon=horizon)
    shap_imp = build_shap_macro_importance(t23, block=block)
    out = fevd_imp.merge(shap_imp[["factor", "shap_importance"]], on="factor", how="inner")
    fevd_sum = out["fevd_importance"].sum()
    shap_sum = out["shap_importance"].sum()
    out["fevd_share"] = out["fevd_importance"] / fevd_sum if fevd_sum else 0.0
    out["shap_share"] = out["shap_importance"] / shap_sum if shap_sum else 0.0
    return out


def spearman_alignment(alignment_df: pd.DataFrame) -> tuple[float, float]:
    rho, p_value = spearmanr(alignment_df["fevd_importance"], alignment_df["shap_importance"])
    return float(rho), float(p_value)


def fig_69_fevd_vs_shap_bar(alignment_df: pd.DataFrame, rho: float, p_value: float) -> None:
    df = alignment_df.copy()
    if "fevd_share" not in df.columns:
        total = df["fevd_importance"].sum()
        df["fevd_share"] = df["fevd_importance"] / total if total else 0.0
    if "shap_share" not in df.columns:
        total = df["shap_importance"].sum()
        df["shap_share"] = df["shap_importance"] / total if total else 0.0
    plot_df = df[["factor_ru", "fevd_share", "shap_share"]].melt(
        id_vars="factor_ru",
        var_name="Источник",
        value_name="Доля",
    )
    plot_df["Источник"] = plot_df["Источник"].map({
        "fevd_share": "FEVD VAR",
        "shap_share": "SHAP ML",
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="factor_ru", y="Доля", hue="Источник", ax=ax)
    ax.set_title(f"Сопоставление FEVD и SHAP по макро-факторам (ρ={rho:.2f}, p={p_value:.3f})")
    ax.set_xlabel("")
    ax.set_ylabel("Нормированная важность")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="")
    save_figure(fig, "fig_69_fevd_vs_shap_bar")


def fig_70_fevd_shap_scatter(alignment_df: pd.DataFrame, rho: float, p_value: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=alignment_df, x="fevd_importance", y="shap_importance", s=90, ax=ax)
    for _, row in alignment_df.iterrows():
        ax.annotate(row["factor_ru"], (row["fevd_importance"], row["shap_importance"]), xytext=(5, 5), textcoords="offset points")
    ax.set_title(f"FEVD vs SHAP: ρ Спирмена = {rho:.2f}, p = {p_value:.3f}")
    ax.set_xlabel("FEVD: вклад в дисперсию IMOEX")
    ax.set_ylabel("SHAP: mean |SHAP|")
    save_figure(fig, "fig_70_fevd_shap_scatter")


def main() -> None:
    setup_plot_style()
    fevd, t23 = load_inputs()
    alignment = build_var_shap_alignment(fevd, t23, horizon=12, block="global")
    rho, p_value = spearman_alignment(alignment)
    _ensure_dirs()
    out_path = os.path.join(TABLES, SUBFOLDER, "T24_var_shap_alignment.csv")
    alignment.assign(spearman_rho=rho, spearman_p_value=p_value).to_csv(out_path, index=False, encoding="utf-8-sig")
    fig_69_fevd_vs_shap_bar(alignment, rho, p_value)
    fig_70_fevd_shap_scatter(alignment, rho, p_value)
    log.info("Spearman FEVD vs SHAP: rho=%.4f, p=%.4f", rho, p_value)
    log.info("Сохранено: %s, shape=%s", out_path, alignment.shape)


if __name__ == "__main__":
    main()
