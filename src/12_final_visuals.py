"""Шаг 12: итоговые визуализации исследования (fig_74-76)."""
from __future__ import annotations

import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_PROCESSED, FIGURES, TABLES  # noqa: E402
from src.utils import setup_plot_style  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_12")

SUBFOLDER = "step_12"


def _ensure_dirs() -> None:
    os.makedirs(os.path.join(FIGURES, SUBFOLDER), exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> None:
    _ensure_dirs()
    base = os.path.join(FIGURES, SUBFOLDER, name)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_factor_heatmap_matrix(t23: pd.DataFrame) -> pd.DataFrame:
    block_map = {"global": "Глобально", "regime_0": "Медвежий режим", "regime_1": "Бычий режим"}
    df = t23[t23["block"].isin(block_map)].copy()
    df["column"] = df["block"].map(block_map)
    matrix = df.pivot_table(index="feature_ru", columns="column", values="mean_abs_shap", aggfunc="sum").fillna(0)
    ordered_cols = ["Глобально", "Медвежий режим", "Бычий режим"]
    for col in ordered_cols:
        if col not in matrix:
            matrix[col] = 0.0
    matrix = matrix[ordered_cols]
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
    return matrix


def build_timeline_events() -> list[dict[str, str]]:
    return [
        {"date": "2015-01", "label": "Начало выборки"},
        {"date": "2020-03", "label": "COVID-19"},
        {"date": "2022-02", "label": "Начало СВО"},
        {"date": "2023-09", "label": "Высокие ставки"},
        {"date": "2025-12", "label": "Конец выборки"},
    ]


def fig_74_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    nodes = [
        ("Данные\nIMOEX, Urals, USD/RUB,\nставка ЦБ, RGBI", 0.05, 0.62),
        ("Признаки\n25 monthly factors\nбез look-ahead", 0.25, 0.62),
        ("VAR/SVAR\nFEVD, IRF,\nГрейнджер", 0.45, 0.78),
        ("HMM\nскрытые режимы\nрынка", 0.45, 0.46),
        ("Baseline ML\n8 моделей\nwalk-forward", 0.65, 0.62),
        ("Regime ML\nVariant A/B/C", 0.82, 0.62),
        ("SHAP\nфакторы по\nрежимам", 0.82, 0.30),
    ]
    for text, x, y in nodes:
        patch = FancyBboxPatch(
            (x, y), 0.14, 0.14,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#F2F4F8",
            edgecolor="#2F4B7C",
            linewidth=1.4,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + 0.07, y + 0.07, text, ha="center", va="center", transform=ax.transAxes, fontsize=11)
    arrows = [(0.19, 0.69, 0.25, 0.69), (0.39, 0.69, 0.45, 0.84), (0.39, 0.69, 0.45, 0.53), (0.59, 0.53, 0.65, 0.69), (0.79, 0.69, 0.82, 0.69), (0.89, 0.62, 0.89, 0.44), (0.59, 0.84, 0.82, 0.38)]
    for x1, y1, x2, y2 in arrows:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), transform=ax.transAxes, arrowstyle="->", mutation_scale=14, linewidth=1.2, color="#333333"))
    ax.set_title("Архитектура исследования: от данных к режимным моделям и SHAP", fontsize=15)
    save_figure(fig, "fig_74_research_architecture")


def fig_75_timeline() -> None:
    regimes = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_regimes.parquet"))["regime"].astype(int)
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = {0: "#E45756", 1: "#54A24B", 2: "#F2CF5B"}
    for date, regime in regimes.items():
        ax.axvspan(date, date + pd.offsets.MonthEnd(1), color=colors.get(int(regime), "0.8"), alpha=0.45)
    ax.plot(regimes.index, np.ones(len(regimes)), alpha=0)
    for event in build_timeline_events():
        date = pd.Timestamp(event["date"])
        ax.axvline(date, color="black", linestyle="--", linewidth=1)
        ax.text(date, 1.03, event["label"], rotation=90, ha="right", va="bottom", fontsize=9)
    ax.set_ylim(0.95, 1.20)
    ax.set_yticks([])
    ax.set_title("Timeline 2015–2025: ключевые события и скрытые рыночные режимы")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_figure(fig, "fig_75_regime_timeline_events")


def fig_76_factor_map() -> None:
    t23 = pd.read_csv(os.path.join(TABLES, "T23_shap_top_features.csv"))
    matrix = build_factor_heatmap_matrix(t23).head(15)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "Mean |SHAP|"})
    ax.set_title("Итоговая карта факторов: SHAP-важность по режимам")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, "fig_76_final_factor_map")


def main() -> None:
    setup_plot_style()
    fig_74_architecture()
    fig_75_timeline()
    fig_76_factor_map()
    log.info("Сохранены fig_74-fig_76 в %s", os.path.join(FIGURES, SUBFOLDER))


if __name__ == "__main__":
    main()
