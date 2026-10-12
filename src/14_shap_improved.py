"""Шаг 14: SHAP-интерпретация улучшенной модели.

Если winner является блендом/stacking поверх прогнозов, интерпретация идёт по
model-level сигналам. Дополнительно всегда сохраняется FeatureStack analysis,
где признаки = 25 рыночных факторов + 11 прогнозов моделей.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_PROCESSED, FIGURES, TABLES  # noqa: E402
from src.feature_utils import FEATURE_LABELS  # noqa: E402
from src.utils import setup_plot_style  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_14")

SUBFOLDER = "step_14"


def _load_step13():
    path = os.path.join(PROJECT_ROOT, "src", "13_model_improvement.py")
    spec = importlib.util.spec_from_file_location("model_improvement_13", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def select_winner(t24: pd.DataFrame, metric: str = "AUC-ROC") -> pd.Series:
    df = t24.loc[t24["Система"] != "BuyAndHold"].dropna(subset=[metric]).copy()
    return df.loc[df[metric].idxmax()]


def compute_delta_shap(base: pd.DataFrame, improved: pd.DataFrame) -> pd.Series:
    common = base.columns.intersection(improved.columns)
    if len(common) == 0:
        return pd.Series(dtype=float)
    return improved[common].abs().mean() - base[common].abs().mean()


def compute_t26(
    importance: pd.Series,
    t23: pd.DataFrame,
    model_name: str,
    source: str,
    top_n: int = 20,
) -> pd.DataFrame:
    base_ranks = t23.loc[t23["block"] == "global"].set_index("feature")["rank"].to_dict() if "block" in t23 else {}
    rows = []
    for rank, (feature, value) in enumerate(importance.sort_values(ascending=False).head(top_n).items(), start=1):
        base_rank = base_ranks.get(feature, np.nan)
        rows.append({
            "model": model_name,
            "source": source,
            "rank": rank,
            "feature": feature,
            "feature_ru": FEATURE_LABELS.get(feature, feature),
            "importance": float(value),
            "rank_t23": base_rank,
            "rank_change_vs_t23": float(base_rank - rank) if pd.notna(base_rank) else np.nan,
        })
    return pd.DataFrame(rows)


def _linear_shap_from_coefficients(X: pd.DataFrame, coef: pd.Series) -> pd.DataFrame:
    coef = coef.reindex(X.columns).fillna(0)
    return (X - X.mean(axis=0)).mul(coef, axis=1)


def _walk_forward_blend_shap(
    blend_inputs: pd.DataFrame, weights_per_row: pd.DataFrame,
) -> pd.DataFrame:
    """SHAP по сигналу blend через per-row decomposition.

    Для каждой строки t и модели m:
        contribution[t, m] = weights_per_row[t, m] * (blend_inputs[t, m] - mean_blend_inputs[m])

    Веса уже получены walk-forward в step 13, в этой функции никакого нового
    обучения / поиска нет — только разложение готового сигнала.
    """
    blend_inputs = blend_inputs.reindex(weights_per_row.index)
    centered = blend_inputs - blend_inputs.mean(axis=0)
    return centered * weights_per_row.reindex(columns=blend_inputs.columns).fillna(0)


def winner_model_level_shap(winner: str) -> tuple[pd.DataFrame, pd.Series, str]:
    """SHAP победителя, не вводящий look-ahead.

    Для blend-победителей используются ТОЛЬКО walk-forward веса, сохранённые
    шагом 13 в data/processed/improved_blend_weights_*.parquet. Никакого
    повторного in-sample подбора весов внутри шага 14 не выполняется.
    """
    step13 = _load_step13()
    preds, _features, _y, _returns, _regimes, _walk = step13.align_prediction_frame()
    if winner == "Blend_optuna":
        weights_path = os.path.join(DATA_PROCESSED, "improved_blend_weights_optuna.parquet")
        weights_per_row = pd.read_parquet(weights_path)
        blend_inputs = preds[list(weights_per_row.columns)]
        shap_values = _walk_forward_blend_shap(blend_inputs, weights_per_row)
        importance = shap_values.abs().mean().sort_values(ascending=False)
        return shap_values, importance, "model_level_blend_walk_forward"
    if winner == "Blend_softmax_auc":
        weights_path = os.path.join(DATA_PROCESSED, "improved_blend_weights_softmax.parquet")
        weights_per_row = pd.read_parquet(weights_path)
        blend_inputs = preds[list(weights_per_row.columns)]
        shap_values = _walk_forward_blend_shap(blend_inputs, weights_per_row)
        importance = shap_values.abs().mean().sort_values(ascending=False)
        return shap_values, importance, "model_level_blend_walk_forward"
    if winner == "FeatureStack_LogReg":
        return feature_stack_shap()
    if winner in preds.columns:
        base = pd.read_parquet(os.path.join(DATA_PROCESSED, "shap_values_variant_a.parquet"))
        return base, base.abs().mean().sort_values(ascending=False), "reused_variant_a_shap" if winner == "VariantA" else "single_prediction_signal"
    improved = pd.read_parquet(os.path.join(DATA_PROCESSED, "improved_model_predictions.parquet"))
    X = improved[[winner]]
    shap_values = _linear_shap_from_coefficients(X, pd.Series({winner: 1.0}))
    return shap_values, shap_values.abs().mean().sort_values(ascending=False), "single_improved_prediction_signal"


def feature_stack_shap() -> tuple[pd.DataFrame, pd.Series, str]:
    step13 = _load_step13()
    preds, features, _y, _returns, _regimes, _walk = step13.align_prediction_frame()
    coef_table = pd.read_csv(os.path.join(TABLES, "step_13", "T26_feature_stack_coefficients.csv"))
    fs = coef_table.loc[coef_table["model"] == "FeatureStack_LogReg"].copy()
    X_design = step13._stacking_design(preds, features)
    coef = fs.set_index("design_feature")["coefficient"]
    shap_values = _linear_shap_from_coefficients(X_design, coef)
    importance = shap_values.abs().mean().sort_values(ascending=False)
    return shap_values, importance, "feature_plus_model_stack"


def fig_83_importance(t26: pd.DataFrame) -> None:
    top = t26.loc[t26["model"] == t26["model"].iloc[0]].head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, x="importance", y="feature_ru", ax=ax, color="#4C78A8")
    ax.set_title("SHAP-важность финальной улучшенной модели")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("")
    save_figure(fig, "fig_83_final_model_importance")


def fig_84_delta(delta: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if delta.empty:
        ax.text(0.5, 0.5, "Нет общих original-feature SHAP колонок для ΔSHAP", ha="center", va="center")
        ax.axis("off")
    else:
        top = delta.reindex(delta.abs().sort_values(ascending=False).head(15).index).sort_values()
        colors = ["#E45756" if v < 0 else "#54A24B" for v in top.values]
        ax.barh([FEATURE_LABELS.get(f, f) for f in top.index], top.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Δ mean |SHAP|")
        ax.set_title("ΔSHAP улучшенной модели относительно VariantA")
    save_figure(fig, "fig_84_delta_shap_vs_variant_a")


def fig_85_by_regime(shap_values: pd.DataFrame) -> None:
    regimes = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_regimes.parquet"))["regime"].astype(int).reindex(shap_values.index)
    rows = []
    for r in sorted(regimes.dropna().unique()):
        imp = shap_values.loc[regimes == r].abs().mean().sort_values(ascending=False).head(10)
        for feature, value in imp.items():
            rows.append({"regime": "медвежий" if int(r) == 0 else "бычий", "feature": feature, "importance": value})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 7))
    if plot_df.empty:
        ax.axis("off")
    else:
        sns.barplot(data=plot_df, x="importance", y="feature", hue="regime", ax=ax)
        ax.set_title("SHAP улучшенной модели по режимам")
        ax.set_xlabel("Mean |SHAP|")
        ax.set_ylabel("")
    save_figure(fig, "fig_85_final_model_by_regime")


def fig_86_meta_importance(feature_stack_t26: pd.DataFrame) -> None:
    top = feature_stack_t26.head(15).copy()
    top["group"] = np.where(top["feature"].str.startswith("model__") | top["feature"].isin(["VariantA", "VariantB", "VariantC", "SVM"]), "model signal", "market feature")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, x="importance", y="feature_ru", hue="source", dodge=False, ax=ax)
    ax.set_title("FeatureStack: вклад market features и model predictions")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("")
    save_figure(fig, "fig_86_meta_model_importance")


def fig_87_waterfall(shap_values: pd.DataFrame) -> None:
    date = pd.Timestamp("2022-02-28")
    if date not in shap_values.index:
        date = shap_values.index[np.argmin(np.abs((shap_values.index - date).days))]
    row = shap_values.loc[date].sort_values(key=lambda s: s.abs(), ascending=False).head(12).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#E45756" if v < 0 else "#54A24B" for v in row.values]
    ax.barh(row.index, row.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Локальное объяснение улучшенной модели: {date.strftime('%Y-%m')}")
    ax.set_xlabel("SHAP-значение")
    save_figure(fig, "fig_87_final_waterfall_example")


def main() -> None:
    setup_plot_style()
    t24 = pd.read_csv(os.path.join(TABLES, "step_13", "T24_model_improvements.csv"))
    t23 = pd.read_csv(os.path.join(TABLES, "T23_shap_top_features.csv"))
    winner = select_winner(t24, metric="AUC-ROC")
    winner_name = str(winner["Система"])
    winner_shap, winner_importance, winner_source = winner_model_level_shap(winner_name)
    fs_shap, fs_importance, fs_source = feature_stack_shap()
    base_shap = pd.read_parquet(os.path.join(DATA_PROCESSED, "shap_values_variant_a.parquet"))
    delta = compute_delta_shap(base_shap, winner_shap)

    t26_winner = compute_t26(winner_importance, t23, winner_name, winner_source, top_n=20)
    t26_feature_stack = compute_t26(fs_importance, t23, "FeatureStack_LogReg", fs_source, top_n=25)
    t26 = pd.concat([t26_winner, t26_feature_stack], ignore_index=True)
    _ensure_dirs()
    winner_shap.to_parquet(os.path.join(DATA_PROCESSED, "shap_values_improved.parquet"))
    fs_shap.to_parquet(os.path.join(DATA_PROCESSED, "shap_values_feature_stack.parquet"))
    t26.to_csv(os.path.join(TABLES, SUBFOLDER, "T26_final_model_shap_top_features.csv"), index=False, encoding="utf-8-sig")

    fig_83_importance(t26_winner)
    fig_84_delta(delta)
    fig_85_by_regime(winner_shap)
    fig_86_meta_importance(t26_feature_stack)
    fig_87_waterfall(winner_shap)
    log.info("Winner=%s, source=%s", winner_name, winner_source)
    log.info("Сохранены T26 и fig_83-87")


if __name__ == "__main__":
    main()
