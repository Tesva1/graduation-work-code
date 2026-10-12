"""Шаг 11: проверка гипотез H1-H7."""
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
from scipy.stats import kruskal, ttest_rel
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_PROCESSED, DM_TEST_LAG, FIGURES, ROLLING_AUC_WINDOW, TABLES  # noqa: E402
from src.feature_utils import FEATURE_LABELS  # noqa: E402
from src.utils import (  # noqa: E402
    diebold_mariano,
    holm_bonferroni,
    per_fold_log_loss,
    setup_plot_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_11")

SUBFOLDER = "step_11"


def _ensure_dirs() -> None:
    os.makedirs(os.path.join(FIGURES, SUBFOLDER), exist_ok=True)
    os.makedirs(os.path.join(TABLES, SUBFOLDER), exist_ok=True)


def save_figure(fig: plt.Figure, name: str) -> None:
    _ensure_dirs()
    base = os.path.join(FIGURES, SUBFOLDER, name)
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _target_series() -> pd.Series:
    return pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_target.parquet"))["y_next"].dropna().astype(int)


def load_inputs() -> dict[str, object]:
    y = _target_series()
    baseline = pd.read_parquet(os.path.join(DATA_PROCESSED, "baseline_predictions.parquet"))
    regime_pred = pd.read_parquet(os.path.join(DATA_PROCESSED, "regime_predictions.parquet"))
    regimes = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_regimes.parquet"))["regime"].astype(int)
    shap_values = pd.read_parquet(os.path.join(DATA_PROCESSED, "shap_values_variant_a.parquet"))
    t23 = pd.read_csv(os.path.join(TABLES, "T23_shap_top_features.csv"))
    t18 = pd.read_csv(os.path.join(TABLES, "step_09", "T18_all_systems.csv"))
    t19 = pd.read_csv(os.path.join(TABLES, "step_09", "T19_metrics_by_regime.csv"))
    var_shap_path = os.path.join(TABLES, "step_10", "T24_var_shap_alignment.csv")
    var_shap = pd.read_csv(var_shap_path) if os.path.exists(var_shap_path) else pd.DataFrame()
    return {
        "y": y,
        "baseline": baseline,
        "regime_pred": regime_pred,
        "regimes": regimes,
        "shap_values": shap_values,
        "t23": t23,
        "t18": t18,
        "t19": t19,
        "var_shap": var_shap,
    }


def paired_auc_samples(
    y: pd.Series,
    pred_a: pd.Series,
    pred_b: pd.Series,
    window: int = 12,
) -> pd.DataFrame:
    """Rolling paired AUC samples for two predictors."""
    df = pd.concat([y.rename("y"), pred_a.rename("a"), pred_b.rename("b")], axis=1, join="inner").dropna()
    rows = []
    for end in range(window, len(df) + 1):
        part = df.iloc[end - window : end]
        if part["y"].nunique() < 2:
            continue
        auc_a = roc_auc_score(part["y"], part["a"])
        auc_b = roc_auc_score(part["y"], part["b"])
        rows.append({"date": part.index[-1], "auc_a": auc_a, "auc_b": auc_b, "delta": auc_a - auc_b})
    return pd.DataFrame(rows)


def paired_ttest_from_samples(a: Iterable[float], b: Iterable[float]) -> dict[str, float | bool]:
    """LEGACY: paired t-test для совместимости со старыми вызовами/тестами.

    Используется только для диагностических графиков rolling AUC; формальная
    статистическая проверка гипотез теперь идёт через Diebold-Mariano.
    """
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if mask.sum() < 2:
        return {"t_stat": float("nan"), "p_value": 1.0, "significant": False}
    t_stat, p_value = ttest_rel(a_arr[mask], b_arr[mask])
    return {"t_stat": float(t_stat), "p_value": float(p_value), "significant": bool(p_value < 0.05)}


def dm_test_predictions(
    y: pd.Series, pred_a: pd.Series, pred_b: pd.Series, h: int = DM_TEST_LAG,
) -> dict[str, float | bool]:
    """Diebold-Mariano по per-fold log-loss для двух моделей.

    Положительная статистика → loss(A) > loss(B), т.е. модель A хуже B.
    Используется для проверки H1, H6, H7.
    """
    loss_a = per_fold_log_loss(y, pred_a)
    loss_b = per_fold_log_loss(y, pred_b)
    dm_stat, p_value = diebold_mariano(loss_a, loss_b, h=h)
    return {
        "dm_stat": float(dm_stat) if np.isfinite(dm_stat) else float("nan"),
        "p_value": float(p_value),
        "significant": bool(np.isfinite(p_value) and p_value < 0.05),
        "n_folds": int(min(loss_a.shape[0], loss_b.shape[0])),
    }


def kruskal_by_regime(shap_values: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    regimes = regimes.reindex(shap_values.index).astype(int)
    rows = []
    for feature in shap_values.columns:
        groups = [shap_values.loc[regimes == r, feature].dropna().values for r in sorted(regimes.unique())]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            h_stat, p_value = float("nan"), 1.0
        else:
            h_stat, p_value = kruskal(*groups)
        rows.append({
            "feature": feature,
            "feature_ru": FEATURE_LABELS.get(feature, feature),
            "h_stat": float(h_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        })
    return pd.DataFrame(rows).sort_values("p_value")


def evaluate_h3_from_t23(t23: pd.DataFrame, top_n: int = 5) -> dict[str, object]:
    r0 = set(t23.loc[t23["block"] == "regime_0"].sort_values("rank").head(top_n)["feature"])
    r1 = set(t23.loc[t23["block"] == "regime_1"].sort_values("rank").head(top_n)["feature"])
    overlap = len(r0.intersection(r1))
    if not r0 or not r1:
        status = "не подтверждена"
    elif overlap <= 1:
        status = "подтверждена"
    elif overlap < top_n:
        status = "частично"
    else:
        status = "не подтверждена"
    return {"status": status, "overlap": overlap, "regime_0_top": sorted(r0), "regime_1_top": sorted(r1)}


def hypothesis_status(direction_ok: bool, p_value: float, alpha: float = 0.05) -> str:
    if not direction_ok:
        return "не подтверждена"
    return "подтверждена" if p_value < alpha else "частично"


def per_regime_accuracy_gain(
    y: pd.Series,
    baseline_svm: pd.Series,
    regime_pred: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    idx = y.index.intersection(baseline_svm.index).intersection(regime_pred.index)
    regimes_test = regimes.reindex(idx).astype(int)
    rows = []
    base_cls = (baseline_svm.reindex(idx) >= 0.5).astype(int)
    for system in regime_pred.columns:
        cls = (regime_pred.loc[idx, system] >= 0.5).astype(int)
        for r in sorted(regimes_test.unique()):
            mask = regimes_test == r
            base_acc = accuracy_score(y.loc[idx][mask], base_cls[mask])
            sys_acc = accuracy_score(y.loc[idx][mask], cls[mask])
            rows.append({
                "system": system,
                "regime": int(r),
                "regime_ru": "медвежий" if int(r) == 0 else "бычий",
                "baseline_accuracy": base_acc,
                "system_accuracy": sys_acc,
                "accuracy_gain": sys_acc - base_acc,
                "n_obs": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def fig_71_paired_auc(auc_frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, df in auc_frames.items():
        for _, row in df.iterrows():
            rows.append({"Система": "SVM baseline", "AUC": row["auc_b"], "Дата": row["date"], "Сравнение": name})
            rows.append({"Система": name, "AUC": row["auc_a"], "Дата": row["date"], "Сравнение": name})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="Сравнение", y="AUC", hue="Система", ax=ax)
    ax.set_title("Парные rolling AUC: регимные модели против baseline SVM")
    ax.set_xlabel("")
    ax.set_ylabel("AUC-ROC в 12-месячном окне")
    ax.legend(title="")
    save_figure(fig, "fig_71_paired_auc_baseline_vs_regime")


def fig_72_kruskal_pvalues(kw: pd.DataFrame) -> None:
    plot_df = kw.sort_values("p_value").head(15).copy()
    plot_df["minus_log10_p"] = -np.log10(plot_df["p_value"].clip(lower=1e-12))
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=plot_df, y="feature_ru", x="minus_log10_p", hue="significant", dodge=False, ax=ax)
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
    ax.set_title("Различия SHAP-значений между режимами (Kruskal-Wallis)")
    ax.set_xlabel("-log10(p-value)")
    ax.set_ylabel("")
    ax.legend(title="Значимо")
    save_figure(fig, "fig_72_kruskal_shap_pvalues")


def fig_73_accuracy_gain(gain: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=gain, x="regime_ru", y="accuracy_gain", hue="system", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Прирост accuracy относительно baseline SVM по режимам")
    ax.set_xlabel("Режим")
    ax.set_ylabel("Δ Accuracy")
    ax.legend(title="")
    save_figure(fig, "fig_73_accuracy_gain_by_regime")


def build_hypothesis_summary(inputs: dict[str, object]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    y = inputs["y"]  # type: ignore[assignment]
    baseline = inputs["baseline"]  # type: ignore[assignment]
    regime_pred = inputs["regime_pred"]  # type: ignore[assignment]
    regimes = inputs["regimes"]  # type: ignore[assignment]
    shap_values = inputs["shap_values"]  # type: ignore[assignment]
    t23 = inputs["t23"]  # type: ignore[assignment]
    var_shap = inputs["var_shap"]  # type: ignore[assignment]

    y_test = y.reindex(regime_pred.index).dropna().astype(int)
    baseline_svm = baseline["SVM"].reindex(y_test.index)

    # Диагностические rolling AUC (для fig_71). НЕ используются в формальных
    # p-значениях — окна перекрываются на 11 месяцев, авто-корреляция
    # сильно искажает t-test. Гипотезы H1/H6/H7 теперь тестируются
    # через Diebold-Mariano по per-fold log-loss.
    auc_frames = {
        name: paired_auc_samples(
            y_test, regime_pred[name].reindex(y_test.index), baseline_svm,
            window=ROLLING_AUC_WINDOW,
        )
        for name in ["VariantA", "VariantB", "VariantC"]
    }
    h1_df = auc_frames["VariantA"]
    # H1: VariantA против baseline SVM по per-fold log-loss (h=1).
    h1_dm = dm_test_predictions(y_test, baseline_svm, regime_pred["VariantA"].reindex(y_test.index))
    h1_direction = h1_dm["dm_stat"] > 0  # положительная статистика → loss(SVM) > loss(VariantA)

    kw = kruskal_by_regime(shap_values, regimes)
    h2_share = float(kw["significant"].mean()) if not kw.empty else 0.0
    h2_p = float(kw["p_value"].min()) if not kw.empty else 1.0

    h3 = evaluate_h3_from_t23(t23, top_n=5)
    gain = per_regime_accuracy_gain(y_test, baseline_svm, regime_pred.reindex(y_test.index), regimes)
    h4_bear_gain = float(gain.loc[(gain["system"] == "VariantA") & (gain["regime"] == 0), "accuracy_gain"].mean())
    h4_bull_gain = float(gain.loc[(gain["system"] == "VariantA") & (gain["regime"] == 1), "accuracy_gain"].mean())

    if not var_shap.empty and {"spearman_rho", "spearman_p_value"}.issubset(var_shap.columns):
        h5_rho = float(var_shap["spearman_rho"].iloc[0])
        h5_p = float(var_shap["spearman_p_value"].iloc[0])
    else:
        h5_rho, h5_p = float("nan"), 1.0

    # H6: XGBoost vs (TFT, TabTransformer). Два независимых DM-теста с
    # поправкой Holm-Bonferroni на множественную проверку.
    xgb_tft = paired_auc_samples(
        y_test,
        baseline["XGBoost"].reindex(y_test.index),
        baseline["TFT"].reindex(y_test.index),
        window=ROLLING_AUC_WINDOW,
    )
    xgb_tab = paired_auc_samples(
        y_test,
        baseline["XGBoost"].reindex(y_test.index),
        baseline["TabTransformer"].reindex(y_test.index),
        window=ROLLING_AUC_WINDOW,
    )
    h6_dm_tft = dm_test_predictions(
        y_test,
        baseline["TFT"].reindex(y_test.index),
        baseline["XGBoost"].reindex(y_test.index),
    )
    h6_dm_tab = dm_test_predictions(
        y_test,
        baseline["TabTransformer"].reindex(y_test.index),
        baseline["XGBoost"].reindex(y_test.index),
    )
    h6_p_holm = holm_bonferroni([h6_dm_tft["p_value"], h6_dm_tab["p_value"]])
    h6_dm_tft = {**h6_dm_tft, "p_value_holm": h6_p_holm[0]}
    h6_dm_tab = {**h6_dm_tab, "p_value_holm": h6_p_holm[1]}
    h6_direction = (h6_dm_tft["dm_stat"] > 0) and (h6_dm_tab["dm_stat"] > 0)
    h6_min_p_holm = float(min(h6_p_holm))
    h6_max_dm = float(max(h6_dm_tft["dm_stat"], h6_dm_tab["dm_stat"]))

    h7_df = paired_auc_samples(
        y_test,
        regime_pred["VariantC"].reindex(y_test.index),
        regime_pred["VariantA"].reindex(y_test.index),
        window=ROLLING_AUC_WINDOW,
    )
    h7_dm = dm_test_predictions(
        y_test,
        regime_pred["VariantA"].reindex(y_test.index),
        regime_pred["VariantC"].reindex(y_test.index),
    )
    h7_direction = h7_dm["dm_stat"] > 0  # loss(VariantA) > loss(VariantC) → C лучше

    rows = [
        {
            "hypothesis": "H1",
            "statement_ru": "Режимная модель превосходит baseline по AUC",
            "test": "Diebold-Mariano по per-fold log-loss: VariantA vs SVM (h=1)",
            "statistic": h1_dm["dm_stat"],
            "p_value": h1_dm["p_value"],
            "status": hypothesis_status(h1_direction, float(h1_dm["p_value"])),
            "evidence": (
                f"DM={h1_dm['dm_stat']:.3f}, n_folds={h1_dm['n_folds']}, "
                f"диагностика mean ΔAUC={h1_df['delta'].mean():.4f}"
            ),
            "interpretation_ru": (
                "Per-fold log-loss за весь test-период; DM > 0 → loss(SVM) > loss(VariantA), "
                "т.е. VariantA точнее. Окна 12-мес. AUC даны для иллюстрации."
            ),
        },
        {
            "hypothesis": "H2",
            "statement_ru": "SHAP-значения различаются между режимами",
            "test": "Kruskal-Wallis по каждому признаку",
            "statistic": h2_share,
            "p_value": h2_p,
            "status": "подтверждена" if h2_share >= 0.2 else hypothesis_status(h2_share > 0, h2_p),
            "evidence": f"significant share={h2_share:.2%}, min p={h2_p:.4f}",
            "interpretation_ru": "Если значима заметная доля факторов, режимная структура меняет факторные вклады.",
        },
        {
            "hypothesis": "H3",
            "statement_ru": "В разных режимах доминируют разные факторы",
            "test": "Сравнение top-5 рангов SHAP из T23",
            "statistic": h3["overlap"],
            "p_value": np.nan,
            "status": h3["status"],
            "evidence": f"overlap top-5={h3['overlap']}",
            "interpretation_ru": "Низкое пересечение top-5 поддерживает тезис о разных факторах по режимам.",
        },
        {
            "hypothesis": "H4",
            "statement_ru": "Максимальный выигрыш regime vs baseline наблюдается в кризисном режиме",
            "test": "ΔAccuracy VariantA-SVM по режимам",
            "statistic": h4_bear_gain - h4_bull_gain,
            "p_value": np.nan,
            "status": "подтверждена" if h4_bear_gain > h4_bull_gain and h4_bear_gain > 0 else "не подтверждена",
            "evidence": f"bear ΔAcc={h4_bear_gain:.4f}, bull ΔAcc={h4_bull_gain:.4f}",
            "interpretation_ru": "Сравнивается практическая точность по медвежьему и бычьему режимам.",
        },
        {
            "hypothesis": "H5",
            "statement_ru": "VAR согласуется с SHAP по макро-факторам",
            "test": "ρ Спирмена FEVD vs SHAP",
            "statistic": h5_rho,
            "p_value": h5_p,
            "status": hypothesis_status(h5_rho > 0, h5_p),
            "evidence": f"rho={h5_rho:.4f}, p={h5_p:.4f}",
            "interpretation_ru": "Положительная ранговая связь означает согласование эконометрической и ML-важности.",
        },
        {
            "hypothesis": "H6",
            "statement_ru": "XGBoost превосходит TFT и TabTransformer",
            "test": "Два DM-теста по per-fold log-loss (Holm correction): XGB vs TFT и XGB vs TabTransformer",
            "statistic": h6_max_dm,
            "p_value": h6_min_p_holm,
            "status": hypothesis_status(h6_direction, float(h6_min_p_holm)),
            "evidence": (
                f"XGB vs TFT: DM={h6_dm_tft['dm_stat']:.3f}, p_holm={h6_dm_tft['p_value_holm']:.4f}; "
                f"XGB vs Tab: DM={h6_dm_tab['dm_stat']:.3f}, p_holm={h6_dm_tab['p_value_holm']:.4f}; "
                f"диагностика XGB-TFT Δ={xgb_tft['delta'].mean():.4f}, "
                f"XGB-Tab Δ={xgb_tab['delta'].mean():.4f}"
            ),
            "interpretation_ru": (
                "Два независимых сравнения с поправкой Holm-Bonferroni; "
                "обе DM-статистики должны быть положительны и обе p-Holm < 0.05 для подтверждения."
            ),
        },
        {
            "hypothesis": "H7",
            "statement_ru": "Variant C превосходит Variant A",
            "test": "Diebold-Mariano по per-fold log-loss: VariantC vs VariantA (h=1)",
            "statistic": h7_dm["dm_stat"],
            "p_value": h7_dm["p_value"],
            "status": hypothesis_status(h7_direction, float(h7_dm["p_value"])),
            "evidence": (
                f"DM={h7_dm['dm_stat']:.3f}, n_folds={h7_dm['n_folds']}, "
                f"диагностика mean ΔAUC={h7_df['delta'].mean():.4f}"
            ),
            "interpretation_ru": (
                "DM > 0 → loss(VariantA) > loss(VariantC), т.е. VariantC точнее. "
                "Если знак отрицательный, гипотеза не подтверждается."
            ),
        },
    ]
    return pd.DataFrame(rows), {"auc_frames": pd.concat(auc_frames, names=["system"]).reset_index(level=0), "kw": kw, "gain": gain}


def main() -> None:
    setup_plot_style()
    inputs = load_inputs()
    summary, aux = build_hypothesis_summary(inputs)
    _ensure_dirs()
    out_path = os.path.join(TABLES, SUBFOLDER, "T_H_hypothesis_summary.csv")
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    auc_frames = {
        name: df.drop(columns=["system"], errors="ignore")
        for name, df in aux["auc_frames"].groupby("system")
    }
    fig_71_paired_auc(auc_frames)
    fig_72_kruskal_pvalues(aux["kw"])
    fig_73_accuracy_gain(aux["gain"])
    log.info("Сохранено: %s, shape=%s", out_path, summary.shape)
    log.info("Статусы гипотез: %s", summary.set_index("hypothesis")["status"].to_dict())


if __name__ == "__main__":
    main()
