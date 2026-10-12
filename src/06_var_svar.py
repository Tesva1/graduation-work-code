"""Шаг 6: VAR/SVAR анализ — выбор лагов, тест Грейнджера, IRF, FEVD.

Вход:  data/processed/monthly_features.parquet
Выход: figures/step_06/ (7 PNG), tables/ (T7–T10 CSV)

Запуск: python src/06_var_svar.py
"""
from __future__ import annotations

import logging
import os
import sys
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR, VARResults

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PROCESSED, SVAR_ORDER, VAR_MAX_LAGS
from src.utils import save_fig, save_table, setup_plot_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_06")

_SUBFOLDER = "step_06"

_VAR_LABELS: dict[str, str] = {
    "imoex_ret": "Доходность IMOEX",
    "urals_ret": "Доходность Urals",
    "usdrub_change": "Изменение USD/RUB",
    "key_rate_change": "Изменение ставки ЦБ",
    "imoex_vol": "Волатильность IMOEX",
}


class VARFitResult(NamedTuple):
    """Результат подбора VAR-модели."""

    result: VARResults
    best_lag: int
    lag_df: pd.DataFrame


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------


def load_var_data() -> pd.DataFrame:
    """Загружает месячные признаки и формирует датасет для VAR-модели.

    Returns:
        DataFrame с колонками, соответствующими SVAR_ORDER, без NaN.
    """
    path = os.path.join(DATA_PROCESSED, "monthly_features.parquet")
    df_raw = pd.read_parquet(path)

    cols_map = {
        "imoex_ret_1m": "imoex_ret",
        "imoex_vol_20d": "imoex_vol",
        "urals_ret_1m": "urals_ret",
    }
    df = df_raw[
        ["urals_ret_1m", "usdrub_change", "key_rate_change", "imoex_vol_20d", "imoex_ret_1m"]
    ].rename(columns=cols_map)

    df = df.dropna()
    log.info("VAR датасет: shape=%s, период %s – %s", df.shape, df.index.min().date(), df.index.max().date())
    return df


# ---------------------------------------------------------------------------
# Подбор и оценка VAR
# ---------------------------------------------------------------------------


def fit_var(data: pd.DataFrame) -> VARFitResult:
    """Подбирает оптимальное число лагов по BIC и оценивает VAR(p).

    Args:
        data: DataFrame с рядами для VAR-модели.

    Returns:
        Именованный кортеж с результатом, оптимальным лагом и таблицей критериев.
    """
    model = VAR(data)
    rows: list[dict[str, float | int]] = []

    for lag in range(1, VAR_MAX_LAGS + 1):
        try:
            r = model.fit(lag)
            rows.append({"lag": lag, "AIC": r.aic, "BIC": r.bic, "HQIC": r.hqic})
        except Exception as exc:
            log.warning("Не удалось оценить VAR(%d): %s", lag, exc)
            break

    lag_df = pd.DataFrame(rows)
    best_lag = int(lag_df.loc[lag_df["BIC"].idxmin(), "lag"])
    log.info("Оптимальное число лагов по BIC: %d", best_lag)

    result = model.fit(best_lag)
    log.info("VAR(%d) оценён. AIC=%.2f, BIC=%.2f", best_lag, result.aic, result.bic)
    return VARFitResult(result=result, best_lag=best_lag, lag_df=lag_df)


# ---------------------------------------------------------------------------
# Таблицы
# ---------------------------------------------------------------------------


def table_T7_lag_selection(lag_df: pd.DataFrame) -> None:
    """Сохраняет таблицу критериев выбора лагов VAR.

    Args:
        lag_df: DataFrame с колонками lag, AIC, BIC, HQIC.
    """
    out = lag_df.copy()
    for col in ["AIC", "BIC", "HQIC"]:
        out[col] = out[col].round(2)
    out["lag"] = out["lag"].astype(int)
    save_table(out.set_index("lag"), "T7_var_lag_selection")
    log.info("T7: таблица критериев лагов сохранена (%d строк)", len(out))


def table_T8_granger(result: VARResults, data: pd.DataFrame) -> pd.DataFrame:
    """Тест Грейнджера: каждая переменная → imoex_ret.

    Args:
        result: Оценённая VAR-модель.
        data: DataFrame с исходными рядами.

    Returns:
        DataFrame с результатами тестов Грейнджера.
    """
    target = "imoex_ret"
    rows: list[dict[str, object]] = []

    for cause in [c for c in data.columns if c != target]:
        try:
            tests = grangercausalitytests(data[[target, cause]], maxlag=result.k_ar, verbose=False)
            pval = tests[result.k_ar][0]["ssr_ftest"][1]
        except Exception as exc:
            log.warning("Тест Грейнджера (%s → %s) не удался: %s", cause, target, exc)
            pval = float("nan")

        label = _VAR_LABELS.get(cause, cause)
        rows.append({
            "Переменная": label,
            "Код": cause,
            "F-тест p-value": round(pval, 4) if not np.isnan(pval) else pval,
            "Значимо (5%)": "Да" if (not np.isnan(pval) and pval < 0.05) else "Нет",
        })
        log.info("Грейнджер %s → imoex_ret: p=%.4f", cause, pval if not np.isnan(pval) else -1)

    df = pd.DataFrame(rows)
    save_table(df.set_index("Код"), "T8_granger_causality")
    return df


def table_T9_fevd(result: VARResults) -> None:
    """Сохраняет декомпозицию дисперсии FEVD для уравнения imoex_ret.

    Args:
        result: Оценённая VAR-модель.
    """
    fevd = result.fevd(12)
    imoex_idx = list(result.names).index("imoex_ret")
    rows: list[dict[str, object]] = []

    for h in [1, 3, 6, 12]:
        row: dict[str, object] = {"Горизонт": h}
        for j, name in enumerate(result.names):
            row[_VAR_LABELS.get(name, name)] = round(float(fevd.decomp[imoex_idx, h - 1, j]), 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Горизонт")
    save_table(df, "T9_fevd")
    log.info("T9: FEVD сохранена для горизонтов 1, 3, 6, 12")


def table_T10_diagnostics(result: VARResults) -> None:
    """Диагностика VAR: Durbin-Watson, ARCH-LM, стабильность модели.

    Args:
        result: Оценённая VAR-модель.
    """
    rows: list[dict[str, object]] = []

    for eq_name in result.names:
        resid = result.resid[eq_name].values
        dw = float(durbin_watson(resid))
        try:
            arch_stat, arch_pval, *_ = het_arch(resid, nlags=5)
        except Exception:
            arch_stat, arch_pval = float("nan"), float("nan")

        rows.append({
            "Уравнение": _VAR_LABELS.get(eq_name, eq_name),
            "Durbin-Watson": round(dw, 3),
            "ARCH-LM статистика": round(float(arch_stat), 3) if not np.isnan(arch_stat) else arch_stat,
            "ARCH-LM p-value": round(float(arch_pval), 4) if not np.isnan(arch_pval) else arch_pval,
            "Гетероскедастичность (5%)": "Да" if (not np.isnan(arch_pval) and arch_pval < 0.05) else "Нет",
        })

    roots = result.roots
    max_root = float(np.max(np.abs(roots)))
    stable = max_root < 1.0
    log.info(
        "VAR стабильность: %s (max |корень| = %.4f)",
        "стабилен" if stable else "НЕСТАБИЛЕН",
        max_root,
    )

    df = pd.DataFrame(rows).set_index("Уравнение")
    df.attrs["stable"] = stable
    df.attrs["max_root"] = max_root
    save_table(df, "T10_var_diagnostics")
    log.info("T10: диагностика VAR сохранена")


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------


def fig_17_lag_selection(lag_df: pd.DataFrame) -> None:
    """График критериев AIC и BIC для выбора числа лагов VAR.

    Args:
        lag_df: DataFrame с колонками lag, AIC, BIC, HQIC.
    """
    best_bic_lag = int(lag_df.loc[lag_df["BIC"].idxmin(), "lag"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lag_df["lag"], lag_df["AIC"], marker="o", label="AIC", linewidth=1.5)
    ax.plot(lag_df["lag"], lag_df["BIC"], marker="s", label="BIC", linewidth=1.5)
    ax.plot(lag_df["lag"], lag_df["HQIC"], marker="^", label="HQIC", linewidth=1.5, linestyle="--")
    ax.axvline(best_bic_lag, color="red", linestyle=":", linewidth=1.2, label=f"Оптимум BIC (lag={best_bic_lag})")
    ax.set_title("Критерии информации VAR-модели (AIC, BIC, HQIC)")
    ax.set_xlabel("Число лагов")
    ax.set_ylabel("Значение критерия")
    ax.set_xticks(lag_df["lag"].tolist())
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_17_lag_selection", subfolder=_SUBFOLDER)
    log.info("fig_17 сохранён")


def _get_irf_values_and_stderr(
    result: VARResults, periods: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Вычисляет IRF и стандартные ошибки методом аналитической аппроксимации.

    Args:
        result: Оценённая VAR-модель.
        periods: Горизонт прогноза в месяцах.

    Returns:
        Кортеж (irfs, stderr), каждый shape (periods+1, k, k).
    """
    irf_obj = result.irf(periods)
    irfs = irf_obj.irfs
    try:
        stderr = irf_obj.stderr(orth=False)
    except Exception:
        log.warning("Не удалось вычислить stderr IRF; использую нулевые")
        stderr = np.zeros_like(irfs)
    return irfs, stderr


def fig_18_irf_imoex_shock(result: VARResults) -> None:
    """IRF: реакция всех 5 переменных на шок доходности IMOEX.

    Args:
        result: Оценённая VAR-модель.
    """
    irfs, _ = _get_irf_values_and_stderr(result, 12)
    shock_idx = list(result.names).index("imoex_ret")
    horizons = np.arange(irfs.shape[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    for resp_idx, name in enumerate(result.names):
        label = _VAR_LABELS.get(name, name)
        ax.plot(horizons, irfs[:, resp_idx, shock_idx], marker="o", markersize=3, label=label, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title("IRF: реакция переменных на шок доходности IMOEX")
    ax.set_xlabel("Горизонт (месяцев)")
    ax.set_ylabel("Отклик")
    ax.set_xticks(horizons)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig_18_irf_imoex_shock", subfolder=_SUBFOLDER)
    log.info("fig_18 сохранён")


def _plot_single_irf_with_ci(
    ax: plt.Axes,
    irfs: np.ndarray,
    stderr: np.ndarray,
    resp_idx: int,
    shock_idx: int,
    title: str,
    ylabel: str = "Отклик",
) -> None:
    """Рисует IRF с 95% доверительным интервалом на заданных осях.

    Args:
        ax: Оси для отрисовки.
        irfs: Массив IRF shape (periods+1, k, k).
        stderr: Стандартные ошибки shape (periods+1, k, k).
        resp_idx: Индекс переменной-отклика.
        shock_idx: Индекс переменной-шока.
        title: Заголовок графика.
        ylabel: Подпись оси Y.
    """
    horizons = np.arange(irfs.shape[0])
    vals = irfs[:, resp_idx, shock_idx]
    se = stderr[:, resp_idx, shock_idx]

    ax.plot(horizons, vals, "b-", linewidth=1.8, label="IRF")
    ax.fill_between(
        horizons,
        vals - 1.96 * se,
        vals + 1.96 * se,
        alpha=0.20,
        color="blue",
        label="95% ДИ",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Горизонт (месяцев)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(horizons)
    ax.legend()


def fig_19_irf_urals_shock(result: VARResults) -> None:
    """IRF: реакция доходности IMOEX на шок цены нефти Urals с 95% ДИ.

    Args:
        result: Оценённая VAR-модель.
    """
    irfs, stderr = _get_irf_values_and_stderr(result, 12)
    names = list(result.names)
    imoex_idx = names.index("imoex_ret")
    shock_idx = names.index("urals_ret")

    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_single_irf_with_ci(
        ax, irfs, stderr, imoex_idx, shock_idx,
        title="IRF: реакция доходности IMOEX на шок цены нефти Urals",
        ylabel="Отклик доходности IMOEX",
    )
    fig.tight_layout()
    save_fig(fig, "fig_19_irf_urals_shock", subfolder=_SUBFOLDER)
    log.info("fig_19 сохранён")


def fig_20_irf_usdrub_shock(result: VARResults) -> None:
    """IRF: реакция доходности IMOEX на шок курса USD/RUB с 95% ДИ.

    Args:
        result: Оценённая VAR-модель.
    """
    irfs, stderr = _get_irf_values_and_stderr(result, 12)
    names = list(result.names)
    imoex_idx = names.index("imoex_ret")
    shock_idx = names.index("usdrub_change")

    fig, ax = plt.subplots(figsize=(10, 5))
    _plot_single_irf_with_ci(
        ax, irfs, stderr, imoex_idx, shock_idx,
        title="IRF: реакция доходности IMOEX на шок курса USD/RUB",
        ylabel="Отклик доходности IMOEX",
    )
    fig.tight_layout()
    save_fig(fig, "fig_20_irf_usdrub_shock", subfolder=_SUBFOLDER)
    log.info("fig_20 сохранён")


def fig_21_irf_all_to_imoex(result: VARResults) -> None:
    """IRF: отклик доходности IMOEX на 4 различных шока — все линии на одном графике.

    Args:
        result: Оценённая VAR-модель.
    """
    irfs, _ = _get_irf_values_and_stderr(result, 12)
    names = list(result.names)
    imoex_idx = names.index("imoex_ret")
    shocks = [n for n in names if n != "imoex_ret"]
    horizons = np.arange(irfs.shape[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    for shock_name in shocks:
        shock_idx = names.index(shock_name)
        label = _VAR_LABELS.get(shock_name, shock_name)
        ax.plot(horizons, irfs[:, imoex_idx, shock_idx], marker="o", markersize=3, label=label, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("IRF: реакция доходности IMOEX на различные шоки")
    ax.set_xlabel("Горизонт (месяцев)")
    ax.set_ylabel("Отклик доходности IMOEX")
    ax.set_xticks(horizons)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_21_irf_all_to_imoex", subfolder=_SUBFOLDER)
    log.info("fig_21 сохранён")


def fig_22_fevd(result: VARResults) -> None:
    """Столбчатая диаграмма с накоплением: FEVD доходности IMOEX.

    Args:
        result: Оценённая VAR-модель.
    """
    fevd = result.fevd(12)
    imoex_idx = list(result.names).index("imoex_ret")
    horizons = [1, 3, 6, 12]

    data_plot = pd.DataFrame(
        [fevd.decomp[imoex_idx, h - 1, :] for h in horizons],
        index=horizons,
        columns=[_VAR_LABELS.get(n, n) for n in result.names],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    data_plot.plot(kind="bar", stacked=True, ax=ax, width=0.55)
    ax.set_title("Декомпозиция дисперсии (FEVD) доходности IMOEX")
    ax.set_xlabel("Горизонт прогноза (месяцев)")
    ax.set_ylabel("Доля объяснённой дисперсии")
    ax.set_xticklabels([str(h) for h in horizons], rotation=0)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig_22_fevd", subfolder=_SUBFOLDER)
    log.info("fig_22 сохранён")


def fig_23_svar_irf(result: VARResults, data: pd.DataFrame) -> None:
    """SVAR IRF: структурные отклики IMOEX с порядком Холесского из SVAR_ORDER.

    Переупорядочивает ряды согласно SVAR_ORDER и строит IRF на переоценённой
    VAR. Шок imoex_ret исключается — показаны 4 внешних шока → IMOEX.

    Args:
        result: Оценённая VAR-модель (для получения числа лагов).
        data: Исходный DataFrame с рядами (колонки = переменные VAR).
    """
    try:
        data_svar = data[SVAR_ORDER]
        model_svar = VAR(data_svar)
        result_svar = model_svar.fit(result.k_ar)
        irf_svar = result_svar.irf(12)
        irfs_svar = irf_svar.irfs
        try:
            stderr_svar = irf_svar.stderr(orth=False)
        except Exception:
            log.warning("SVAR: не удалось вычислить stderr; использую нулевые")
            stderr_svar = np.zeros_like(irfs_svar)
    except Exception as exc:
        log.error("Не удалось оценить SVAR: %s; использую исходную VAR с порядком Холесского", exc)
        irfs_svar = result.irf(12).irfs
        stderr_svar = np.zeros_like(irfs_svar)
        result_svar = result

    names_svar = list(result_svar.names)
    imoex_idx_svar = names_svar.index("imoex_ret")
    shocks = [n for n in names_svar if n != "imoex_ret"]
    horizons = np.arange(irfs_svar.shape[0])

    fig, ax = plt.subplots(figsize=(12, 6))
    for shock_name in shocks:
        shock_idx = names_svar.index(shock_name)
        label = _VAR_LABELS.get(shock_name, shock_name)
        vals = irfs_svar[:, imoex_idx_svar, shock_idx]
        se = stderr_svar[:, imoex_idx_svar, shock_idx]
        (line,) = ax.plot(horizons, vals, marker="o", markersize=3, label=label, linewidth=1.5)
        ax.fill_between(horizons, vals - 1.96 * se, vals + 1.96 * se, alpha=0.12, color=line.get_color())

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("SVAR IRF: структурные импульсные отклики доходности IMOEX (порядок Холесского)")
    ax.set_xlabel("Горизонт (месяцев)")
    ax.set_ylabel("Структурный отклик")
    ax.set_xticks(horizons)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_23_svar_irf", subfolder=_SUBFOLDER)
    log.info("fig_23 сохранён")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def main() -> None:
    """Основная функция: запускает полный VAR/SVAR анализ."""
    setup_plot_style()

    data = load_var_data()
    fit_result = fit_var(data)
    result = fit_result.result
    lag_df = fit_result.lag_df

    table_T7_lag_selection(lag_df)
    table_T8_granger(result, data)
    table_T9_fevd(result)
    table_T10_diagnostics(result)

    fig_17_lag_selection(lag_df)
    fig_18_irf_imoex_shock(result)
    fig_19_irf_urals_shock(result)
    fig_20_irf_usdrub_shock(result)
    fig_21_irf_all_to_imoex(result)
    fig_22_fevd(result)
    fig_23_svar_irf(result, data)

    log.info("VAR/SVAR анализ завершён. Фигуры: figures/step_06/ (7 PNG), таблицы: T7–T10")


if __name__ == "__main__":
    main()
