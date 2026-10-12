"""Шаг 6: обнаружение рыночных режимов с помощью скрытой марковской модели (HMM).

Вход:  data/raw/imoex.csv (дневные данные)
Выход: data/processed/daily_regimes.parquet     — дневные режимы (0,1,2)
       data/processed/monthly_regimes.parquet   — месячные режимы (мода)
       figures/step_07/fig_24..fig_31.png
       tables/T11_regime_characteristics.csv
       tables/T12_transition_matrix.csv
       tables/T13_regime_tests.csv

Запуск: python src/07_regime_detection.py
"""
from __future__ import annotations

import logging
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_PROCESSED,
    DATA_RAW,
    END_DATE,
    HMM_MIN_REGIME_SIZE,
    HMM_N_STATES_CANDIDATES,
    RANDOM_STATE,
    START_DATE,
)
from src.utils import save_fig, save_table, setup_plot_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_07")

_REGIME_COLORS_3 = {0: "#ff9999", 1: "#ffff99", 2: "#99ff99"}
_REGIME_COLORS_2 = {0: "#ff9999", 1: "#99ff99"}
_REGIME_COLORS_4 = {0: "#cc0000", 1: "#ff9999", 2: "#ffff99", 3: "#99ff99"}


def _regime_colors(n_states: int) -> dict[int, str]:
    """Возвращает цветовую схему для заданного числа режимов."""
    mapping = {2: _REGIME_COLORS_2, 3: _REGIME_COLORS_3, 4: _REGIME_COLORS_4}
    return mapping.get(n_states, _REGIME_COLORS_3)


def _regime_names(n_states: int) -> dict[int, str]:
    """Возвращает словарь названий режимов для заданного числа состояний.

    Args:
        n_states: Число скрытых состояний HMM.

    Returns:
        Словарь {индекс_режима: название_режима}.
    """
    if n_states == 2:
        return {0: "Медвежий", 1: "Бычий"}
    if n_states == 3:
        return {0: "Медвежий/стресс", 1: "Нейтральный", 2: "Бычий"}
    return {0: "Паника", 1: "Медвежий/стресс", 2: "Нейтральный", 3: "Бычий"}


def load_imoex() -> pd.DataFrame:
    """Загружает дневные данные IMOEX и вычисляет дневную доходность.

    Returns:
        DataFrame с колонками: close, volume, daily_ret; индекс — DatetimeIndex.
    """
    path = os.path.join(DATA_RAW, "imoex.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
    df = df.loc[START_DATE:END_DATE, ["close", "volume"]]
    df["daily_ret"] = df["close"].pct_change()
    log.info("IMOEX загружен: shape=%s, период=%s — %s", df.shape, df.index[0].date(), df.index[-1].date())
    return df


def build_hmm_features(df: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Формирует матрицу признаков для обучения HMM.

    Три признака: дневная доходность, аннуализированная волатильность (20 дней),
    логарифм дневного объёма.

    Args:
        df: DataFrame с колонками close, volume, daily_ret.

    Returns:
        Кортеж (X, dates): двумерный массив (n_days, 3) и соответствующий DatetimeIndex.
    """
    ret = df["daily_ret"].fillna(0)
    vol = df["daily_ret"].rolling(20).std().bfill() * np.sqrt(252)
    log_vol = np.log1p(df["volume"])

    features = pd.DataFrame({"ret": ret, "vol": vol, "log_volume": log_vol}, index=df.index)
    features = features.dropna()
    X = features.values.astype(np.float64)
    log.info("Матрица признаков HMM: shape=%s", X.shape)
    return X, features.index


def _check_model_valid(model: GaussianHMM, X: np.ndarray) -> bool:
    """Проверяет, что все режимы модели содержат достаточно наблюдений.

    Args:
        model: Обученная модель GaussianHMM.
        X: Матрица признаков (n_days, n_features).

    Returns:
        True, если все режимы содержат >= HMM_MIN_REGIME_SIZE наблюдений.
    """
    states = model.predict(X)
    counts = pd.Series(states).value_counts()
    return bool((counts >= HMM_MIN_REGIME_SIZE).all())


def select_hmm(
    X: np.ndarray,
) -> tuple[GaussianHMM, int, dict[int, float]]:
    """Перебирает число состояний HMM и выбирает модель по минимуму BIC.

    При наличии вырожденных режимов (< HMM_MIN_REGIME_SIZE наблюдений) модель
    с данным числом состояний пропускается; выбирается следующая по BIC валидная.

    Args:
        X: Матрица признаков (n_days, n_features).

    Returns:
        Кортеж (лучшая_модель, лучшее_n_states, словарь_bic).
    """
    n_features = X.shape[1]
    bic_dict: dict[int, float] = {}
    models: dict[int, GaussianHMM] = {}

    for n in HMM_N_STATES_CANDIDATES:
        model = GaussianHMM(
            n_components=n,
            covariance_type="full",
            n_iter=1000,
            random_state=RANDOM_STATE,
        )
        model.fit(X)
        log_likelihood = model.score(X)
        k = (
            n * n
            + n * n_features
            + n * (n_features * (n_features + 1) // 2)
        )
        bic = -2.0 * log_likelihood + k * np.log(len(X))
        bic_dict[n] = bic
        models[n] = model
        valid = _check_model_valid(model, X)
        log.info(
            "HMM n_states=%d: log_likelihood=%.4f, BIC=%.2f, валидна=%s",
            n, log_likelihood, bic, valid,
        )

    sorted_by_bic = sorted(bic_dict, key=lambda k: bic_dict[k])
    best_model: GaussianHMM | None = None
    best_n = 0
    for n in sorted_by_bic:
        if _check_model_valid(models[n], X):
            best_model = models[n]
            best_n = n
            break

    if best_model is None:
        log.warning("Ни одна модель не прошла валидацию. Использую n_states=%d.", sorted_by_bic[0])
        best_n = sorted_by_bic[0]
        best_model = models[best_n]

    log.info("Выбрано n_states=%d (BIC=%.2f)", best_n, bic_dict[best_n])
    return best_model, best_n, bic_dict


def decode_states(
    model: GaussianHMM,
    X: np.ndarray,
    feature_dates: pd.DatetimeIndex,
) -> pd.Series:
    """Декодирует скрытые состояния HMM методом Витерби.

    Args:
        model: Обученная модель GaussianHMM.
        X: Матрица признаков (n_days, n_features).
        feature_dates: DatetimeIndex, соответствующий строкам X.

    Returns:
        Series с дневными режимами; индекс — feature_dates.
    """
    states = model.predict(X)
    daily_states = pd.Series(states, index=feature_dates, name="regime")
    return daily_states


def label_regimes(
    daily_states: pd.Series,
    daily_ret: pd.Series,
) -> tuple[pd.Series, dict[int, str]]:
    """Перенумеровывает режимы: 0 = наименьшая средняя доходность (медвежий).

    Args:
        daily_states: Series с исходными метками режимов.
        daily_ret: Series дневных доходностей (выровненный по индексу).

    Returns:
        Кортеж (relabeled_states, regime_names_dict).
    """
    n_states = daily_states.nunique()
    means = {
        r: daily_ret.loc[daily_states == r].mean()
        for r in range(n_states)
    }
    sorted_regimes = sorted(means, key=lambda r: means[r])
    mapping = {old: new for new, old in enumerate(sorted_regimes)}
    relabeled = daily_states.map(mapping)
    relabeled.name = "regime"
    names = _regime_names(n_states)
    log.info("Переупорядочивание режимов: %s", mapping)
    for new_label, name in names.items():
        original = sorted_regimes[new_label]
        log.info("  Режим %d (%s): средняя дневная доходность=%.5f", new_label, name, means[original])
    return relabeled, names


def aggregate_monthly(
    daily_states: pd.Series,
    monthly_features: pd.DataFrame,
) -> pd.Series:
    """Агрегирует дневные режимы в месячные путём вычисления моды.

    Args:
        daily_states: Series с дневными метками режимов.
        monthly_features: DataFrame с месячным индексом (DatetimeIndex).

    Returns:
        Series месячных режимов с индексом monthly_features.index.
    """
    monthly_regimes = daily_states.resample("ME").apply(
        lambda x: int(x.mode().iloc[0]) if len(x) > 0 else np.nan
    )
    monthly_regimes = monthly_regimes.reindex(monthly_features.index, method="nearest")
    monthly_regimes.name = "regime"
    log.info(
        "Месячные режимы: %d наблюдений, распределение:\n%s",
        len(monthly_regimes),
        monthly_regimes.value_counts().sort_index(),
    )
    return monthly_regimes


def save_regimes(daily_states: pd.Series, monthly_regimes: pd.Series) -> None:
    """Сохраняет дневные и месячные режимы в parquet-файлы.

    Args:
        daily_states: Series с дневными метками режимов.
        monthly_regimes: Series с месячными метками режимов.
    """
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    daily_states.to_frame().to_parquet(os.path.join(DATA_PROCESSED, "daily_regimes.parquet"))
    monthly_regimes.to_frame().to_parquet(os.path.join(DATA_PROCESSED, "monthly_regimes.parquet"))
    log.info("Режимы сохранены в %s", DATA_PROCESSED)


def table_T11_characteristics(
    daily_states: pd.Series,
    daily_ret: pd.Series,
    df: pd.DataFrame,
    regime_names: dict[int, str],
) -> None:
    """Рассчитывает и сохраняет характеристики каждого режима (T11).

    Args:
        daily_states: Series с дневными метками режимов.
        daily_ret: Series дневных доходностей.
        df: Исходный DataFrame IMOEX.
        regime_names: Словарь {индекс: название} режимов.
    """
    records = []
    for r, name in regime_names.items():
        mask = daily_states == r
        rets = daily_ret.loc[mask]
        count = int(mask.sum())
        fraction = count / len(daily_states) * 100
        mean_d = rets.mean()
        ann_ret = (1 + mean_d) ** 252 - 1
        std_d = rets.std()
        ann_vol = std_d * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        records.append({
            "Режим": name,
            "Число дней": count,
            "Доля, %": round(fraction, 2),
            "Средняя дневная доходность": round(mean_d, 6),
            "Годовая доходность": round(ann_ret, 4),
            "Стд дневная": round(std_d, 6),
            "Годовая волатильность": round(ann_vol, 4),
            "Коэф. Шарпа": round(sharpe, 4),
            "Медианная дневная доходность": round(rets.median(), 6),
            "Мин. дневная доходность": round(rets.min(), 6),
            "Макс. дневная доходность": round(rets.max(), 6),
        })
    t11 = pd.DataFrame(records).set_index("Режим")
    save_table(t11, "T11_regime_characteristics")
    log.info("T11 сохранена:\n%s", t11.to_string())


def table_T12_transition(
    model: GaussianHMM,
    regime_names: dict[int, str],
) -> None:
    """Сохраняет матрицу переходов HMM (T12).

    Args:
        model: Обученная модель GaussianHMM.
        regime_names: Словарь {индекс: название} режимов.
    """
    labels = [regime_names[i] for i in sorted(regime_names)]
    t12 = pd.DataFrame(
        np.round(model.transmat_, 4),
        index=labels,
        columns=labels,
    )
    t12.index.name = "Из / В"
    save_table(t12, "T12_transition_matrix")
    log.info("T12 (матрица переходов):\n%s", t12.to_string())


def table_T13_tests(
    daily_states: pd.Series,
    daily_ret: pd.Series,
    regime_names: dict[int, str],
) -> None:
    """Проводит статистические тесты различия режимов и сохраняет результаты (T13).

    Args:
        daily_states: Series с дневными метками режимов.
        daily_ret: Series дневных доходностей.
        regime_names: Словарь {индекс: название} режимов.
    """
    groups = [daily_ret.loc[daily_states == r].dropna().values for r in sorted(regime_names)]
    h_stat, p_kw = stats.kruskal(*groups)

    records = [{"Тест": "Kruskal-Wallis (все режимы)", "Статистика": round(h_stat, 4), "p-значение": round(p_kw, 6)}]

    regime_ids = sorted(regime_names)
    for i in range(len(regime_ids)):
        for j in range(i + 1, len(regime_ids)):
            r_i, r_j = regime_ids[i], regime_ids[j]
            u_stat, p_mw = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            records.append({
                "Тест": f"Mann-Whitney U: {regime_names[r_i]} vs {regime_names[r_j]}",
                "Статистика": round(u_stat, 2),
                "p-значение": round(p_mw, 6),
            })

    t13 = pd.DataFrame(records).set_index("Тест")
    save_table(t13, "T13_regime_tests")
    log.info("T13 (тесты различия режимов):\n%s", t13.to_string())
    if p_kw >= 0.05:
        log.warning("Kruskal-Wallis p=%.4f >= 0.05: режимы статистически НЕ различимы!", p_kw)
    else:
        log.info("Kruskal-Wallis p=%.6f < 0.05: режимы статистически различимы.", p_kw)


def fig_24_bic_selection(bic_dict: dict[int, float]) -> None:
    """Столбчатая диаграмма значений BIC для разного числа состояний HMM.

    Args:
        bic_dict: Словарь {n_states: BIC}.
    """
    best_n = min(bic_dict, key=lambda k: bic_dict[k])
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3" if n != best_n else "#F44336" for n in bic_dict]
    bars = ax.bar(list(bic_dict.keys()), list(bic_dict.values()), color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Число скрытых состояний")
    ax.set_ylabel("BIC")
    ax.set_title("Выбор числа состояний HMM по критерию BIC")
    ax.set_xticks(list(bic_dict.keys()))

    for bar, n in zip(bars, bic_dict):
        yval = bic_dict[n]
        ax.text(bar.get_x() + bar.get_width() / 2, yval + abs(yval) * 0.005,
                f"{yval:.0f}", ha="center", va="bottom", fontsize=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#F44336", edgecolor="black", label=f"Выбранное (n={best_n})"),
        Patch(facecolor="#2196F3", edgecolor="black", label="Остальные"),
    ]
    ax.legend(handles=legend_elements)
    save_fig(fig, "fig_24_bic_selection", subfolder="step_07")
    log.info("fig_24 сохранён.")


def fig_25_imoex_regimes(
    close: pd.Series,
    daily_states: pd.Series,
    regime_names: dict[int, str],
) -> None:
    """Главный график: IMOEX с цветовой разметкой рыночных режимов.

    Args:
        close: Series цены закрытия IMOEX.
        daily_states: Series с дневными метками режимов.
        regime_names: Словарь {индекс: название} режимов.
    """
    n_states = len(regime_names)
    colors = _regime_colors(n_states)

    fig, ax = plt.subplots(figsize=(14, 7))

    dates_list = daily_states.index.tolist()
    regimes_list = daily_states.tolist()

    prev_regime: int | None = None
    start_date = None
    for date, regime in zip(dates_list, regimes_list):
        if regime != prev_regime:
            if prev_regime is not None and start_date is not None:
                ax.axvspan(start_date, date, alpha=0.3, color=colors[prev_regime], linewidth=0)
            start_date = date
            prev_regime = regime

    if prev_regime is not None and start_date is not None:
        ax.axvspan(start_date, daily_states.index[-1], alpha=0.3, color=colors[prev_regime], linewidth=0)

    ax.plot(close.index, close.values, color="black", linewidth=1.2, zorder=3)

    patches = [
        mpatches.Patch(facecolor=colors[r], alpha=0.5, edgecolor="gray", label=regime_names[r])
        for r in sorted(regime_names)
    ]
    ax.legend(handles=patches, loc="upper left")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Значение индекса IMOEX, пунктов")
    ax.set_title("Динамика IMOEX с выделением рыночных режимов (HMM)")
    fig.autofmt_xdate()
    save_fig(fig, "fig_25_imoex_regimes", subfolder="step_07")
    log.info("fig_25 сохранён.")


def fig_26_return_distributions(
    daily_states: pd.Series,
    daily_ret: pd.Series,
    regime_names: dict[int, str],
) -> None:
    """Гистограммы дневных доходностей для каждого режима (отдельные файлы).

    Args:
        daily_states: Series с дневными метками режимов.
        daily_ret: Series дневных доходностей.
        regime_names: Словарь {индекс: название} режимов.
    """
    suffixes = ["a", "b", "c", "d"]
    n_states = len(regime_names)
    colors = _regime_colors(n_states)

    for idx, r in enumerate(sorted(regime_names)):
        rets = daily_ret.loc[daily_states == r].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(rets, bins=50, density=True, color=colors[r], edgecolor="white", linewidth=0.5, alpha=0.85)

        mu, sigma = rets.mean(), rets.std()
        x_grid = np.linspace(rets.min(), rets.max(), 200)
        ax.plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), color="black", linewidth=1.5, linestyle="--",
                label=f"N(μ={mu:.4f}, σ={sigma:.4f})")
        ax.set_xlabel("Доходность, д.е.")
        ax.set_ylabel("Плотность вероятности")
        ax.set_title(f"Распределение доходностей IMOEX — {regime_names[r]}")
        ax.legend()
        suffix = suffixes[idx] if idx < len(suffixes) else str(idx)
        save_fig(fig, f"fig_26{suffix}_regime{r}_returns", subfolder="step_07")

    log.info("fig_26a-c сохранены.")


def fig_27_vol_distributions(
    daily_states: pd.Series,
    rolling_vol: pd.Series,
    regime_names: dict[int, str],
) -> None:
    """Гистограммы волатильности для каждого режима (отдельные файлы).

    Args:
        daily_states: Series с дневными метками режимов.
        rolling_vol: Series аннуализированной волатильности (20 дней).
        regime_names: Словарь {индекс: название} режимов.
    """
    suffixes = ["a", "b", "c", "d"]
    n_states = len(regime_names)
    colors = _regime_colors(n_states)

    for idx, r in enumerate(sorted(regime_names)):
        vol_r = rolling_vol.loc[daily_states == r].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(vol_r, bins=50, density=True, color=colors[r], edgecolor="white", linewidth=0.5, alpha=0.85)

        mu, sigma = vol_r.mean(), vol_r.std()
        if sigma > 0:
            x_grid = np.linspace(max(vol_r.min(), 0), vol_r.max(), 200)
            ax.plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), color="black", linewidth=1.5, linestyle="--",
                    label=f"N(μ={mu:.3f}, σ={sigma:.3f})")
            ax.legend()
        ax.set_xlabel("Волатильность, σ (аннуал.)")
        ax.set_ylabel("Плотность вероятности")
        ax.set_title(f"Распределение волатильности IMOEX — {regime_names[r]}")
        suffix = suffixes[idx] if idx < len(suffixes) else str(idx)
        save_fig(fig, f"fig_27{suffix}_regime{r}_vol", subfolder="step_07")

    log.info("fig_27a-c сохранены.")


def fig_28_box_returns(
    daily_states: pd.Series,
    daily_ret: pd.Series,
    regime_names: dict[int, str],
) -> None:
    """Ящичковые диаграммы дневных доходностей по режимам.

    Args:
        daily_states: Series с дневными метками режимов.
        daily_ret: Series дневных доходностей.
        regime_names: Словарь {индекс: название} режимов.
    """
    n_states = len(regime_names)
    colors = _regime_colors(n_states)

    data = [daily_ret.loc[daily_states == r].dropna().values for r in sorted(regime_names)]
    labels = [regime_names[r] for r in sorted(regime_names)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, r in zip(bp["boxes"], sorted(regime_names)):
        patch.set_facecolor(colors[r])
        patch.set_alpha(0.8)

    ax.set_xlabel("Режим рынка")
    ax.set_ylabel("Дневная доходность, д.е.")
    ax.set_title("Ящичковые диаграммы доходностей IMOEX по режимам")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    save_fig(fig, "fig_28_box_returns", subfolder="step_07")
    log.info("fig_28 сохранён.")


def fig_29_transition_heatmap(
    model: GaussianHMM,
    regime_names: dict[int, str],
) -> None:
    """Тепловая карта матрицы переходов HMM.

    Args:
        model: Обученная модель GaussianHMM.
        regime_names: Словарь {индекс: название} режимов.
    """
    labels = [regime_names[i] for i in sorted(regime_names)]
    transmat = pd.DataFrame(model.transmat_, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        transmat,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Режим (t+1)")
    ax.set_ylabel("Режим (t)")
    ax.set_title("Матрица вероятностей переходов между режимами (HMM)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    save_fig(fig, "fig_29_transition_heatmap", subfolder="step_07")
    log.info("fig_29 сохранён.")


def fig_30_regime_by_year(
    monthly_regimes: pd.Series,
    regime_names: dict[int, str],
) -> None:
    """Стопчатая диаграмма: доля месяцев каждого режима по годам.

    Args:
        monthly_regimes: Series с месячными метками режимов.
        regime_names: Словарь {индекс: название} режимов.
    """
    n_states = len(regime_names)
    colors = _regime_colors(n_states)

    df = monthly_regimes.to_frame("regime")
    df["year"] = df.index.year
    yearly = df.groupby(["year", "regime"]).size().unstack(fill_value=0)

    for r in sorted(regime_names):
        if r not in yearly.columns:
            yearly[r] = 0
    yearly = yearly[sorted(regime_names)]

    yearly_frac = yearly.div(yearly.sum(axis=1), axis=0)
    yearly_frac.columns = [regime_names[r] for r in sorted(regime_names)]

    color_list = [colors[r] for r in sorted(regime_names)]
    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_frac.plot(kind="bar", stacked=True, ax=ax, color=color_list, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Год")
    ax.set_ylabel("Доля месяцев")
    ax.set_title("Доля месяцев каждого режима по годам")
    ax.legend(title="Режим рынка", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim(0, 1)
    save_fig(fig, "fig_30_regime_by_year", subfolder="step_07")
    log.info("fig_30 сохранён.")


def fig_31_risk_return(
    daily_states: pd.Series,
    daily_ret: pd.Series,
    regime_names: dict[int, str],
) -> None:
    """Диаграмма рассеяния «риск — доходность» по режимам (одна точка на режим).

    Args:
        daily_states: Series с дневными метками режимов.
        daily_ret: Series дневных доходностей.
        regime_names: Словарь {индекс: название} режимов.
    """
    n_states = len(regime_names)
    colors = _regime_colors(n_states)

    ann_vols, ann_rets, labels_list, color_list = [], [], [], []
    for r, name in regime_names.items():
        rets = daily_ret.loc[daily_states == r].dropna()
        ann_vol = rets.std() * np.sqrt(252)
        ann_ret = (1 + rets.mean()) ** 252 - 1
        ann_vols.append(ann_vol)
        ann_rets.append(ann_ret)
        labels_list.append(name)
        color_list.append(colors[r])

    fig, ax = plt.subplots(figsize=(8, 6))
    for x, y, label, color in zip(ann_vols, ann_rets, labels_list, color_list):
        ax.scatter(x, y, s=200, color=color, edgecolor="black", linewidth=1.2, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Годовая волатильность, σ")
    ax.set_ylabel("Годовая доходность, д.е.")
    ax.set_title("Соотношение риска и доходности по рыночным режимам")
    save_fig(fig, "fig_31_risk_return", subfolder="step_07")
    log.info("fig_31 сохранён.")


def validate(
    daily_states: pd.Series,
    daily_ret: pd.Series,
    model: GaussianHMM,
) -> None:
    """Проверяет качество обнаруженных режимов.

    Args:
        daily_states: Series с дневными метками режимов.
        daily_ret: Series дневных доходностей.
        model: Обученная модель GaussianHMM.

    Raises:
        AssertionError: Если какой-либо режим содержит менее HMM_MIN_REGIME_SIZE наблюдений.
    """
    counts = daily_states.value_counts()
    log.info("Число наблюдений по режимам:\n%s", counts.to_string())

    assert (counts >= HMM_MIN_REGIME_SIZE).all(), (
        f"Режим(ы) с недостаточным числом наблюдений: "
        f"{counts[counts < HMM_MIN_REGIME_SIZE].to_dict()}"
    )
    log.info("OK: все режимы содержат >= %d наблюдений.", HMM_MIN_REGIME_SIZE)

    diag = np.diag(model.transmat_)
    if not (diag > 0.5).all():
        log.warning("Матрица переходов: некоторые диагональные элементы <= 0.5: %s", diag)
    else:
        log.info("OK: все диагональные элементы матрицы переходов > 0.5: %s", diag.round(3))

    groups = [daily_ret.loc[daily_states == r].dropna().values for r in sorted(daily_states.unique())]
    h_stat, p_kw = stats.kruskal(*groups)
    if p_kw >= 0.05:
        log.warning("Kruskal-Wallis p=%.4f >= 0.05: режимы статистически НЕ различимы по доходностям!", p_kw)
    else:
        log.info("OK: Kruskal-Wallis p=%.6f < 0.05 — режимы статистически различимы.", p_kw)


def main() -> None:
    """Главная функция: полный pipeline обнаружения режимов HMM."""
    setup_plot_style()

    df = load_imoex()
    X, feature_dates = build_hmm_features(df)
    best_model, best_n, bic_dict = select_hmm(X)
    daily_states_raw = decode_states(best_model, X, feature_dates)
    daily_states, regime_names = label_regimes(daily_states_raw, df["daily_ret"].loc[feature_dates])

    features = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_features.parquet"))
    monthly_regimes = aggregate_monthly(daily_states, features)

    save_regimes(daily_states, monthly_regimes)
    validate(daily_states, df["daily_ret"].loc[daily_states.index], best_model)

    table_T11_characteristics(daily_states, df["daily_ret"].loc[daily_states.index], df, regime_names)
    table_T12_transition(best_model, regime_names)
    table_T13_tests(daily_states, df["daily_ret"].loc[daily_states.index], regime_names)

    rolling_vol = df["daily_ret"].rolling(20).std() * np.sqrt(252)
    rolling_vol_aligned = rolling_vol.loc[daily_states.index]

    fig_24_bic_selection(bic_dict)
    fig_25_imoex_regimes(df["close"].loc[daily_states.index], daily_states, regime_names)
    fig_26_return_distributions(daily_states, df["daily_ret"].loc[daily_states.index], regime_names)
    fig_27_vol_distributions(daily_states, rolling_vol_aligned, regime_names)
    fig_28_box_returns(daily_states, df["daily_ret"].loc[daily_states.index], regime_names)
    fig_29_transition_heatmap(best_model, regime_names)
    fig_30_regime_by_year(monthly_regimes, regime_names)
    fig_31_risk_return(daily_states, df["daily_ret"].loc[daily_states.index], regime_names)

    log.info("HMM анализ завершён.")


if __name__ == "__main__":
    main()
