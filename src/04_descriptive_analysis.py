"""Шаг 3 (Описательная статистика): расчёт таблиц T3–T6 и графиков fig_10–fig_16.

Вычисляет описательные статистики, тесты стационарности ADF/KPSS,
параметры GARCH(1,1) и тест Жарка–Бера для дневных и месячных рядов IMOEX.
Строит визуализации: гистограмма доходностей, QQ-график, скользящая волатильность,
GARCH vs реализованная волатильность, ACF/PACF квадратов доходностей,
скользящая корреляция с Urals, скользящий VaR/CVaR.

Выходные файлы (tables/):
    T3_descriptive_stats.csv — описательные статистики доходностей IMOEX
    T4_stationarity.csv      — тесты стационарности ADF и KPSS
    T5_garch_params.csv      — параметры модели GARCH(1,1)
    T6_jarque_bera.csv       — тест Жарка–Бера на нормальность

Выходные графики (figures/step_04/):
    fig_10_returns_histogram.png   — гистограмма дневных доходностей + нормальная кривая
    fig_11_qq_plot.png             — QQ-график дневных доходностей
    fig_12a_rolling_vol_20d.png    — скользящая волатильность (окно 20 дней)
    fig_12b_rolling_vol_60d.png    — скользящая волатильность (окно 60 дней)
    fig_13_garch_vs_realized.png   — GARCH(1,1) vs реализованная волатильность
    fig_14a_acf.png                — ACF квадратов дневных доходностей
    fig_14b_pacf.png               — PACF квадратов дневных доходностей
    fig_15_rolling_correlation.png — скользящая корреляция IMOEX и Urals (60 дней)
    fig_16a_rolling_var.png        — скользящий VaR(5%) (окно 60 дней)
    fig_16b_rolling_cvar.png       — скользящий CVaR(5%) (окно 60 дней)
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PROCESSED, DATA_RAW, END_DATE, START_DATE
from src.utils import save_fig, save_table, setup_plot_style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("step_04")


def _load_daily_series(filename: str, date_col: str = "date", value_col: str = "close") -> pd.DataFrame:
    """Загружает дневной ряд из CSV, парсит даты, сортирует, дедуплицирует.

    Args:
        filename: Имя файла в data/raw/.
        date_col: Название столбца с датой.
        value_col: Название столбца со значением.

    Returns:
        DataFrame с DatetimeIndex и столбцом value_col, отфильтрованный по периоду.
    """
    path = os.path.join(DATA_RAW, filename)
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
    df = df.set_index(date_col)
    df = df.loc[START_DATE:END_DATE]
    return df


def load_data() -> dict[str, pd.DataFrame]:
    """Загружает все необходимые данные для описательного анализа.

    Returns:
        Словарь с ключами:
            imoex_daily — дневные OHLCV IMOEX с вычисленной daily_ret;
            urals_daily — дневные цены Urals с вычисленной daily_ret;
            features    — месячные признаки (132×25);
            target      — месячная целевая переменная (132×1).
    """
    imoex = _load_daily_series("imoex.csv")
    imoex["daily_ret"] = imoex["close"].pct_change()
    log.info("IMOEX daily: shape=%s, диапазон %s — %s", imoex.shape, imoex.index.min(), imoex.index.max())

    urals = _load_daily_series("urals.csv")
    urals["daily_ret"] = urals["close"].pct_change()
    log.info("Urals daily: shape=%s, диапазон %s — %s", urals.shape, urals.index.min(), urals.index.max())

    usdrub = _load_daily_series("usdrub.csv")
    log.info("USD/RUB daily: shape=%s", usdrub.shape)

    key_rate = _load_daily_series("key_rate.csv", value_col="rate")
    log.info("Ключевая ставка: shape=%s", key_rate.shape)

    features = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_features.parquet"))
    target = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_target.parquet"))
    log.info("Месячные признаки: shape=%s", features.shape)

    return {
        "imoex_daily": imoex,
        "urals_daily": urals,
        "usdrub_daily": usdrub,
        "key_rate_daily": key_rate,
        "features": features,
        "target": target,
    }


def compute_stats(returns: pd.Series, label: str, is_daily: bool = True) -> dict:
    """Вычисляет описательные статистики ряда доходностей.

    Все показатели аннуализируются: дневные умножаются на 252,
    месячные — на 12.

    Args:
        returns: Ряд доходностей (без NaN).
        label: Человеческое название ряда (для индекса строки).
        is_daily: True — дневные доходности (множитель 252), False — месячные (12).

    Returns:
        Словарь с вычисленными метриками.
    """
    rets = returns.dropna()
    ann_factor = 252 if is_daily else 12

    mean_ann = rets.mean() * ann_factor
    std_ann = rets.std() * np.sqrt(ann_factor)
    median_val = rets.median()
    skewness = float(stats.skew(rets))
    excess_kurt = float(stats.kurtosis(rets))

    var_5pct = float(rets.quantile(0.05))
    cvar_5pct = float(rets[rets <= var_5pct].mean())

    sharpe = mean_ann / std_ann if std_ann != 0 else float("nan")

    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    jb_stat, jb_pval = stats.jarque_bera(rets)

    return {
        "label": label,
        "Среднее (аннуал.)": round(mean_ann, 4),
        "Медиана": round(median_val, 4),
        "Std (аннуал.)": round(std_ann, 4),
        "Асимметрия": round(skewness, 4),
        "Эксцесс": round(excess_kurt, 4),
        "VaR 5%": round(var_5pct, 4),
        "CVaR 5%": round(cvar_5pct, 4),
        "Коэф. Шарпа": round(sharpe, 4),
        "Макс. просадка": round(max_drawdown, 4),
        "JB статистика": round(float(jb_stat), 4),
        "JB p-value": round(float(jb_pval), 6),
    }


def table_T3_descriptive(raw: dict) -> pd.DataFrame:
    """Формирует таблицу T3: описательные статистики доходностей IMOEX.

    Включает строки: дневные и месячные доходности.

    Args:
        raw: Словарь данных из load_data().

    Returns:
        DataFrame с двумя строками (дневные / месячные доходности).
    """
    daily_rets = raw["imoex_daily"]["close"].pct_change().dropna()
    monthly_rets = raw["features"]["imoex_ret_1m"].dropna()

    rows = [
        compute_stats(daily_rets, "IMOEX дневные доходности", is_daily=True),
        compute_stats(monthly_rets, "IMOEX месячные доходности", is_daily=False),
    ]

    df = pd.DataFrame(rows).set_index("label")
    df.index.name = "Ряд"
    save_table(df, "T3_descriptive_stats")
    log.info("T3 сохранена:\n%s", df.to_string())
    return df


def _run_adf(series: pd.Series) -> tuple[float, float, bool]:
    """Тест Дики–Фуллера (ADF) на единичный корень.

    H0: единичный корень (нестационарный ряд).
    Стационарен, если p < 0.05.

    Args:
        series: Временной ряд без NaN.

    Returns:
        Кортеж (adf_stat, p_value, is_stationary).
    """
    result = adfuller(series.dropna(), autolag="AIC")
    adf_stat = round(float(result[0]), 4)
    p_val = round(float(result[1]), 4)
    is_stationary = p_val < 0.05
    return adf_stat, p_val, is_stationary


def _run_kpss(series: pd.Series) -> tuple[float, float, bool]:
    """Тест KPSS на стационарность.

    H0: ряд стационарен.
    Стационарен, если p > 0.05.

    Args:
        series: Временной ряд без NaN.

    Returns:
        Кортеж (kpss_stat, p_value, is_stationary).
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(series.dropna(), regression="c", nlags="auto")

    kpss_stat = round(float(result[0]), 4)
    p_val = round(float(result[1]), 4)
    is_stationary = p_val > 0.05
    return kpss_stat, p_val, is_stationary


def _stationarity_conclusion(adf_stationary: bool, kpss_stationary: bool) -> str:
    """Формирует текстовый вывод о стационарности.

    Args:
        adf_stationary: Стационарен по ADF (p<0.05).
        kpss_stationary: Стационарен по KPSS (p>0.05).

    Returns:
        Строка-заключение.
    """
    if adf_stationary and kpss_stationary:
        return "Стационарный"
    if not adf_stationary and not kpss_stationary:
        return "Нестационарный"
    return "Неоднозначно"


def table_T4_stationarity(raw: dict) -> pd.DataFrame:
    """Формирует таблицу T4: тесты стационарности ADF и KPSS.

    Проверяет 6 рядов: цены и доходности IMOEX, Urals, USD/RUB, ставка ЦБ.

    Args:
        raw: Словарь данных из load_data().

    Returns:
        DataFrame с результатами ADF и KPSS для каждого ряда.
    """
    imoex_price = raw["imoex_daily"]["close"]
    imoex_ret_daily = raw["imoex_daily"]["close"].pct_change().dropna()
    imoex_ret_monthly = raw["features"]["imoex_ret_1m"].dropna()
    urals_price = raw["urals_daily"]["close"]
    usdrub_price = raw["usdrub_daily"]["close"]
    key_rate = raw["key_rate_daily"]["rate"]

    series_to_test: list[tuple[str, pd.Series]] = [
        ("IMOEX (уровни)", imoex_price),
        ("IMOEX (дневные доходности)", imoex_ret_daily),
        ("IMOEX (месячные доходности)", imoex_ret_monthly),
        ("Urals (уровни)", urals_price),
        ("USD/RUB (уровни)", usdrub_price),
        ("Ключевая ставка ЦБ РФ", key_rate),
    ]

    rows: list[dict] = []
    for name, series in series_to_test:
        adf_stat, adf_p, adf_ok = _run_adf(series)
        kpss_stat, kpss_p, kpss_ok = _run_kpss(series)
        conclusion = _stationarity_conclusion(adf_ok, kpss_ok)
        rows.append({
            "Ряд": name,
            "ADF статистика": adf_stat,
            "ADF p-value": adf_p,
            "ADF стационарен": "Да" if adf_ok else "Нет",
            "KPSS статистика": kpss_stat,
            "KPSS p-value": kpss_p,
            "KPSS стационарен": "Да" if kpss_ok else "Нет",
            "Заключение": conclusion,
        })
        log.info("T4 | %s → ADF p=%.4f, KPSS p=%.4f → %s", name, adf_p, kpss_p, conclusion)

    df = pd.DataFrame(rows).set_index("Ряд")
    save_table(df, "T4_stationarity")
    log.info("T4 сохранена")
    return df


def table_T5_garch_params(raw: dict) -> pd.DataFrame:
    """Формирует таблицу T5: параметры GARCH(1,1) на дневных доходностях IMOEX.

    Оценивает GARCH(1,1) с t-распределением на полном доступном ряду.
    Рассчитывает: omega, alpha, beta, персистентность, период полураспада,
    степени свободы (nu), log-likelihood, AIC, BIC.

    Args:
        raw: Словарь данных из load_data().

    Returns:
        Однострочный DataFrame с параметрами модели.
    """
    returns = raw["imoex_daily"]["close"].pct_change().dropna()
    log.info("GARCH: оценка на %d дневных доходностях", len(returns))

    am = arch_model(returns * 100, vol="Garch", p=1, q=1, mean="Constant", dist="t")
    res = am.fit(disp="off", show_warning=False)

    params = res.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    persistence = alpha + beta
    half_life = float(np.log(0.5) / np.log(persistence)) if persistence < 1 else float("inf")
    nu = float(params["nu"])

    row = {
        "Параметр": ["GARCH(1,1) — IMOEX дневные доходности"],
        "omega (×10⁻⁴)": [round(omega, 4)],
        "alpha[1]": [round(alpha, 4)],
        "beta[1]": [round(beta, 4)],
        "Персистентность (α+β)": [round(persistence, 4)],
        "Период полураспада (дней)": [round(half_life, 2)],
        "nu (ст.св. t)": [round(nu, 4)],
        "Log-likelihood": [round(float(res.loglikelihood), 4)],
        "AIC": [round(float(res.aic), 4)],
        "BIC": [round(float(res.bic), 4)],
    }

    df = pd.DataFrame(row).set_index("Параметр")
    save_table(df, "T5_garch_params")
    log.info(
        "T5 сохранена: alpha=%.4f, beta=%.4f, persistence=%.4f, half-life=%.1f дней",
        alpha, beta, persistence, half_life,
    )
    return df


def table_T6_jarque_bera(raw: dict) -> pd.DataFrame:
    """Формирует таблицу T6: тест Жарка–Бера на нормальность распределений.

    Проверяет три ряда: дневные и месячные доходности IMOEX, дневные доходности Urals.

    Args:
        raw: Словарь данных из load_data().

    Returns:
        DataFrame с результатами теста JB для каждого ряда.
    """
    series_list: list[tuple[str, pd.Series]] = [
        ("IMOEX дневные доходности", raw["imoex_daily"]["close"].pct_change().dropna()),
        ("IMOEX месячные доходности", raw["features"]["imoex_ret_1m"].dropna()),
        ("Urals дневные доходности", raw["urals_daily"]["close"].pct_change().dropna()),
    ]

    rows: list[dict] = []
    for name, series in series_list:
        n = len(series)
        jb_stat, jb_p = stats.jarque_bera(series)
        skewness = round(float(stats.skew(series)), 4)
        excess_kurt = round(float(stats.kurtosis(series)), 4)
        normal = "Нет" if jb_p < 0.05 else "Да"
        rows.append({
            "Ряд": name,
            "N": n,
            "JB статистика": round(float(jb_stat), 4),
            "JB p-value": round(float(jb_p), 6),
            "Асимметрия": skewness,
            "Эксцесс": excess_kurt,
            "Нормальное распределение": normal,
        })
        log.info("T6 | %s: JB=%.2f, p=%.4f, нормальное=%s", name, jb_stat, jb_p, normal)

    df = pd.DataFrame(rows).set_index("Ряд")
    save_table(df, "T6_jarque_bera")
    log.info("T6 сохранена")
    return df


def fig_10_returns_histogram(raw: dict) -> None:
    """Строит гистограмму дневных доходностей IMOEX с наложенной нормальной кривой.

    Args:
        raw: Словарь данных из load_data().
    """
    returns = raw["imoex_daily"]["close"].pct_change().dropna()
    mean_val = float(returns.mean())
    std_val = float(returns.std())

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_color = colors[0] if colors else "steelblue"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        returns,
        bins=60,
        density=True,
        alpha=0.7,
        color=bar_color,
        edgecolor="black",
        linewidth=0.3,
        label="Эмпирическое распределение",
    )

    x_grid = np.linspace(returns.min(), returns.max(), 200)
    normal_pdf = stats.norm.pdf(x_grid, mean_val, std_val)
    ax.plot(x_grid, normal_pdf, color="red", linestyle="--", linewidth=2.0,
            label="Нормальное распределение")

    ax.set_title("Распределение дневных доходностей IMOEX")
    ax.set_xlabel("Доходность, д.е.")
    ax.set_ylabel("Плотность вероятности")
    ax.legend()

    save_fig(fig, "fig_10_returns_histogram", subfolder="step_04")
    log.info("fig_10 сохранён")


def fig_11_qq_plot(raw: dict) -> None:
    """Строит QQ-график дневных доходностей IMOEX относительно нормального распределения.

    Args:
        raw: Словарь данных из load_data().
    """
    returns = raw["imoex_daily"]["close"].pct_change().dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(returns, dist="norm", plot=ax)

    ax.set_title("QQ-график дневных доходностей IMOEX (vs. нормальное распределение)")
    ax.set_xlabel("Теоретические квантили нормального распределения")
    ax.set_ylabel("Эмпирические квантили доходностей")

    save_fig(fig, "fig_11_qq_plot", subfolder="step_04")
    log.info("fig_11 сохранён")


def fig_12a_rolling_vol_20d(raw: dict) -> None:
    """Строит скользящую аннуализированную волатильность IMOEX с окном 20 дней.

    Args:
        raw: Словарь данных из load_data().
    """
    returns = raw["imoex_daily"]["close"].pct_change().dropna()
    vol_20 = returns.rolling(20).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(vol_20.index, vol_20.values, linewidth=1.2)
    ax.set_title("Скользящая волатильность IMOEX (окно 20 дней, аннуализированная)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Волатильность, σ (аннуал.)")

    save_fig(fig, "fig_12a_rolling_vol_20d", subfolder="step_04")
    log.info("fig_12a сохранён")


def fig_12b_rolling_vol_60d(raw: dict) -> None:
    """Строит скользящую аннуализированную волатильность IMOEX с окном 60 дней.

    Args:
        raw: Словарь данных из load_data().
    """
    returns = raw["imoex_daily"]["close"].pct_change().dropna()
    vol_60 = returns.rolling(60).std() * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(vol_60.index, vol_60.values, linewidth=1.2)
    ax.set_title("Скользящая волатильность IMOEX (окно 60 дней, аннуализированная)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Волатильность, σ (аннуал.)")

    save_fig(fig, "fig_12b_rolling_vol_60d", subfolder="step_04")
    log.info("fig_12b сохранён")


def fig_13_garch_vs_realized(raw: dict) -> None:
    """Сравнивает условную волатильность GARCH(1,1) с реализованной волатильностью IMOEX.

    Реализованная волатильность: 20-дневное скользящее std, аннуализированное,
    агрегированное до месячного уровня (последнее значение месяца).
    Условная волатильность GARCH: из столбца imoex_garch_vol в monthly_features.parquet.

    Args:
        raw: Словарь данных из load_data().
    """
    returns = raw["imoex_daily"]["close"].pct_change().dropna()
    realized_daily = returns.rolling(20).std() * np.sqrt(252)
    realized_monthly = realized_daily.resample("ME").last().dropna()

    garch_monthly = raw["features"]["imoex_garch_vol"].dropna()

    combined = pd.DataFrame({
        "realized": realized_monthly,
        "garch": garch_monthly,
    }).dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(combined.index, combined["realized"], linewidth=1.2,
            label="Реализованная волатильность (20 дней)")
    ax.plot(combined.index, combined["garch"], linewidth=1.2, linestyle="--",
            label="Условная волатильность GARCH(1,1)")
    ax.set_title("Условная волатильность GARCH(1,1) vs реализованная волатильность IMOEX")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Волатильность, σ (аннуал.)")
    ax.legend()

    save_fig(fig, "fig_13_garch_vs_realized", subfolder="step_04")
    log.info("fig_13 сохранён")


def fig_14a_acf(raw: dict) -> None:
    """Строит автокорреляционную функцию (ACF) квадратов дневных доходностей IMOEX.

    Args:
        raw: Словарь данных из load_data().
    """
    returns = raw["imoex_daily"]["close"].pct_change().dropna()
    sq_returns = (returns ** 2).dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(sq_returns, lags=40, ax=ax, alpha=0.05)
    ax.set_title("Автокорреляционная функция (ACF) квадратов дневных доходностей IMOEX")
    ax.set_xlabel("Лаг (дни)")
    ax.set_ylabel("Коэффициент автокорреляции")

    save_fig(fig, "fig_14a_acf", subfolder="step_04")
    log.info("fig_14a сохранён")


def fig_14b_pacf(raw: dict) -> None:
    """Строит частную автокорреляционную функцию (PACF) квадратов дневных доходностей IMOEX.

    Args:
        raw: Словарь данных из load_data().
    """
    returns = raw["imoex_daily"]["close"].pct_change().dropna()
    sq_returns = (returns ** 2).dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_pacf(sq_returns, lags=40, ax=ax, alpha=0.05)
    ax.set_title("Частная АКФ (PACF) квадратов дневных доходностей IMOEX")
    ax.set_xlabel("Лаг (дни)")
    ax.set_ylabel("Коэффициент автокорреляции")

    save_fig(fig, "fig_14b_pacf", subfolder="step_04")
    log.info("fig_14b сохранён")


def fig_15_rolling_correlation(raw: dict) -> None:
    """Строит скользящую 60-дневную корреляцию доходностей IMOEX и нефти Urals.

    Args:
        raw: Словарь данных из load_data().
    """
    imoex_ret = raw["imoex_daily"]["close"].pct_change()
    urals_ret = raw["urals_daily"]["close"].pct_change()

    combined = pd.DataFrame({
        "imoex": imoex_ret,
        "urals": urals_ret,
    }).dropna()

    rolling_corr = combined["imoex"].rolling(60).corr(combined["urals"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_corr.index, rolling_corr.values, linewidth=1.2)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_title("Скользящая корреляция доходностей IMOEX и нефти Urals (окно 60 дней)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Коэффициент корреляции")

    save_fig(fig, "fig_15_rolling_correlation", subfolder="step_04")
    log.info("fig_15 сохранён")


def fig_16a_rolling_var(raw: dict) -> None:
    """Строит скользящий VaR(5%) дневных доходностей IMOEX с окном 60 дней.

    Args:
        raw: Словарь данных из load_data().
    """
    returns = raw["imoex_daily"]["close"].pct_change().dropna()
    rolling_var = returns.rolling(60).quantile(0.05)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_var.index, rolling_var.values, linewidth=1.2)
    ax.set_title("Скользящий VaR(5%) дневных доходностей IMOEX (окно 60 дней)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("VaR, д.е.")

    save_fig(fig, "fig_16a_rolling_var", subfolder="step_04")
    log.info("fig_16a сохранён")


def fig_16b_rolling_cvar(raw: dict) -> None:
    """Строит скользящий CVaR(5%) дневных доходностей IMOEX с окном 60 дней.

    CVaR вычисляется как среднее доходностей, не превышающих квантиль 5%.

    Args:
        raw: Словарь данных из load_data().
    """
    returns = raw["imoex_daily"]["close"].pct_change().dropna()
    rolling_cvar = returns.rolling(60).apply(
        lambda x: x[x <= x.quantile(0.05)].mean(),
        raw=False,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_cvar.index, rolling_cvar.values, linewidth=1.2)
    ax.set_title("Скользящий CVaR(5%) дневных доходностей IMOEX (окно 60 дней)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("CVaR, д.е.")

    save_fig(fig, "fig_16b_rolling_cvar", subfolder="step_04")
    log.info("fig_16b сохранён")


if __name__ == "__main__":
    setup_plot_style()
    raw = load_data()
    T3 = table_T3_descriptive(raw)
    T4 = table_T4_stationarity(raw)
    T5 = table_T5_garch_params(raw)
    T6 = table_T6_jarque_bera(raw)
    log.info("Все таблицы сохранены")
    fig_10_returns_histogram(raw)
    fig_11_qq_plot(raw)
    fig_12a_rolling_vol_20d(raw)
    fig_12b_rolling_vol_60d(raw)
    fig_13_garch_vs_realized(raw)
    fig_14a_acf(raw)
    fig_14b_pacf(raw)
    fig_15_rolling_correlation(raw)
    fig_16a_rolling_var(raw)
    fig_16b_rolling_cvar(raw)
    log.info("Все графики сохранены в figures/step_04/")
