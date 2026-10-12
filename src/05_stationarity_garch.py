"""Шаг 4: тесты стационарности ADF/KPSS и оценка GARCH(1,1) на дневных доходностях IMOEX.

Вход:  data/raw/imoex.csv, data/raw/urals.csv, data/raw/usdrub.csv, data/raw/key_rate.csv
Выход: data/processed/garch_conditional_vol.parquet — дневная условная волатильность

Запуск: python src/05_stationarity_garch.py
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PROCESSED, DATA_RAW, END_DATE, RANDOM_STATE, START_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_05")


def load_daily(fname: str, price_col: str) -> pd.Series:
    """Загружает дневной ценовой ряд из CSV-файла.

    Args:
        fname: Имя CSV-файла в директории data/raw (например, «imoex.csv»).
        price_col: Название столбца с ценой.

    Returns:
        Отсортированный дедуплицированный ряд значений за период START_DATE:END_DATE
        с DatetimeIndex по столбцу «date».

    Raises:
        FileNotFoundError: Если файл не найден в DATA_RAW.
        KeyError: Если столбец price_col отсутствует в файле.
    """
    path = os.path.join(DATA_RAW, fname)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").set_index("date")
    df = df.loc[START_DATE:END_DATE]
    return df[price_col]


def run_stationarity(series: pd.Series, label: str) -> None:
    """Выполняет тесты стационарности ADF и KPSS и логирует результаты.

    Тест ADF: H0 — ряд нестационарен (наличие единичного корня).
    Тест KPSS: H0 — ряд стационарен.
    Совместная интерпретация:
      - Оба указывают на стационарность → СТАЦИОНАРНЫЙ.
      - Оба указывают на нестационарность → НЕСТАЦИОНАРНЫЙ.
      - Результаты противоречат друг другу → ПРОТИВОРЕЧИВО.

    Args:
        series: Временной ряд для тестирования.
        label: Человекочитаемое название ряда для вывода в лог.
    """
    clean = series.dropna()
    log.info("=== Тест стационарности: %s (n=%d) ===", label, len(clean))

    adf_stat, adf_p, _, _, adf_crit, _ = adfuller(clean, autolag="AIC")
    adf_stationary = adf_p < 0.05
    log.info(
        "ADF: stat=%.4f  p=%.4f  критич. 5%%=%.4f → %s",
        adf_stat,
        adf_p,
        adf_crit["5%"],
        "стационарный" if adf_stationary else "нестационарный",
    )

    kpss_stat, kpss_p, _, kpss_crit = kpss(clean, regression="c", nlags="auto")
    kpss_stationary = kpss_p > 0.05
    log.info(
        "KPSS: stat=%.4f  p=%.4f  критич. 5%%=%.4f → %s",
        kpss_stat,
        kpss_p,
        kpss_crit["5%"],
        "стационарный" if kpss_stationary else "нестационарный",
    )

    if adf_stationary and kpss_stationary:
        verdict = "СТАЦИОНАРНЫЙ"
    elif not adf_stationary and not kpss_stationary:
        verdict = "НЕСТАЦИОНАРНЫЙ"
    else:
        verdict = "ПРОТИВОРЕЧИВО"

    log.info("Вывод: %s", verdict)


def fit_garch(returns: pd.Series) -> object:
    """Оценивает GARCH(1,1) с t-распределением на дневных доходностях IMOEX.

    Доходности масштабируются на 100 перед передачей в arch_model, поскольку
    библиотека arch работает стабильнее с процентными доходностями.

    Args:
        returns: Ряд дневных логарифмических доходностей IMOEX.

    Returns:
        Объект ARCHModelResult с результатами оценки.
    """
    np.random.seed(RANDOM_STATE)
    am = arch_model(returns.dropna() * 100, vol="Garch", p=1, q=1, mean="Constant", dist="t")
    res = am.fit(disp="off", show_warning=False)

    params = res.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]
    persistence = alpha + beta

    half_life = np.log(0.5) / np.log(persistence) if persistence < 1.0 else np.inf

    log.info("=== GARCH(1,1) — результаты оценки ===")
    log.info("omega    = %.6f", omega)
    log.info("alpha[1] = %.6f", alpha)
    log.info("beta[1]  = %.6f", beta)
    log.info("persistence (alpha+beta) = %.6f", persistence)
    log.info("half-life = %.2f дней", half_life)
    log.info("AIC = %.4f", res.aic)
    log.info("BIC = %.4f", res.bic)

    if persistence >= 1.0:
        log.warning("Persistence >= 1: IGARCH-эффект, волатильность не убывает к среднему")

    return res


def save_conditional_vol(res: object, returns_index: pd.DatetimeIndex) -> None:
    """Извлекает условную волатильность из результатов GARCH и сохраняет в parquet.

    Условная волатильность делится на 100, чтобы вернуть её к исходному масштабу
    (доходностей, не процентов).

    Args:
        res: Объект ARCHModelResult с оценёнными параметрами.
        returns_index: DatetimeIndex исходного ряда доходностей для совмещения индексов.
    """
    cond_vol_raw = res.conditional_volatility / 100.0
    aligned_index = returns_index.dropna() if hasattr(returns_index, "dropna") else returns_index

    vol_series = pd.Series(
        cond_vol_raw.values,
        index=cond_vol_raw.index,
        name="garch_cond_vol",
    )
    vol_series = vol_series.reindex(aligned_index)

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    out_path = os.path.join(DATA_PROCESSED, "garch_conditional_vol.parquet")
    vol_series.to_frame().to_parquet(out_path)

    log.info("Условная волатильность сохранена: %s", out_path)
    log.info("Shape: %s  |  NaN: %d  |  диапазон: %.6f – %.6f",
             vol_series.shape,
             vol_series.isna().sum(),
             vol_series.dropna().min(),
             vol_series.dropna().max())


def main() -> None:
    """Основная функция: загрузка данных, тесты стационарности, оценка GARCH.

    Последовательность шагов:
    1. Загрузка дневных данных IMOEX, Urals, USD/RUB, ключевой ставки.
    2. Расчёт дневных логарифмических доходностей IMOEX.
    3. Тесты стационарности ADF/KPSS для пяти рядов.
    4. Оценка GARCH(1,1) на дневных доходностях IMOEX.
    5. Сохранение условной волатильности в data/processed/garch_conditional_vol.parquet.
    """
    log.info("Шаг 5: стационарность и GARCH(1,1) — начало")

    imoex_close = load_daily("imoex.csv", "close")
    imoex_returns = np.log(imoex_close / imoex_close.shift(1)).dropna()
    log.info("IMOEX: %d торговых дней, доходности: %d наблюдений", len(imoex_close), len(imoex_returns))

    urals_close = load_daily("urals.csv", "close")
    log.info("Urals: %d наблюдений", len(urals_close))

    usdrub_close = load_daily("usdrub.csv", "close")
    log.info("USD/RUB: %d наблюдений", len(usdrub_close))

    key_rate = load_daily("key_rate.csv", "rate")
    log.info("Ключевая ставка: %d наблюдений", len(key_rate))

    run_stationarity(imoex_close, "IMOEX (уровни)")
    run_stationarity(imoex_returns, "IMOEX (дневные доходности)")
    run_stationarity(urals_close, "Urals (уровни)")
    run_stationarity(usdrub_close, "USD/RUB (уровни)")
    run_stationarity(key_rate, "Ключевая ставка ЦБ (уровни)")

    garch_result = fit_garch(imoex_returns)

    save_conditional_vol(garch_result, imoex_returns.index)

    log.info("Шаг 5: завершён успешно")


if __name__ == "__main__":
    main()
