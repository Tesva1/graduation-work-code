"""Чистые функции расчёта признаков для шага 2 (feature engineering).

Все функции принимают pandas.Series/DataFrame и возвращают то же,
не делая I/O.
"""
from __future__ import annotations

import numpy as np

# Человеческие названия признаков для подписей на графиках (русский язык)
FEATURE_LABELS: dict[str, str] = {
    "imoex_ret_1m":                 "Доходность IMOEX за 1 месяц",
    "imoex_ret_3m":                 "Доходность IMOEX за 3 месяца",
    "imoex_ret_6m":                 "Доходность IMOEX за 6 месяцев",
    "imoex_ret_12m":                "Доходность IMOEX за 12 месяцев",
    "imoex_ret_lastweek":           "Доходность IMOEX за последнюю неделю месяца",
    "imoex_vol_20d":                "Волатильность IMOEX 20 дней (аннуал.)",
    "imoex_vol_60d":                "Волатильность IMOEX 60 дней (аннуал.)",
    "imoex_vol_ratio_20_60":        "Отношение волатильности 20д / 60д",
    "imoex_max_drawdown_intramonth":"Макс. просадка IMOEX внутри месяца",
    "imoex_skew_20d":               "Асимметрия дневных доходностей IMOEX (20д)",
    "imoex_garch_vol":              "Условная волатильность GARCH(1,1)",
    "imoex_volume_ratio_20_60":     "Отношение объёма торгов 20д / 60д",
    "imoex_log_turnover":           "Лог. среднедневного оборота IMOEX",
    "imoex_rsi_14":                 "RSI(14) индекса IMOEX",
    "imoex_bollinger_pos":          "Позиция IMOEX в полосах Боллинджера",
    "imoex_sma_ratio":              "SMA(20) / SMA(60) − 1 (трендовый сигнал)",
    "urals_ret_1m":                 "Доходность нефти Urals за 1 месяц",
    "urals_ret_3m":                 "Доходность нефти Urals за 3 месяца",
    "urals_brent_spread":           "Дисконт Urals к Brent (спред / Brent)",
    "usdrub_change":                "Изменение курса USD/RUB за месяц",
    "key_rate_level":               "Ключевая ставка ЦБ РФ, %",
    "key_rate_change":              "Изменение ключевой ставки ЦБ РФ, п.п.",
    "rgbi_ret_1m":                  "Доходность индекса гособлигаций RGBI",
    "urals_x_usdrub":               "Нефть Urals × изменение USD/RUB (взаимодействие)",
    "imoex_excess_return":          "Премия за риск IMOEX над ставкой ЦБ",
}

# Подписи оси Y для каждого признака (с единицами измерения)
FEATURE_YLABEL: dict[str, str] = {
    "imoex_ret_1m":                 "Доходность, д.е.",
    "imoex_ret_3m":                 "Доходность, д.е.",
    "imoex_ret_6m":                 "Доходность, д.е.",
    "imoex_ret_12m":                "Доходность, д.е.",
    "imoex_ret_lastweek":           "Доходность, д.е.",
    "imoex_vol_20d":                "Волатильность, σ (аннуал.)",
    "imoex_vol_60d":                "Волатильность, σ (аннуал.)",
    "imoex_vol_ratio_20_60":        "Отношение волатильностей, д.е.",
    "imoex_max_drawdown_intramonth":"Просадка, д.е.",
    "imoex_skew_20d":               "Асимметрия, д.е.",
    "imoex_garch_vol":              "Условная волатильность, σ",
    "imoex_volume_ratio_20_60":     "Отношение объёмов, д.е.",
    "imoex_log_turnover":           "ln(оборот + 1), руб.",
    "imoex_rsi_14":                 "RSI, пунктов (0–100)",
    "imoex_bollinger_pos":          "Позиция, д.е. (0 = нижняя, 1 = верхняя полоса)",
    "imoex_sma_ratio":              "SMA(20) / SMA(60) − 1, д.е.",
    "urals_ret_1m":                 "Доходность, д.е.",
    "urals_ret_3m":                 "Доходность, д.е.",
    "urals_brent_spread":           "Дисконт, д.е.",
    "usdrub_change":                "Изменение курса, д.е.",
    "key_rate_level":               "Ставка, % годовых",
    "key_rate_change":              "Изменение, п.п.",
    "rgbi_ret_1m":                  "Доходность, д.е.",
    "urals_x_usdrub":               "Взаимодействие, д.е.",
    "imoex_excess_return":          "Премия, д.е.",
}
import pandas as pd


def monthly_simple_return(prices: pd.Series, n_months: int) -> pd.Series:
    """Простая доходность за n месяцев: P_t / P_{t-n} - 1.

    prices.index — DatetimeIndex с месячной частотой.
    Первые n элементов — NaN.
    """
    if n_months <= 0:
        raise ValueError("n_months must be positive")
    return prices.pct_change(n_months)


def last_week_return(daily_prices: pd.Series, month_end: pd.Timestamp, window: int = 5) -> float:
    """Доходность за последние `window` торговых дней до month_end включительно.

    Берём tail(window+1) из daily_prices.loc[:month_end].
    Если меньше window+1 значений — возвращаем float('nan').
    Возвращает float: sliced.iloc[-1] / sliced.iloc[0] - 1
    """
    subset = daily_prices.loc[:month_end]
    sliced = subset.tail(window + 1)
    if len(sliced) < window + 1:
        return float("nan")
    return float(sliced.iloc[-1] / sliced.iloc[0] - 1)


def annualised_volatility(
    daily_returns: pd.Series, window: int, trading_days: int = 252
) -> pd.Series:
    """Аннуализированная скользящая волатильность: rolling(window).std() * sqrt(trading_days).

    Первые window-1 элементов — NaN.
    """
    return daily_returns.rolling(window).std() * np.sqrt(trading_days)


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (EWM сглаживание с alpha=1/period).

    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    Первые `period` элементов — NaN.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    return out


def bollinger_position(
    prices: pd.Series, window: int = 20, n_std: float = 2.0
) -> pd.Series:
    """Позиция цены в полосах Боллинджера: (P - lower) / (upper - lower).

    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = upper - lower
    Когда width == 0 — NaN (используй .where(width > 0)).
    """
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = upper - lower
    position = (prices - lower) / width
    return position.where(width > 0)


def garch_conditional_volatility_at(
    daily_returns: pd.Series,
    dates: pd.DatetimeIndex,
    window: int = 252,
    random_state: int = 42,
) -> pd.Series:
    """На каждую дату t в `dates`:
      - берём tail(window) из daily_returns.loc[:t]
      - если меньше window строк → NaN
      - иначе обучаем arch_model(window_data * 100, vol='Garch', p=1, q=1,
        mean='Constant', dist='t')
      - fit(disp='off', show_warning=False)
      - берём последнее conditional_volatility / 100
      - при любом исключении — fallback: window_data.tail(20).std()
    Возвращает pd.Series(out, name='garch_vol').reindex(dates)
    """
    from arch import arch_model

    np.random.seed(random_state)

    out: dict[pd.Timestamp, float] = {}
    for t in dates:
        window_data = daily_returns.loc[:t].tail(window)
        if len(window_data) < window:
            out[t] = float("nan")
            continue
        try:
            am = arch_model(
                window_data * 100,
                vol="Garch",
                p=1,
                q=1,
                mean="Constant",
                dist="t",
            )
            res = am.fit(disp="off", show_warning=False)
            cond_vol = res.conditional_volatility.iloc[-1] / 100
            out[t] = float(cond_vol)
        except Exception:
            out[t] = float(window_data.tail(20).std())

    return pd.Series(out, name="garch_vol").reindex(dates)


def intramonth_max_drawdown(daily_prices: pd.Series, month_end: pd.Timestamp) -> float:
    """Максимальная просадка внутри календарного месяца, оканчивающегося на month_end.

    month_start = month_end.replace(day=1)
    sub = daily_prices.loc[month_start:month_end]
    Если len(sub) < 2 → float('nan')
    running_max = sub.cummax()
    return float((sub / running_max - 1).min())
    """
    month_start = month_end.replace(day=1)
    sub = daily_prices.loc[month_start:month_end]
    if len(sub) < 2:
        return float("nan")
    running_max = sub.cummax()
    return float((sub / running_max - 1).min())
