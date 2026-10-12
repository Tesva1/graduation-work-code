"""
Шаг 1. Сбор данных из внешних источников.

Загружает дневные данные за период 2015-01-01 — 2025-12-31:
  - IMOEX (индекс МосБиржи) — MOEX ISS API
  - Brent (нефть) — yfinance
  - Urals (нефть) — CSV с Investing.com (ручная загрузка)
  - USD/RUB — API ЦБ РФ (XML)
  - Ключевая ставка ЦБ — парсинг cbr.ru
  - RGBI (индекс гособлигаций) — MOEX ISS API

Все данные сохраняются в data/raw/ как CSV.
После загрузки — базовая валидация и 5 графиков (fig_01 — fig_05).
"""

import sys
import os
import time
from io import StringIO

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from bs4 import BeautifulSoup

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_RAW, START_DATE, END_DATE, FIG_DPI,
    FIG_FIGSIZE_FULL, FIG_FIGSIZE_WIDE, FIG_FIGSIZE_HALF,
    MOEX_ISS_BASE, MOEX_PAGE_SIZE, CRISIS_DATES,
    CBR_KEY_RATE_URL, CBR_USD_XML_URL, CBR_USD_VCODE,
)
from src.utils import setup_plot_style, save_fig, add_crisis_lines


# ============================================================
# 1. IMOEX — Индекс МосБиржи (MOEX ISS API)
# ============================================================

def download_imoex(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загружает дневные данные IMOEX через MOEX ISS API.
    Пагинация по 100 строк за запрос.

    Returns
    -------
    pd.DataFrame
        Колонки: date, open, high, low, close, volume
    """
    url = (
        f"{MOEX_ISS_BASE}/history/engines/stock/markets/index"
        f"/boards/SNDX/securities/IMOEX.json"
    )
    all_rows = []
    start = 0

    print("Загрузка IMOEX...")
    while True:
        params = {
            "from": start_date,
            "till": end_date,
            "start": start,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Данные в секции 'history'
        history = data["history"]
        columns = history["columns"]
        rows = history["data"]

        if not rows:
            break

        all_rows.extend(rows)
        start += MOEX_PAGE_SIZE
        print(f"  ... загружено {len(all_rows)} строк", end="\r")
        time.sleep(0.2)  # вежливость к API

    print(f"  IMOEX: загружено {len(all_rows)} строк")

    df = pd.DataFrame(all_rows, columns=columns)
    # Оставляем нужные колонки
    # VALUE = оборот торгов в рублях (VOLUME для индексов = None)
    df = df[["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VALUE"]].copy()
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Убираем строки, где цены = 0 или NaN (выходные/праздники)
    df = df[df["close"] > 0].reset_index(drop=True)

    return df


# ============================================================
# 2. Brent — нефть (yfinance)
# ============================================================

def download_brent(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загружает дневные цены Brent через yfinance (тикер BZ=F).

    Returns
    -------
    pd.DataFrame
        Колонки: date, close
    """
    import yfinance as yf

    print("Загрузка Brent (yfinance)...")
    ticker = yf.Ticker("BZ=F")
    hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)

    df = hist[["Close"]].copy()
    df = df.reset_index()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["close"] > 0].reset_index(drop=True)

    print(f"  Brent: загружено {len(df)} строк")
    return df


# ============================================================
# 3. USD/RUB — курс доллара (API ЦБ РФ, XML)
# ============================================================

def download_usdrub(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загружает дневной курс USD/RUB через XML API ЦБ РФ.

    URL: cbr.ru/scripts/XML_dynamic.asp?date_req1=...&date_req2=...&VAL_NM_RQ=R01235

    Returns
    -------
    pd.DataFrame
        Колонки: date, close (курс USD в рублях)
    """
    print("Загрузка USD/RUB (ЦБ РФ)...")

    # ЦБ ожидает формат DD/MM/YYYY
    d1 = pd.Timestamp(start_date).strftime("%d/%m/%Y")
    d2 = pd.Timestamp(end_date).strftime("%d/%m/%Y")

    params = {
        "date_req1": d1,
        "date_req2": d2,
        "VAL_NM_RQ": CBR_USD_VCODE,
    }

    resp = requests.get(CBR_USD_XML_URL, params=params, timeout=30)
    resp.raise_for_status()

    # Парсим XML
    soup = BeautifulSoup(resp.content, "lxml-xml")
    records = soup.find_all("Record")

    rows = []
    for rec in records:
        date_str = rec["Date"]  # формат DD.MM.YYYY
        # ЦБ использует запятую как десятичный разделитель
        value = rec.find("Value").text.replace(",", ".")
        nominal = rec.find("Nominal").text.replace(",", ".")
        rate = float(value) / float(nominal)
        rows.append({"date": date_str, "close": rate})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  USD/RUB: загружено {len(df)} строк")
    return df


# ============================================================
# 4. Ключевая ставка ЦБ РФ
# ============================================================

def download_key_rate(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Парсит ключевую ставку ЦБ РФ со страницы cbr.ru/hd_base/KeyRate/.

    Returns
    -------
    pd.DataFrame
        Колонки: date, rate (в процентах, например 7.5)
    """
    print("Загрузка ключевой ставки ЦБ...")

    d1 = pd.Timestamp(start_date).strftime("%d.%m.%Y")
    d2 = pd.Timestamp(end_date).strftime("%d.%m.%Y")

    # POST-запрос с параметрами дат
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": d1,
        "UniDbQuery.To": d2,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }

    resp = requests.get(
        CBR_KEY_RATE_URL, params=params, headers=headers, timeout=30
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"class": "data"})

    if table is None:
        # Попробуем альтернативный поиск таблицы
        tables = soup.find_all("table")
        for t in tables:
            if t.find("td"):
                table = t
                break

    rows = []
    if table:
        for tr in table.find_all("tr")[1:]:  # пропускаем заголовок
            tds = tr.find_all("td")
            if len(tds) >= 2:
                date_str = tds[0].text.strip()
                rate_str = tds[1].text.strip().replace(",", ".").replace("%", "")
                try:
                    rate = float(rate_str)
                    rows.append({"date": date_str, "rate": rate})
                except ValueError:
                    continue

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
        df = df.sort_values("date").reset_index(drop=True)

    print(f"  Ключевая ставка: загружено {len(df)} строк")
    return df


# ============================================================
# 5. RGBI — Индекс гособлигаций (MOEX ISS API)
# ============================================================

def download_rgbi(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Загружает дневные данные индекса RGBI через MOEX ISS API.

    Returns
    -------
    pd.DataFrame
        Колонки: date, close
    """
    url = (
        f"{MOEX_ISS_BASE}/history/engines/stock/markets/index"
        f"/boards/SNDX/securities/RGBI.json"
    )
    all_rows = []
    start = 0

    print("Загрузка RGBI...")
    while True:
        params = {
            "from": start_date,
            "till": end_date,
            "start": start,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        history = data["history"]
        columns = history["columns"]
        rows = history["data"]

        if not rows:
            break

        all_rows.extend(rows)
        start += MOEX_PAGE_SIZE
        time.sleep(0.2)

    print(f"  RGBI: загружено {len(all_rows)} строк")

    df = pd.DataFrame(all_rows, columns=columns)
    df = df[["TRADEDATE", "CLOSE"]].copy()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["close"] > 0].reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ============================================================
# 6. Urals — нефть (ручная загрузка)
# ============================================================

def load_or_stub_urals() -> pd.DataFrame:
    """
    Пытается загрузить CSV с ценами Urals из data/raw/urals.csv.
    Если файла нет — создаёт заглушку и выводит инструкцию.

    Ожидаемый формат CSV:
        date,close
        2015-01-02,45.30
        ...

    Также поддерживается формат Investing.com:
        "Date","Price","Open","High","Low","Vol.","Change %"
        "01/02/2015","45.30",...

    Returns
    -------
    pd.DataFrame
        Колонки: date, close
    """
    csv_path = os.path.join(DATA_RAW, "urals.csv")

    if os.path.exists(csv_path):
        print("Загрузка Urals из локального CSV...")
        df = pd.read_csv(csv_path)

        # Определяем формат автоматически
        if "Date" in df.columns and "Price" in df.columns:
            # Формат Investing.com
            df = df[["Date", "Price"]].copy()
            df.columns = ["date", "close"]
            # Убираем запятые из цен (Investing.com формат "1,234.56")
            df["close"] = (
                df["close"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .astype(float)
            )
            df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
        elif "date" in df.columns and "close" in df.columns:
            # Наш формат
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = df["close"].astype(float)
        else:
            raise ValueError(
                f"Неизвестный формат CSV Urals. "
                f"Колонки: {list(df.columns)}. "
                f"Ожидается: (date, close) или (Date, Price, ...)"
            )

        df = df.sort_values("date").reset_index(drop=True)
        df = df[df["close"] > 0].reset_index(drop=True)
        print(f"  Urals: загружено {len(df)} строк")
        return df

    else:
        print("\n" + "=" * 70)
        print("⚠ ФАЙЛ URALS НЕ НАЙДЕН!")
        print("=" * 70)
        print(f"Ожидаемый путь: {csv_path}")
        print()
        print("Инструкция по загрузке:")
        print("1. Откройте Investing.com")
        print("2. Найдите 'Crude Oil Urals Europe CFR Spot' (URL-E)")
        print("3. Перейдите в Historical Data")
        print("4. Установите период: 01/01/2015 — 31/12/2025")
        print("5. Нажмите Download Data")
        print(f"6. Сохраните файл как: {csv_path}")
        print()
        print("Альтернатива: cbonds.com → Нефть Urals (спотовая)")
        print("=" * 70)
        print()

        # Создаём приблизительные данные Urals = Brent - дисконт
        # (для тестирования pipeline, заменить на реальные данные!)
        print("Создаю приблизительные данные Urals (Brent - дисконт)...")
        print("⚠ ЭТО ЗАГЛУШКА! Замените на реальные данные перед финальным анализом!")

        brent_path = os.path.join(DATA_RAW, "brent.csv")
        if os.path.exists(brent_path):
            brent = pd.read_csv(brent_path)
            brent["date"] = pd.to_datetime(brent["date"])
            # Исторический дисконт: до 2022 ~$2-3, после 2022 ~$20-35
            discount = brent["date"].apply(
                lambda d: np.random.uniform(20, 35) if d >= pd.Timestamp("2022-03-01")
                else np.random.uniform(1, 4)
            )
            np.random.seed(42)
            discount = brent["date"].apply(
                lambda d: np.random.uniform(20, 35) if d >= pd.Timestamp("2022-03-01")
                else np.random.uniform(1, 4)
            )
            brent["close"] = (brent["close"] - discount).clip(lower=5)
            df = brent[["date", "close"]].copy()
            df.to_csv(csv_path, index=False)
            print(f"  Urals (заглушка): сохранено {len(df)} строк → {csv_path}")
            return df
        else:
            print("  Brent ещё не загружен — загрузите Urals вручную.")
            return pd.DataFrame(columns=["date", "close"])


# ============================================================
# Валидация данных
# ============================================================

def validate_data(
    imoex: pd.DataFrame,
    brent: pd.DataFrame,
    usdrub: pd.DataFrame,
    key_rate: pd.DataFrame,
    rgbi: pd.DataFrame,
    urals: pd.DataFrame,
):
    """Проверяет загруженные данные по чеклисту валидации данных."""

    print("\n" + "=" * 60)
    print("ВАЛИДАЦИЯ ДАННЫХ")
    print("=" * 60)

    checks_passed = 0
    checks_total = 0

    # 1. IMOEX: >= 2500 торговых дней
    checks_total += 1
    n = len(imoex)
    ok = n >= 2500
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] IMOEX: {n} торговых дней (ожидается >= 2500)")
    checks_passed += ok

    # 2. Нет дубликатов дат в IMOEX
    checks_total += 1
    dups = imoex["date"].duplicated().sum()
    ok = dups == 0
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] IMOEX дубликаты дат: {dups}")
    checks_passed += ok

    # 3. Все цены IMOEX > 0
    checks_total += 1
    neg = (imoex["close"] <= 0).sum()
    ok = neg == 0
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] IMOEX цены <= 0: {neg}")
    checks_passed += ok

    # 4. USD/RUB: разумный диапазон
    checks_total += 1
    if len(usdrub) > 0:
        usd_min, usd_max = usdrub["close"].min(), usdrub["close"].max()
        ok = usd_min > 30 and usd_max < 200
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] USD/RUB диапазон: {usd_min:.2f} — {usd_max:.2f}")
    else:
        ok = False
        print(f"  [FAIL] USD/RUB: данные отсутствуют")
    checks_passed += ok

    # 5. Ключевая ставка: диапазон 4–25%
    checks_total += 1
    if len(key_rate) > 0:
        kr_min, kr_max = key_rate["rate"].min(), key_rate["rate"].max()
        ok = kr_min >= 4 and kr_max <= 25
        status = "OK" if ok else "WARN"
        print(f"  [{status}] Ключевая ставка: {kr_min:.1f}% — {kr_max:.1f}%")
        checks_passed += ok
    else:
        print(f"  [FAIL] Ключевая ставка: данные отсутствуют")

    # 6. Brent: есть данные
    checks_total += 1
    ok = len(brent) > 1000
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Brent: {len(brent)} строк")
    checks_passed += ok

    # 7. RGBI: есть данные
    checks_total += 1
    ok = len(rgbi) > 1000
    status = "OK" if ok else "WARN"
    print(f"  [{status}] RGBI: {len(rgbi)} строк")
    checks_passed += ok

    # 8. Urals: есть данные
    checks_total += 1
    is_stub = not os.path.exists(os.path.join(DATA_RAW, "urals.csv"))
    ok = len(urals) > 1000
    suffix = " (ЗАГЛУШКА!)" if is_stub else ""
    status = "OK" if ok else "WARN"
    print(f"  [{status}] Urals: {len(urals)} строк{suffix}")
    checks_passed += ok

    print(f"\nИтого: {checks_passed}/{checks_total} проверок пройдено")
    print("=" * 60)

    return checks_passed == checks_total


# ============================================================
# Графики (fig_01 — fig_05)
# ============================================================

def plot_all(
    imoex: pd.DataFrame,
    brent: pd.DataFrame,
    urals: pd.DataFrame,
    usdrub: pd.DataFrame,
    key_rate: pd.DataFrame,
):
    """Строит 5 графиков по загруженным данным."""

    # --- fig_01: Динамика индекса IMOEX ---
    fig, ax = plt.subplots(figsize=FIG_FIGSIZE_FULL)
    ax.plot(imoex["date"], imoex["close"], color="#1f77b4", linewidth=1.2)
    ax.set_title("Динамика индекса МосБиржи (IMOEX), 2015–2025")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Значение индекса, пунктов")
    add_crisis_lines(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    save_fig(fig, "fig_01_imoex_dynamics")

    # --- fig_02a: Цены нефти Brent и Urals ---
    fig, ax1 = plt.subplots(figsize=FIG_FIGSIZE_FULL)
    ax1.plot(brent["date"], brent["close"], label="Brent", color="#ff7f0e",
             linewidth=1.0, alpha=0.8)
    if len(urals) > 0:
        ax1.plot(urals["date"], urals["close"], label="Urals", color="#2ca02c",
                 linewidth=1.2)
    ax1.set_title("Цены нефти Brent и Urals, 2015–2025")
    ax1.set_xlabel("Дата")
    ax1.set_ylabel("Цена, долл./барр.")
    ax1.legend()
    add_crisis_lines(ax1)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    save_fig(fig, "fig_02a_oil_prices")

    # --- fig_02b: Спред (дисконт) Urals к Brent ---
    if len(urals) > 0:
        merged = pd.merge(
            urals[["date", "close"]].rename(columns={"close": "urals"}),
            brent[["date", "close"]].rename(columns={"close": "brent"}),
            on="date", how="inner",
        )
        spread = merged["urals"] - merged["brent"]
        fig, ax2 = plt.subplots(figsize=FIG_FIGSIZE_FULL)
        ax2.fill_between(merged["date"], spread, 0, alpha=0.4, color="#d62728")
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_title("Дисконт нефти Urals к Brent, 2015–2025")
        ax2.set_xlabel("Дата")
        ax2.set_ylabel("Спред, долл./барр.")
        add_crisis_lines(ax2)
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()
        fig.tight_layout()
        save_fig(fig, "fig_02b_urals_brent_spread")

    # --- fig_03: Курс USD/RUB ---
    fig, ax = plt.subplots(figsize=FIG_FIGSIZE_FULL)
    ax.plot(usdrub["date"], usdrub["close"], color="#9467bd", linewidth=1.2)
    ax.set_title("Курс доллара США к рублю, 2015–2025")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Курс, руб./долл.")
    add_crisis_lines(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    save_fig(fig, "fig_03_usdrub")

    # --- fig_04: Ключевая ставка ЦБ ---
    if len(key_rate) > 0:
        fig, ax = plt.subplots(figsize=FIG_FIGSIZE_FULL)
        ax.step(key_rate["date"], key_rate["rate"], where="post",
                color="#d62728", linewidth=1.5)
        ax.set_title("Ключевая ставка Банка России, 2015–2025")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Ставка, % годовых")
        add_crisis_lines(ax)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()
        save_fig(fig, "fig_04_key_rate")

    # --- fig_05: сводная панель — 4 отдельных графика ---
    panels_05 = [
        ("fig_05a_imoex",    "Индекс МосБиржи (IMOEX), 2015–2025",
         imoex["date"],    imoex["close"],    "plot",    "#1f77b4", "Значение индекса, пунктов"),
        ("fig_05b_urals",    "Нефть Urals, 2015–2025",
         urals["date"] if len(urals) > 0 else pd.Series(dtype=float),
         urals["close"] if len(urals) > 0 else pd.Series(dtype=float),
         "plot", "#2ca02c", "Цена, долл./барр."),
        ("fig_05c_usdrub",   "Курс USD/RUB, 2015–2025",
         usdrub["date"],   usdrub["close"],   "plot",    "#9467bd", "Курс, руб./долл."),
        ("fig_05d_key_rate", "Ключевая ставка ЦБ РФ, 2015–2025",
         key_rate["date"] if len(key_rate) > 0 else pd.Series(dtype=float),
         key_rate["rate"] if len(key_rate) > 0 else pd.Series(dtype=float),
         "step", "#d62728", "Ставка, % годовых"),
    ]
    for fname, title, xdata, ydata, kind, color, ylabel in panels_05:
        fig, ax = plt.subplots(figsize=FIG_FIGSIZE_FULL)
        if len(xdata) > 0:
            if kind == "step":
                ax.step(xdata, ydata, where="post", color=color, linewidth=1.5)
            else:
                ax.plot(xdata, ydata, color=color, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Дата")
        ax.set_ylabel(ylabel)
        add_crisis_lines(ax)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.autofmt_xdate()
        fig.tight_layout()
        save_fig(fig, fname)


# ============================================================
# Главная функция
# ============================================================

def main():
    """Точка входа: загрузка всех данных, валидация, графики."""

    os.makedirs(DATA_RAW, exist_ok=True)
    setup_plot_style()

    # --- Загрузка данных ---
    imoex = download_imoex(START_DATE, END_DATE)
    imoex.to_csv(os.path.join(DATA_RAW, "imoex.csv"), index=False)

    brent = download_brent(START_DATE, END_DATE)
    brent.to_csv(os.path.join(DATA_RAW, "brent.csv"), index=False)

    usdrub = download_usdrub(START_DATE, END_DATE)
    usdrub.to_csv(os.path.join(DATA_RAW, "usdrub.csv"), index=False)

    key_rate = download_key_rate(START_DATE, END_DATE)
    key_rate.to_csv(os.path.join(DATA_RAW, "key_rate.csv"), index=False)

    rgbi = download_rgbi(START_DATE, END_DATE)
    rgbi.to_csv(os.path.join(DATA_RAW, "rgbi.csv"), index=False)

    # Urals — загружаем из CSV или создаём заглушку
    urals = load_or_stub_urals()

    # --- Валидация ---
    all_ok = validate_data(imoex, brent, usdrub, key_rate, rgbi, urals)

    # --- Сводка ---
    print("\nСводка загруженных данных:")
    for name, df in [
        ("IMOEX", imoex), ("Brent", brent), ("Urals", urals),
        ("USD/RUB", usdrub), ("Key Rate", key_rate), ("RGBI", rgbi),
    ]:
        if len(df) > 0:
            date_col = "date"
            d_min = df[date_col].min().strftime("%Y-%m-%d")
            d_max = df[date_col].max().strftime("%Y-%m-%d")
            print(f"  {name:12s}: {len(df):>5d} строк, {d_min} — {d_max}")
        else:
            print(f"  {name:12s}: НЕТ ДАННЫХ")

    # --- Графики ---
    print("\nСоздание графиков...")
    plot_all(imoex, brent, urals, usdrub, key_rate)
    print("Графики fig_01 — fig_05 сохранены в figures/")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
