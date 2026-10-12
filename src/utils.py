"""
Утилиты проекта: настройка стиля графиков, сохранение фигур и таблиц.

Стиль: строгий академический (ГОСТ), Times New Roman / DejaVu Serif,
подписи на русском языке. Оптимизирован для чёрно-белой печати.
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем корень проекта в путь, чтобы config.py был доступен
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FIGURES, TABLES, FIG_DPI


def setup_plot_style():
    """
    Устанавливает единый стиль для всех графиков проекта.

    Шрифт: Times New Roman (или DejaVu Serif как fallback).
    Сетка: лёгкая (alpha=0.3).
    Палитра: colorblind-friendly, пригодная для ч/б печати.
    """
    plt.rcParams.update({
        # Шрифт — академический serif
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
        "font.size": 12,

        # Заголовки и подписи
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Качество вывода
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,

        # Сетка
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",

        # Рамка осей
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",

        # Линии
        "lines.linewidth": 1.5,

        # Легенда
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })
    # Палитра: пригодна для дальтоников и ч/б печати
    sns.set_palette("colorblind")


def save_fig(fig, name, subfolder=None):
    """
    Сохраняет фигуру в PNG и PDF в папку figures/.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Объект фигуры для сохранения.
    name : str
        Имя файла без расширения (например, 'fig_01_imoex_dynamics').
    subfolder : str, optional
        Подпапка внутри figures/ (например, 'step_01').
    """
    base = os.path.join(FIGURES, subfolder) if subfolder else FIGURES
    os.makedirs(base, exist_ok=True)

    path = os.path.join(base, f"{name}.png")
    fig.savefig(path, format="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_table(df, name, subfolder=None):
    """
    Сохраняет DataFrame как CSV в папку tables/.

    Parameters
    ----------
    df : pandas.DataFrame
        Таблица для сохранения.
    name : str
        Имя файла без расширения (например, 'T01_descriptive_stats').
    subfolder : str, optional
        Подпапка внутри tables/.
    """
    base = os.path.join(TABLES, subfolder) if subfolder else TABLES
    os.makedirs(base, exist_ok=True)

    path = os.path.join(base, f"{name}.csv")
    df.to_csv(path, index=True, encoding="utf-8-sig")  # utf-8-sig для Excel


def add_crisis_lines(ax, crisis_dates=None):
    """
    Добавляет вертикальные линии кризисных событий на график.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Оси графика.
    crisis_dates : dict, optional
        Словарь {дата: описание}. По умолчанию — из config.py.
    """
    import pandas as pd
    from config import CRISIS_DATES

    if crisis_dates is None:
        crisis_dates = CRISIS_DATES

    for date_str, label in crisis_dates.items():
        date = pd.Timestamp(date_str)
        ax.axvline(date, color="red", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(
            date, ax.get_ylim()[1] * 0.95, label,
            rotation=90, fontsize=8, color="red",
            ha="right", va="top",
        )


# ============================================================================
# Walk-forward HMM
# ----------------------------------------------------------------------------
# Утилита переобучения HMM на каждой точке walk-forward валидации без
# look-ahead: на момент прогноза t используются только дневные данные
# строго до начала тестового месяца минус gap (config.GAP_MONTHS).
# Используется режимно-адаптивными моделями (шаг 8).
# ============================================================================

import importlib.util as _importlib_util
import logging as _logging
import time as _time
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as _np
import pandas as _pd

if TYPE_CHECKING:  # pragma: no cover
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM


_log = _logging.getLogger("walk_forward_hmm")


@lru_cache(maxsize=1)
def _load_step07_module():
    """Загружает модуль src/07_regime_detection.py для переиспользования функций.

    Имя файла начинается с цифры, поэтому стандартный import не работает —
    приходится использовать importlib.util.

    Returns:
        Загруженный модуль 07_regime_detection.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_path = os.path.join(project_root, "src", "07_regime_detection.py")
    spec = _importlib_util.spec_from_file_location("regime_detection_07", module_path)
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_daily_imoex_for_hmm() -> "_pd.DataFrame":
    """Загружает дневной IMOEX из data/raw/imoex.csv для walk-forward HMM.

    Возвращает DataFrame с DatetimeIndex и колонками close, volume, daily_ret,
    отсортированный по возрастанию даты, очищенный от дубликатов и обрезанный
    периодом из config (START_DATE..END_DATE).

    Returns:
        DataFrame[date -> close, volume, daily_ret].
    """
    from config import DATA_RAW, START_DATE, END_DATE

    path = os.path.join(DATA_RAW, "imoex.csv")
    df = _pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
    df = df.loc[START_DATE:END_DATE, ["close", "volume"]]
    df["daily_ret"] = df["close"].pct_change()
    return df


def fit_hmm_window(
    daily_df: "_pd.DataFrame",
    end_date: "_pd.Timestamp",
    n_states: "int | None" = None,
):
    """Обучает HMM на дневных данных СТРОГО до end_date (без look-ahead).

    Делегирует построение признаков и обучение в src/07_regime_detection.py:
    build_hmm_features() и select_hmm() — чтобы не дублировать научную логику.

    Args:
        daily_df: Полный DataFrame дневных данных IMOEX (close, volume, daily_ret).
        end_date: Граница «строго меньше» (exclusive) — данные >= end_date исключены.
        n_states: Если None — выбор по BIC из HMM_N_STATES_CANDIDATES;
                  иначе — обучить ровно с этим числом состояний.

    Returns:
        Кортеж (model | None, n_chosen, bic_per_state):
        - model: обученная GaussianHMM (или None, если все кандидаты не прошли валидацию)
        - n_chosen: фактически использованное число состояний
        - bic_per_state: словарь {n_states: BIC}.
    """
    from config import HMM_N_ITER, HMM_N_STATES_CANDIDATES, RANDOM_STATE
    from hmmlearn.hmm import GaussianHMM

    step07 = _load_step07_module()

    end_ts = _pd.Timestamp(end_date)
    window_df = daily_df.loc[daily_df.index < end_ts].copy()
    if len(window_df) < 60:
        _log.warning(
            "fit_hmm_window: < 60 дней до %s (%d дней), HMM пропущен",
            end_ts.date(), len(window_df),
        )
        return None, 0, {}

    X, _feature_dates = step07.build_hmm_features(window_df)

    if n_states is None:
        # Делегируем в существующий select_hmm — он перебирает кандидатов
        # из HMM_N_STATES_CANDIDATES и выбирает по BIC.
        try:
            best_model, best_n, bic_dict = step07.select_hmm(X)
        except Exception as exc:  # pragma: no cover - защитная ветка
            _log.warning("fit_hmm_window: select_hmm упал на %s: %s", end_ts.date(), exc)
            return None, 0, {}
        return best_model, best_n, bic_dict

    # Принудительный режим: обучаем ровно одну модель.
    n_features = X.shape[1]
    model = GaussianHMM(
        n_components=int(n_states),
        covariance_type="full",
        n_iter=HMM_N_ITER,
        random_state=RANDOM_STATE,
    )
    try:
        model.fit(X)
    except Exception as exc:  # pragma: no cover
        _log.warning("fit_hmm_window: fit упал на %s, n=%d: %s", end_ts.date(), n_states, exc)
        return None, int(n_states), {}

    if not step07._check_model_valid(model, X):
        _log.warning(
            "Forced n_states=%d gives degenerate regimes (some < HMM_MIN_REGIME_SIZE) at end_date=%s",
            n_states, end_ts,
        )

    log_likelihood = model.score(X)
    k = (
        n_states * n_states
        + n_states * n_features
        + n_states * (n_features * (n_features + 1) // 2)
    )
    # BIC formula: standard textbook form -2*LL + k*log(N).
    bic = -2.0 * log_likelihood + k * _np.log(len(X))
    return model, int(n_states), {int(n_states): float(bic)}


def predict_test_month_regime(
    model,
    daily_df: "_pd.DataFrame",
    test_month_start: "_pd.Timestamp",
    test_month_end: "_pd.Timestamp",
):
    """Прогнозирует распределение режима для тестового месяца.

    Берёт дневные данные в пределах [test_month_start, test_month_end] (включительно
    с обеих сторон), строит признаки тем же конвейером, что и при обучении HMM,
    и усредняет predict_proba по дням.

    Args:
        model: Обученная GaussianHMM.
        daily_df: Полный DataFrame дневных данных.
        test_month_start: Начало тестового месяца (включительно).
        test_month_end: Конец тестового месяца (включительно).

    Returns:
        Кортеж (regime_probs, regime_hard):
        - regime_probs: ndarray формы (n_states,) — средние вероятности.
        - regime_hard: int — argmax(regime_probs); -1 если нет данных.
    """
    if model is None:
        return _np.array([]), -1

    step07 = _load_step07_module()

    start_ts = _pd.Timestamp(test_month_start)
    end_ts = _pd.Timestamp(test_month_end)
    mask = (daily_df.index >= start_ts) & (daily_df.index <= end_ts)
    window_df = daily_df.loc[mask].copy()
    if len(window_df) == 0:
        n_states = int(model.n_components)
        return _np.full(n_states, 1.0 / n_states), -1

    # Для корректного расчёта rolling-волатильности нужно немного истории
    # ПЕРЕД тестовым месяцем, иначе bfill() даст NaN на первых строках.
    # Берём последние 30 дней ДО test_month_start как «контекст», но
    # predict_proba считаем только на строках самого тестового месяца.
    context_mask = (daily_df.index < start_ts)
    context_df = daily_df.loc[context_mask].tail(30)
    combined = _pd.concat([context_df, window_df]).sort_index()

    X, feature_dates = step07.build_hmm_features(combined)
    if len(X) == 0:
        n_states = int(model.n_components)
        return _np.full(n_states, 1.0 / n_states), -1

    # Берём только строки тестового месяца.
    in_test = (feature_dates >= start_ts) & (feature_dates <= end_ts)
    X_test = X[in_test]
    if len(X_test) == 0:
        n_states = int(model.n_components)
        return _np.full(n_states, 1.0 / n_states), -1

    proba = model.predict_proba(X_test)  # (n_test_days, n_states)
    regime_probs = proba.mean(axis=0)
    regime_hard = int(_np.argmax(regime_probs))
    return regime_probs, regime_hard


def walk_forward_regime_predictions(
    monthly_features_index: "_pd.DatetimeIndex",
    test_start: str,
    test_end: str,
    n_states: "int | None" = None,
) -> "_pd.DataFrame":
    """Walk-forward переобучение HMM и прогноз режима на каждый тестовый месяц.

    Для каждого месяца t в диапазоне [test_start, test_end]:
      1. cutoff = начало t минус GAP_MONTHS (по календарю);
      2. HMM обучается на всех дневных данных СТРОГО ДО cutoff;
      3. для t рассчитывается P(regime | данные месяца t) и hard label.

    Если n_states is None — на каждом шаге выбирается лучшее число состояний
    по BIC из HMM_N_STATES_CANDIDATES (число режимов может варьироваться от
    fold к fold). Если n_states задан — все шаги используют ровно его.

    Если HMM-фит на конкретном фолде падает (или данных мало) — строка
    заполняется regime_hard=-1, prob-колонки = NaN.

    Args:
        monthly_features_index: Индекс месячных фичей (концы месяцев).
        test_start: Строка вида "YYYY-MM" — начало тестового периода.
        test_end: Строка вида "YYYY-MM" — конец тестового периода.
        n_states: Принудительное число состояний или None (BIC).

    Returns:
        DataFrame с индексом по тестовым месяцам и колонками:
            regime_hard (int), regime_prob_0..regime_prob_{N-1} (float),
            n_states_used (int), bic_chosen (float).
    """
    from config import GAP_MONTHS
    from tqdm import tqdm

    daily_df = load_daily_imoex_for_hmm()

    test_months = _pd.period_range(test_start, test_end, freq="M")
    monthly_periods = monthly_features_index.to_period("M") if len(monthly_features_index) else None

    rows: list[dict] = []
    max_states = 0
    t_start = _time.time()

    for tm in tqdm(test_months, desc="walk-forward HMM", total=len(test_months)):
        # Тестовый месяц как timestamps.
        month_ts = tm.to_timestamp(how="start")
        month_end_ts = tm.to_timestamp(how="end")
        cutoff = month_ts - _pd.DateOffset(months=GAP_MONTHS)

        # Если для этого месяца нет соответствующей строки в monthly_features —
        # пропускаем (мы предсказываем только то, что есть в индексе фичей).
        if monthly_periods is not None and tm not in monthly_periods:
            continue
        # Берём timestamp из monthly_features_index, соответствующий месяцу tm.
        idx_ts = monthly_features_index[monthly_periods == tm][0]

        model, n_used, bic_dict = fit_hmm_window(daily_df, cutoff, n_states=n_states)
        if model is None:
            row = {
                "index": idx_ts,
                "regime_hard": -1,
                "n_states_used": int(n_used) if n_used else (int(n_states) if n_states else 0),
                "bic_chosen": _np.nan,
                "end_date_used": cutoff,
            }
            rows.append(row)
            continue

        probs, hard = predict_test_month_regime(model, daily_df, month_ts, month_end_ts)
        row = {
            "index": idx_ts,
            "regime_hard": int(hard),
            "n_states_used": int(n_used),
            "bic_chosen": float(bic_dict.get(n_used, _np.nan)) if bic_dict else _np.nan,
            "end_date_used": cutoff,
        }
        for j, p in enumerate(probs):
            row[f"regime_prob_{j}"] = float(p)
        max_states = max(max_states, len(probs))
        rows.append(row)

    # Гарантируем равное число prob-колонок.
    for row in rows:
        for j in range(max_states):
            row.setdefault(f"regime_prob_{j}", _np.nan)

    elapsed = _time.time() - t_start
    _log.info(
        "walk_forward_regime_predictions: %d фолдов за %.1f с (n_states=%s)",
        len(rows), elapsed, "BIC" if n_states is None else str(n_states),
    )

    if not rows:
        return _pd.DataFrame()

    df = _pd.DataFrame(rows).set_index("index").sort_index()
    df.index.name = "date"
    return df


# ============================================================================
# Walk-forward / leakage-safe shared utilities
# ----------------------------------------------------------------------------
# Эти утилиты используются этапами 10/13/14 (SHAP, blend, threshold tuning)
# для обеспечения strict no-look-ahead. См. plan/dazzling-humming-nova.md.
# ============================================================================

def expanding_walk_forward_indices(n: int, min_train: int):
    """Итератор (train_slice, predict_idx) для expanding-window walk-forward.

    Для каждого индекса i, начиная с min_train, возвращает (slice(0, i), i).
    Гарантия: предсказание в точке i делается только по данным `[:i]`.

    Args:
        n: Длина серии (число точек).
        min_train: Минимум точек обучения перед первым предсказанием.

    Yields:
        Кортеж (slice объект, целочисленный индекс).
    """
    for i in range(min_train, n):
        yield slice(0, i), i


def max_drawdown(returns):
    """Максимальная просадка серии месячных доходностей.

    Args:
        returns: pandas.Series — месячные доходности (NaN заменяются на 0).

    Returns:
        float ≤ 0; равно 0 для пустой серии.
    """
    r = returns.fillna(0) if hasattr(returns, "fillna") else _pd.Series(returns).fillna(0)
    if len(r) == 0:
        return 0.0
    equity = (1 + r).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def per_fold_log_loss(y, pred):
    """Per-fold (per-month) log-loss для DM-теста.

    Каждый месяц теста — это отдельный fold (1 наблюдение). Log-loss для
    одного наблюдения: L = -[y log p + (1-y) log(1-p)] с клиппингом p,
    чтобы избежать -inf на крайних значениях.
    """
    df = _pd.concat(
        [y.rename("y"), pred.rename("p")], axis=1, join="inner",
    ).dropna()
    p = df["p"].clip(1e-6, 1 - 1e-6)
    return -(df["y"] * _np.log(p) + (1 - df["y"]) * _np.log(1 - p))


def diebold_mariano(loss_a, loss_b, h: int = 1):
    """Diebold-Mariano test на разнице потерь loss_a - loss_b.

    Для h=1 (1-step ahead, как в дипломе: прогноз на месяц вперёд) HAC-поправка
    не нужна; для h>1 — Newey-West-стиль с весами Бартлетта. p-value
    двусторонний (тест различия). Если выборка < 4 точек — (nan, 1.0).

    Положительная статистика → loss_a > loss_b (модель A хуже модели B).
    """
    from scipy.stats import norm

    df = _pd.concat(
        [loss_a.rename("a"), loss_b.rename("b")], axis=1, join="inner",
    ).dropna()
    if len(df) < 4:
        return float("nan"), 1.0
    d = df["a"].values - df["b"].values
    n = len(d)
    d_mean = d.mean()
    var_d = float(_np.var(d, ddof=1))
    if h > 1:
        for k in range(1, h):
            gamma_k = float(_np.cov(d[k:], d[:-k], ddof=1)[0, 1])
            var_d += 2.0 * (1.0 - k / h) * gamma_k
    if var_d <= 0:
        return float("nan"), 1.0
    dm_stat = d_mean / _np.sqrt(var_d / n)
    p_value = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down корректировка p-значений.

    Возвращает скорректированные p-values в исходном порядке.
    NaN остаётся NaN (выпадает из ранжирования).
    """
    p_list = list(p_values)
    n = len(p_list)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_list), key=lambda kv: (_np.isnan(kv[1]), kv[1]))
    adjusted = [0.0] * n
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        if _np.isnan(p):
            adjusted[orig_idx] = float("nan")
            continue
        adj = (n - rank + 1) * p
        adj = min(adj, 1.0)
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = cum_max
    return adjusted


def rolling_auc(y, pred, window: int):
    """Rolling AUC-ROC по фиксированному окну.

    Окна с одним классом пропускаются. Используется ТОЛЬКО для диагностических
    графиков; не использовать для статистических тестов (окна перекрываются,
    нарушение i.i.d. → завышенная значимость).

    Args:
        y: pandas.Series целевой переменной (0/1).
        pred: pandas.Series предсказанных вероятностей.
        window: размер окна в наблюдениях.

    Returns:
        pandas.Series rolling AUC, индексированная датой конца окна.
    """
    from sklearn.metrics import roc_auc_score

    df = _pd.concat(
        [y.rename("y"), pred.rename("pred")], axis=1, join="inner",
    ).dropna()
    vals: list[float] = []
    dates: list = []
    for end in range(window, len(df) + 1):
        part = df.iloc[end - window : end]
        if part["y"].nunique() < 2:
            continue
        vals.append(float(roc_auc_score(part["y"], part["pred"])))
        dates.append(part.index[-1])
    return _pd.Series(vals, index=dates, name="rolling_auc")
