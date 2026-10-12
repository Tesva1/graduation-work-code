"""
Конфигурация проекта.
Все параметры — здесь и только здесь. Никаких магических чисел в коде скриптов.

Дипломная работа: «Режимно-адаптивный анализ фондового рынка РФ
с применением методов машинного обучения»
"""

import os

# === Пути ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_VALIDATION = os.path.join(PROJECT_ROOT, "data", "validation")
FIGURES = os.path.join(PROJECT_ROOT, "figures")
TABLES = os.path.join(PROJECT_ROOT, "tables")
MODELS = os.path.join(PROJECT_ROOT, "models")
RESULTS = os.path.join(PROJECT_ROOT, "results")
SRC = os.path.join(PROJECT_ROOT, "src")

# === Период исследования ===
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
RANDOM_STATE = 42

# === Walk-forward валидация ===
TRAIN_START = "2015-01"
TEST_START = "2020-02"
TEST_END = "2025-12"
GAP_MONTHS = 1  # зазор между обучающей и тестовой выборкой

# === Технические индикаторы ===
VOLATILITY_WINDOWS = [20, 60]
RSI_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 60
GARCH_WINDOW = 252  # скользящее окно для GARCH (торговых дней)
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2

# === LSTM / TFT ===
LSTM_SEQUENCE_LENGTH = 6  # месяцев на вход последовательных моделей

# === HMM ===
HMM_N_STATES_CANDIDATES = [2, 3, 4]
HMM_N_ITER = 1000
HMM_MIN_REGIME_SIZE = 15  # fallback на baseline если < 15 наблюдений в режиме

# === Optuna ===
OPTUNA_N_TRIALS = 50
BASELINE_FAST_MODE = False
BASELINE_FAST_OPTUNA_TRIALS = 8
BASELINE_INNER_CV_SPLITS = 3
BASELINE_REFIT_TUNING_EVERY = 12  # переиспользовать гиперпараметры внутри года
BASELINE_TORCH_EPOCHS = 50
BASELINE_FAST_TORCH_EPOCHS = 15
BASELINE_TORCH_OPTUNA_TRIALS = 8
BASELINE_FAST_TORCH_OPTUNA_TRIALS = 3
BASELINE_TORCH_CV_EPOCHS = 5
BASELINE_FAST_TORCH_CV_EPOCHS = 3
BASELINE_BATCH_SIZE = 32
BASELINE_N_JOBS = 4
BASELINE_TORCH_NUM_THREADS = 1  # PyTorch LSTM на macOS CPU может падать при многопоточности
BASELINE_MODEL_NAMES = [
    "LogisticRegression",
    "SVM",
    "RandomForest",
    "XGBoost",
    "CatBoost",
    "LSTM",
    "TFT",
    "TabTransformer",
]

# === VAR / SVAR ===
VAR_MAX_LAGS = 12
SVAR_ORDER = [
    "urals_ret",
    "usdrub_change",
    "key_rate_change",
    "imoex_vol",
    "imoex_ret",
]

# === Визуализация (ГОСТ-стиль для РЭУ Плеханова) ===
FIG_DPI = 300
FIG_FONTSIZE = 12
FIG_FIGSIZE_FULL = (12, 8)
FIG_FIGSIZE_HALF = (7, 5)
FIG_FIGSIZE_WIDE = (14, 6)

# === MOEX ISS API ===
MOEX_ISS_BASE = "https://iss.moex.com/iss"
MOEX_IMOEX_URL = (
    f"{MOEX_ISS_BASE}/history/engines/stock/markets/index"
    f"/boards/SNDX/securities/IMOEX.json"
)
MOEX_PAGE_SIZE = 100

# === ЦБ РФ ===
CBR_KEY_RATE_URL = "https://cbr.ru/hd_base/KeyRate/"
CBR_USD_XML_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
CBR_USD_VCODE = "R01235"  # код валюты USD в API ЦБ

# === Кризисные даты (для вертикальных линий на графиках) ===
CRISIS_DATES = {
    "2020-03": "COVID-19",
    "2022-02": "Начало СВО",
}

# === Регимные модели (этап 8) ===
REGIME_VARIANT_A_MODEL = "SVM"  # Тип модели для варианта A (фиксированный baseline по AUC)
REGIME_VARIANT_C_MODELS = ["LogisticRegression", "SVM", "RandomForest", "XGBoost", "CatBoost"]
REGIME_INNER_CV_N_SPLITS = 3
REGIME_INNER_CV_GAP = 1

# === Этап 10/13/14: SHAP, blend, threshold tuning ===
# Все параметры walk-forward улучшения и SHAP-интерпретации — здесь.
BLEND_OPTUNA_TRIALS = 50
BLEND_MIN_TRAIN = 24                # минимум прошлых OOF-точек для подбора весов/порога
THRESHOLD_GRID_SIZE = 91            # число точек сетки порогов в [0.05, 0.95]
THRESHOLD_GRID_LOW = 0.05
THRESHOLD_GRID_HIGH = 0.95
ROLLING_AUC_WINDOW = 12             # окно для rolling AUC (для диагностики, не для p-value)
SOFTMAX_BLEND_TEMPERATURE = 8.0     # «температура» softmax(AUC * τ); прежнее значение
SHAP_BACKGROUND_SIZE = 24           # размер фона KernelExplainer
SHAP_NSAMPLES = 200                 # число samples KernelExplainer.shap_values
DM_TEST_LAG = 1                     # горизонт прогноза для Diebold-Mariano (h=1)
