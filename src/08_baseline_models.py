"""Шаг 7: базовые ML-модели (7 алгоритмов) с walk-forward валидацией.

Вход:  data/processed/monthly_features.parquet, monthly_target.parquet
Выход: data/processed/baseline_predictions.parquet — предсказания всех моделей
       figures/step_08/ (12 PNG)
       tables/T14–T17 (CSV)

Запуск: python src/08_baseline_models.py
"""
from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import ttest_rel
import torch
import torch.nn as nn
import optuna
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASELINE_BATCH_SIZE,
    BASELINE_FAST_MODE,
    BASELINE_FAST_OPTUNA_TRIALS,
    BASELINE_FAST_TORCH_CV_EPOCHS,
    BASELINE_FAST_TORCH_OPTUNA_TRIALS,
    BASELINE_FAST_TORCH_EPOCHS,
    BASELINE_INNER_CV_SPLITS,
    BASELINE_MODEL_NAMES,
    BASELINE_N_JOBS,
    BASELINE_REFIT_TUNING_EVERY,
    BASELINE_TORCH_EPOCHS,
    BASELINE_TORCH_CV_EPOCHS,
    BASELINE_TORCH_NUM_THREADS,
    BASELINE_TORCH_OPTUNA_TRIALS,
    DATA_PROCESSED,
    DATA_RAW,
    GAP_MONTHS,
    LSTM_SEQUENCE_LENGTH,
    OPTUNA_N_TRIALS,
    RANDOM_STATE,
    TEST_END,
    TEST_START,
    TRAIN_START,
)
from src.utils import save_fig, save_table, setup_plot_style
from src.feature_utils import FEATURE_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("step_08")

MODEL_NAMES = BASELINE_MODEL_NAMES
TORCH_MODEL_NAMES = {"LSTM", "TFT", "TabTransformer"}
OPTIONAL_MODEL_NAMES = {"CatBoost"}
FAST_MODE = BASELINE_FAST_MODE or os.getenv("BASELINE_FAST", "0") == "1"
VERBOSE_PROGRESS = os.getenv("BASELINE_VERBOSE", "1") != "0"
DEVICE_REQUEST = os.getenv("BASELINE_DEVICE", "auto").lower()
N_TRIALS = BASELINE_FAST_OPTUNA_TRIALS if FAST_MODE else OPTUNA_N_TRIALS
TORCH_EPOCHS = BASELINE_FAST_TORCH_EPOCHS if FAST_MODE else BASELINE_TORCH_EPOCHS
TORCH_TUNE_TRIALS = BASELINE_FAST_TORCH_OPTUNA_TRIALS if FAST_MODE else BASELINE_TORCH_OPTUNA_TRIALS
TORCH_CV_EPOCHS = BASELINE_FAST_TORCH_CV_EPOCHS if FAST_MODE else BASELINE_TORCH_CV_EPOCHS
OPTUNA_HISTORY: list[dict[str, object]] = []
TFT_SALIENCY_CACHE: tuple[list[str], list[str], np.ndarray] | None = None

SUBFOLDER = "step_08"

TORCH_NUM_THREADS = max(1, int(os.getenv("BASELINE_TORCH_THREADS", BASELINE_TORCH_NUM_THREADS)))
torch.set_num_threads(TORCH_NUM_THREADS)
try:
    torch.set_num_interop_threads(TORCH_NUM_THREADS)
except RuntimeError:
    # PyTorch разрешает менять interop threads только до первого параллельного запуска.
    pass


def choose_torch_device() -> torch.device:
    """Выбирает устройство PyTorch с env override для диагностики."""
    if DEVICE_REQUEST == "cpu":
        return torch.device("cpu")
    if DEVICE_REQUEST == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        log.warning("BASELINE_DEVICE=mps задан, но MPS недоступен; fallback на CPU")
        return torch.device("cpu")
    if DEVICE_REQUEST == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("BASELINE_DEVICE=cuda задан, но CUDA недоступна; fallback на CPU")
        return torch.device("cpu")
    return (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )


DEVICE = choose_torch_device()


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch модели
# ──────────────────────────────────────────────────────────────────────────────


class SimpleLSTM(nn.Module):
    """Простая LSTM-сеть для бинарной классификации временных рядов."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Инициализация LSTM.

        Args:
            input_size: Размер входного вектора признаков.
            hidden_size: Размер скрытого слоя LSTM.
            num_layers: Количество слоёв LSTM.
            dropout: Вероятность dropout (применяется при num_layers > 1).
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход.

        Args:
            x: Тензор формы (batch, seq_len, input_size).

        Returns:
            Вероятности класса 1, форма (batch, 1).
        """
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :]))


class SimpleTFT(nn.Module):
    """Упрощённый Temporal Fusion Transformer на основе TransformerEncoder."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Инициализация SimpleTFT.

        Args:
            input_size: Размер входного вектора признаков.
            d_model: Размерность модели Transformer.
            nhead: Количество голов внимания.
            num_layers: Количество слоёв TransformerEncoder.
            dropout: Вероятность dropout.
        """
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход.

        Args:
            x: Тензор формы (batch, seq_len, input_size).

        Returns:
            Вероятности класса 1, форма (batch, 1).
        """
        x = self.embedding(x)
        x = self.transformer(x)
        return torch.sigmoid(self.fc(x[:, -1, :]))


class SimpleTabTransformer(nn.Module):
    """Упрощённый TabTransformer для табличных данных без последовательностей."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Инициализация SimpleTabTransformer.

        Args:
            input_size: Размер входного вектора признаков.
            d_model: Размерность модели Transformer.
            nhead: Количество голов внимания.
            num_layers: Количество слоёв TransformerEncoder.
            dropout: Вероятность dropout.
        """
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход.

        Args:
            x: Тензор формы (batch, features).

        Returns:
            Вероятности класса 1, форма (batch, 1).
        """
        x = self.embedding(x.unsqueeze(1))
        x = self.transformer(x)
        return torch.sigmoid(self.fc(x[:, 0, :]))


def optional_dependency_available(model_name: str) -> bool:
    """Проверяет доступность опциональных библиотек без жёсткого импорта."""
    if model_name == "CatBoost":
        try:
            import catboost  # noqa: F401

            return True
        except Exception:
            return False
    return True


def active_model_names() -> list[str]:
    """Возвращает модели, которые можно обучить в текущем окружении."""
    names: list[str] = []
    for model_name in MODEL_NAMES:
        if FAST_MODE and model_name in {"SVM", "LSTM", "TFT", "TabTransformer"}:
            log.info("FAST_MODE: модель %s пропущена", model_name)
            continue
        if model_name in OPTIONAL_MODEL_NAMES and not optional_dependency_available(model_name):
            log.warning("Модель %s пропущена: пакет не установлен", model_name)
            continue
        names.append(model_name)
    return names


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Загружает признаки, целевую переменную и месячные доходности IMOEX.

    Returns:
        Кортеж (X, y, imoex_monthly_ret):
            X — DataFrame признаков (N, 25),
            y — Series целевой переменной,
            imoex_monthly_ret — Series месячных доходностей для расчёта стратегии.
    """
    features = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_features.parquet"))
    target = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_target.parquet"))

    y = target["y_next"].dropna().astype(int)
    X = features.loc[y.index]

    imoex = pd.read_csv(
        os.path.join(DATA_RAW, "imoex.csv"),
        parse_dates=["date"],
    )
    imoex = imoex.sort_values("date").set_index("date")["close"]
    imoex_monthly_ret = imoex.resample("ME").last().pct_change()

    return X, y, imoex_monthly_ret


def make_sequences(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Формирует скользящие последовательности для LSTM / TFT.

    Args:
        X_arr: Массив признаков формы (N, features).
        y_arr: Массив меток формы (N,).
        seq_len: Длина входной последовательности.

    Returns:
        Кортеж (Xs, ys):
            Xs формы (N - seq_len + 1, seq_len, features),
            ys формы (N - seq_len + 1,).
    """
    Xs: list[np.ndarray] = []
    ys: list[float] = []
    for i in range(seq_len, len(X_arr) + 1):
        Xs.append(X_arr[i - seq_len : i])
        ys.append(y_arr[i - 1])
    return np.array(Xs), np.array(ys)


def walk_forward_splits(
    X: pd.DataFrame,
    y: pd.Series,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Генерирует индексы расширяющегося окна для walk-forward валидации.

    Args:
        X: DataFrame признаков с DatetimeIndex.
        y: Series целевой переменной с DatetimeIndex.

    Returns:
        Список кортежей (train_idx, test_idx) — индексы DatetimeIndex.
    """
    test_months = pd.period_range(TEST_START, TEST_END, freq="M")
    y_periods = y.index.to_period("M")
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

    for test_month in test_months:
        train_end_period = test_month - (GAP_MONTHS + 1)
        train_idx = y.index[y_periods <= train_end_period]
        test_idx = y.index[y_periods == test_month]

        if len(train_idx) < 20 or len(test_idx) == 0:
            continue

        splits.append((train_idx, test_idx))

    return splits


def align_strategy_returns(
    signal_index: pd.DatetimeIndex,
    imoex_ret: pd.Series,
) -> pd.Series:
    """Выравнивает доходность стратегии с таргетом y_next.

    Сигнал на конец месяца t применяется к доходности IMOEX за месяц t+1.
    """
    future_ret = imoex_ret.shift(-1)
    return future_ret.reindex(signal_index)


def train_torch_model(
    model_class: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seq_len: int = 6,
    use_seq: bool = True,
    epochs: int = TORCH_EPOCHS,
    batch_size: int = BASELINE_BATCH_SIZE,
    lr: float = 0.001,
    log_prefix: str | None = None,
    **model_kwargs: object,
) -> np.ndarray:
    """Обучает PyTorch-модель и возвращает предсказанные вероятности для X_test.

    Args:
        model_class: Класс PyTorch-модели (SimpleLSTM, SimpleTFT или SimpleTabTransformer).
        X_train: Обучающие признаки формы (N, features).
        y_train: Метки обучающей выборки формы (N,).
        X_test: Тестовые признаки формы (M, features).
        seq_len: Длина входной последовательности для LSTM / TFT.
        use_seq: True — формировать последовательности (LSTM/TFT), False — таблица (TabTransformer).
        **model_kwargs: Дополнительные аргументы конструктора модели.

    Returns:
        Массив вероятностей класса 1 формы (M,).
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    if use_seq:
        X_seq_train, y_seq_train = make_sequences(X_train, y_train, seq_len)
        if len(X_seq_train) == 0:
            return np.full(len(X_test), 0.5)

        X_test_seq_list: list[np.ndarray] = []
        for i in range(len(X_test)):
            prefix = np.vstack([X_train[-(seq_len - 1) :], X_test[: i + 1]])[-seq_len:]
            if len(prefix) < seq_len:
                padding = np.zeros((seq_len - len(prefix), X_train.shape[1]))
                prefix = np.vstack([padding, prefix])
            X_test_seq_list.append(prefix)
        X_test_seq = np.array(X_test_seq_list)

        input_size: int = X_train.shape[1]
        model: nn.Module = model_class(input_size=input_size, **model_kwargs).to(DEVICE)
        X_t = torch.FloatTensor(X_seq_train).to(DEVICE)
        y_t = torch.FloatTensor(y_seq_train).unsqueeze(1).to(DEVICE)
    else:
        model = model_class(input_size=X_train.shape[1], **model_kwargs).to(DEVICE)
        X_t = torch.FloatTensor(X_train).to(DEVICE)
        y_t = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
        X_test_seq = X_test

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    if log_prefix and VERBOSE_PROGRESS:
        log.info(
            "%s | torch train start: epochs=%d, batch=%d, train=%d, test=%d, device=%s",
            log_prefix,
            epochs,
            batch_size,
            len(X_t),
            len(X_test_seq),
            DEVICE,
        )
    epoch_log_every = max(1, epochs // 5)
    for _epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        total_batches = len(loader)
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            if log_prefix and VERBOSE_PROGRESS and _epoch == 0:
                log.info(
                    "%s | torch epoch %d/%d batch %d/%d forward start",
                    log_prefix,
                    _epoch + 1,
                    epochs,
                    batch_idx,
                    total_batches,
                )
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if log_prefix and VERBOSE_PROGRESS and _epoch == 0:
                log.info(
                    "%s | torch epoch %d/%d batch %d/%d backward start loss=%.5f",
                    log_prefix,
                    _epoch + 1,
                    epochs,
                    batch_idx,
                    total_batches,
                    float(loss.detach().cpu()),
                )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
            if log_prefix and VERBOSE_PROGRESS and _epoch == 0:
                log.info(
                    "%s | torch epoch %d/%d batch %d/%d done",
                    log_prefix,
                    _epoch + 1,
                    epochs,
                    batch_idx,
                    total_batches,
                )
        if log_prefix and VERBOSE_PROGRESS and (
            _epoch == 0 or (_epoch + 1) % epoch_log_every == 0 or (_epoch + 1) == epochs
        ):
            log.info(
                "%s | torch epoch %d/%d loss=%.5f",
                log_prefix,
                _epoch + 1,
                epochs,
                epoch_loss / max(1, batches),
            )

    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test_seq).to(DEVICE)
        raw = model(X_test_t).squeeze().cpu()
        probs = raw.numpy()

    if log_prefix and VERBOSE_PROGRESS:
        log.info("%s | torch predict done", log_prefix)
    return probs if probs.ndim > 0 else np.array([float(probs)])


def make_estimator(model_name: str, params: dict[str, Any] | None = None) -> Any:
    """Создаёт sklearn-совместимую модель по имени."""
    params = params or default_params(model_name)

    if model_name == "LogisticRegression":
        return LogisticRegression(
            C=params["C"],
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    if model_name == "SVM":
        return SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=RANDOM_STATE,
            n_jobs=BASELINE_N_JOBS,
            class_weight="balanced_subsample",
        )
    if model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=BASELINE_N_JOBS,
        )
    if model_name == "CatBoost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params["l2_leaf_reg"],
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=BASELINE_N_JOBS,
            allow_writing_files=False,
        )
    raise ValueError(f"Неизвестная sklearn-модель: {model_name}")


def default_params(model_name: str) -> dict[str, Any]:
    """Консервативные параметры, если Optuna недоступна или CV вырожден."""
    defaults: dict[str, dict[str, Any]] = {
        "LogisticRegression": {"C": 1.0},
        "SVM": {"C": 1.0, "gamma": "scale"},
        "RandomForest": {"n_estimators": 120, "max_depth": 4, "min_samples_leaf": 2},
        "XGBoost": {
            "n_estimators": 120,
            "max_depth": 2,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "CatBoost": {
            "iterations": 120,
            "depth": 3,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
        },
        "LSTM": {"hidden_size": 24, "num_layers": 1, "lr": 0.001},
        "TFT": {"d_model": 24, "nhead": 4, "num_layers": 1, "lr": 0.001},
        "TabTransformer": {"d_model": 32, "nhead": 4, "num_layers": 1, "lr": 0.001},
    }
    return defaults[model_name].copy()


def _sample_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Пространство поиска Optuna для шага 8."""
    if model_name == "LogisticRegression":
        return {"C": trial.suggest_float("C", 0.01, 20.0, log=True)}
    if model_name == "SVM":
        return {
            "C": trial.suggest_float("C", 0.05, 20.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 60, 220, step=40),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 60, 220, step=40),
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 60, 220, step=40),
            "depth": trial.suggest_int("depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
    if model_name == "LSTM":
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 24, 32]),
            "num_layers": 1,
            "lr": trial.suggest_float("lr", 0.0005, 0.003, log=True),
        }
    if model_name == "TFT":
        return {
            "d_model": trial.suggest_categorical("d_model", [16, 24, 32]),
            "nhead": trial.suggest_categorical("nhead", [2, 4]),
            "num_layers": 1,
            "lr": trial.suggest_float("lr", 0.0005, 0.003, log=True),
        }
    if model_name == "TabTransformer":
        return {
            "d_model": trial.suggest_categorical("d_model", [16, 32, 64]),
            "nhead": trial.suggest_categorical("nhead", [2, 4]),
            "num_layers": 1,
            "lr": trial.suggest_float("lr", 0.0005, 0.003, log=True),
        }
    return default_params(model_name)


def _cv_score(y_valid: np.ndarray, probs: np.ndarray) -> float:
    """AUC, если в validation есть оба класса; иначе accuracy."""
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    if len(np.unique(y_valid)) > 1:
        return float(roc_auc_score(y_valid, probs))
    return float(accuracy_score(y_valid, probs >= 0.5))


def _torch_predict_with_params(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    params: dict[str, Any],
    epochs: int | None = None,
    log_prefix: str | None = None,
) -> np.ndarray:
    """Обучает одну из torch-моделей с заданными параметрами."""
    epochs = TORCH_EPOCHS if epochs is None else epochs
    params = params.copy()
    lr = float(params.pop("lr", 0.001))
    if model_name == "LSTM":
        return train_torch_model(
            SimpleLSTM,
            X_train,
            y_train,
            X_valid,
            seq_len=LSTM_SEQUENCE_LENGTH,
            use_seq=True,
            epochs=epochs,
            lr=lr,
            log_prefix=log_prefix,
            **params,
        )
    if model_name == "TFT":
        return train_torch_model(
            SimpleTFT,
            X_train,
            y_train,
            X_valid,
            seq_len=LSTM_SEQUENCE_LENGTH,
            use_seq=True,
            epochs=epochs,
            lr=lr,
            log_prefix=log_prefix,
            **params,
        )
    if model_name == "TabTransformer":
        return train_torch_model(
            SimpleTabTransformer,
            X_train,
            y_train,
            X_valid,
            seq_len=1,
            use_seq=False,
            epochs=epochs,
            lr=lr,
            log_prefix=log_prefix,
            **params,
        )
    raise ValueError(f"Неизвестная torch-модель: {model_name}")


def tune_model_params(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = N_TRIALS,
    test_key: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Подбирает гиперпараметры только на train-данных через TimeSeriesSplit."""
    label = f"{test_key.date() if test_key is not None else 'unknown'} | {model_name}"
    actual_trials = TORCH_TUNE_TRIALS if model_name in TORCH_MODEL_NAMES else n_trials
    if len(np.unique(y_train)) < 2 or len(y_train) < BASELINE_INNER_CV_SPLITS + 8:
        log.info(
            "TUNE %s | пропуск Optuna: train=%d, классы=%d; беру default params",
            label,
            len(y_train),
            len(np.unique(y_train)),
        )
        return default_params(model_name)

    n_splits = min(BASELINE_INNER_CV_SPLITS, max(2, len(y_train) // 18))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    tune_t0 = time.time()
    log.info(
        "TUNE %s | start: trials=%d, cv_splits=%d, train=%d, torch_epochs_per_cv=%d",
        label,
        actual_trials,
        n_splits,
        len(y_train),
        TORCH_CV_EPOCHS if model_name in TORCH_MODEL_NAMES else 0,
    )

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, model_name)
        scores: list[float] = []
        if VERBOSE_PROGRESS:
            log.info(
                "TUNE %s | trial %d/%d start params=%s",
                label,
                trial.number + 1,
                actual_trials,
                params,
            )
        for split_idx, (tr_pos, val_pos) in enumerate(splitter.split(X_train), start=1):
            if len(np.unique(y_train[tr_pos])) < 2:
                continue
            if model_name in TORCH_MODEL_NAMES:
                if VERBOSE_PROGRESS:
                    log.info(
                        "TUNE %s | trial %d/%d split %d/%d torch fit: train=%d, valid=%d",
                        label,
                        trial.number + 1,
                        actual_trials,
                        split_idx,
                        n_splits,
                        len(tr_pos),
                        len(val_pos),
                    )
                probs = _torch_predict_with_params(
                    model_name,
                    X_train[tr_pos],
                    y_train[tr_pos],
                    X_train[val_pos],
                    params,
                    epochs=TORCH_CV_EPOCHS,
                    log_prefix=f"TUNE {label} | trial {trial.number + 1}/{actual_trials} split {split_idx}/{n_splits}",
                )
            else:
                model = make_estimator(model_name, params)
                model.fit(X_train[tr_pos], y_train[tr_pos])
                probs = model.predict_proba(X_train[val_pos])[:, 1]
            score = _cv_score(y_train[val_pos], probs)
            scores.append(score)
            if VERBOSE_PROGRESS and model_name in TORCH_MODEL_NAMES:
                log.info(
                    "TUNE %s | trial %d/%d split %d/%d score=%.4f",
                    label,
                    trial.number + 1,
                    actual_trials,
                    split_idx,
                    n_splits,
                    score,
                )
        mean_score = float(np.mean(scores)) if scores else 0.5
        if VERBOSE_PROGRESS:
            log.info(
                "TUNE %s | trial %d/%d done mean_score=%.4f",
                label,
                trial.number + 1,
                actual_trials,
                mean_score,
            )
        return mean_score

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def trial_callback(study_obj: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        log.info(
            "TUNE %s | progress: trial %d/%d value=%s best=%.4f elapsed=%.1fs",
            label,
            trial.number + 1,
            actual_trials,
            "nan" if trial.value is None else f"{trial.value:.4f}",
            study_obj.best_value,
            time.time() - tune_t0,
        )

    study.optimize(objective, n_trials=actual_trials, show_progress_bar=False, callbacks=[trial_callback])
    for trial in study.trials:
        row: dict[str, object] = {
            "Дата теста": test_key,
            "Модель": model_name,
            "trial": trial.number,
            "value": trial.value,
        }
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        OPTUNA_HISTORY.append(row)
    best_params = default_params(model_name)
    best_params.update(study.best_params)
    log.info(
        "TUNE %s | done: best=%.4f, params=%s, elapsed=%.1fs",
        label,
        study.best_value,
        best_params,
        time.time() - tune_t0,
    )
    return best_params


def predict_one_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """Обучает модель на текущем expanding-window и возвращает вероятности."""
    if model_name in TORCH_MODEL_NAMES:
        return _torch_predict_with_params(
            model_name,
            X_train,
            y_train,
            X_test,
            params,
            log_prefix=f"FIT {model_name}",
        )
    model = make_estimator(model_name, params)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward цикл
# ──────────────────────────────────────────────────────────────────────────────


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    imoex_ret: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Запускает walk-forward для всех 7 моделей.

    Args:
        X: DataFrame признаков.
        y: Series целевой переменной.
        imoex_ret: Series месячных доходностей IMOEX.

    Returns:
        Кортеж (preds_df, timing):
            preds_df — DataFrame предсказанных вероятностей (test_months × 7 моделей),
            timing — словарь суммарного времени обучения по каждой модели.
    """
    model_names = active_model_names()
    all_preds: dict[str, dict[pd.Timestamp, float]] = {m: {} for m in model_names}
    timing: dict[str, float] = {m: 0.0 for m in model_names}
    tuned_params: dict[str, dict[str, Any]] = {}
    hyperparam_rows: list[dict[str, object]] = []

    splits = walk_forward_splits(X, y)
    log.info(
        "Walk-forward: %d тестовых фолдов, моделей: %d, Optuna trials=%d, torch_trials=%d, torch_epochs=%d, torch_cv_epochs=%d, устройство=%s, torch_threads=%d, verbose=%s",
        len(splits),
        len(model_names),
        N_TRIALS,
        TORCH_TUNE_TRIALS,
        TORCH_EPOCHS,
        TORCH_CV_EPOCHS,
        DEVICE,
        torch.get_num_threads(),
        VERBOSE_PROGRESS,
    )
    log.info("Активные модели: %s", ", ".join(model_names))

    fold_bar = tqdm(splits, desc="Walk-forward фолды", unit="фолд", ncols=100)
    for fold_idx, (train_idx, test_idx) in enumerate(fold_bar):
        fold_t0 = time.time()
        test_key: pd.Timestamp = test_idx[0]
        fold_bar.set_postfix({"месяц": str(test_key.date()), "train": len(train_idx)})
        log.info(
            "FOLD %02d/%02d | test=%s | train=%s..%s (%d obs)",
            fold_idx + 1,
            len(splits),
            test_key.date(),
            train_idx.min().date(),
            train_idx.max().date(),
            len(train_idx),
        )

        X_train_raw = X.loc[train_idx].fillna(0).values.astype(float)
        X_test_raw = X.loc[test_idx].fillna(0).values.astype(float)
        y_train = y.loc[train_idx].values.astype(int)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model_bar = tqdm(model_names, desc=f"  Модели", unit="модель", leave=False, ncols=100)
        for model_name in model_bar:
            model_bar.set_description(f"  {model_name:<20}")
            t0 = time.time()
            log.info(
                "MODEL %02d/%02d | fold %02d/%02d | %s start",
                model_names.index(model_name) + 1,
                len(model_names),
                fold_idx + 1,
                len(splits),
                model_name,
            )

            if (
                model_name not in tuned_params
                or fold_idx % max(1, BASELINE_REFIT_TUNING_EVERY) == 0
            ):
                log.info(
                    "MODEL %s | fold %02d | tuning/refit params",
                    model_name,
                    fold_idx + 1,
                )
                tuned_params[model_name] = tune_model_params(
                    model_name,
                    X_train,
                    y_train,
                    test_key=test_key,
                )
                hyperparam_rows.append(
                    {
                        "Дата теста": test_key,
                        "Модель": model_name,
                        "Гиперпараметры": repr(tuned_params[model_name]),
                        "Optuna trials": (
                            TORCH_TUNE_TRIALS
                            if model_name in TORCH_MODEL_NAMES
                            else N_TRIALS
                        ),
                    }
                )
            else:
                log.info(
                    "MODEL %s | fold %02d | reuse params from previous tuning",
                    model_name,
                    fold_idx + 1,
                )

            log.info("MODEL %s | fold %02d | final fit/predict start", model_name, fold_idx + 1)
            p = predict_one_model(model_name, X_train, y_train, X_test, tuned_params[model_name])
            pred_value = float(p[0] if hasattr(p, "__len__") else p)
            all_preds[model_name][test_key] = pred_value

            timing[model_name] += time.time() - t0
            log.info(
                "MODEL %s | fold %02d done: prob=%.4f, elapsed=%.1fs, total_model_time=%.1fs",
                model_name,
                fold_idx + 1,
                pred_value,
                time.time() - t0,
                timing[model_name],
            )

        log.info(
            "FOLD %02d/%02d done: elapsed=%.1fs",
            fold_idx + 1,
            len(splits),
            time.time() - fold_t0,
        )

    preds_df = pd.DataFrame({m: pd.Series(all_preds[m]) for m in model_names})
    preds_df.index = pd.DatetimeIndex(preds_df.index)
    hyperparams_df = pd.DataFrame(hyperparam_rows)

    return preds_df, timing, hyperparams_df


# ──────────────────────────────────────────────────────────────────────────────
# Метрики
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(
    preds_df: pd.DataFrame,
    y: pd.Series,
    imoex_ret: pd.Series,
) -> pd.DataFrame:
    """Вычисляет 6 метрик качества и доходность стратегии для каждой модели.

    Args:
        preds_df: DataFrame предсказанных вероятностей (test_months × модели).
        y: Series истинных меток.
        imoex_ret: Series месячных доходностей IMOEX.

    Returns:
        DataFrame метрик (7 моделей × 6 показателей).
    """
    results: list[dict[str, object]] = []

    for model_name in preds_df.columns:
        probs = preds_df[model_name].dropna()
        if probs.empty:
            continue
        y_true = y.loc[probs.index].astype(int)
        y_pred = (probs >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = np.nan

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)

        try:
            ll = log_loss(y_true, probs)
        except Exception:
            ll = np.nan

        strategy_ret = align_strategy_returns(probs.index, imoex_ret) * y_pred.values
        cumulative_return = float((1 + strategy_ret.fillna(0)).prod() - 1)

        results.append(
            {
                "Модель": model_name,
                "AUC-ROC": round(auc, 4),
                "Accuracy": round(acc, 4),
                "F1 macro": round(f1, 4),
                "Precision(y=1)": round(prec, 4),
                "Log-Loss": round(ll, 4),
                "Доходность стратегии": round(cumulative_return, 4),
            }
        )

    return pd.DataFrame(results).set_index("Модель")


def _best_model_name(metrics_df: pd.DataFrame) -> str:
    """Возвращает название модели с наибольшим AUC-ROC.

    Args:
        metrics_df: DataFrame метрик с индексом по имени модели.

    Returns:
        Строка с названием лучшей модели.
    """
    return str(metrics_df["AUC-ROC"].idxmax())


def _model_names_from_predictions(preds_df: pd.DataFrame) -> list[str]:
    """Модели, по которым есть хотя бы одно предсказание."""
    return [m for m in preds_df.columns if preds_df[m].notna().any()]


# ──────────────────────────────────────────────────────────────────────────────
# Графики
# ──────────────────────────────────────────────────────────────────────────────


def fig_32_auc_bar(metrics_df: pd.DataFrame) -> None:
    """Столбчатая диаграмма AUC-ROC базовых моделей.

    Args:
        metrics_df: DataFrame метрик.
    """
    best = _best_model_name(metrics_df)
    auc_vals = metrics_df["AUC-ROC"]
    colors = ["#1f77b4" if m != best else "#d62728" for m in auc_vals.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(auc_vals.index, auc_vals.values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Случайный классификатор")

    for bar, val in zip(bars, auc_vals.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("AUC-ROC базовых моделей (walk-forward 2020–2025)")
    ax.set_xlabel("Модель")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.3, 0.85)
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    save_fig(fig, "fig_32_auc_bar", subfolder=SUBFOLDER)


def fig_33_metrics_heatmap(metrics_df: pd.DataFrame) -> None:
    """Тепловая карта метрик 7 моделей × 5 показателей.

    Args:
        metrics_df: DataFrame метрик.
    """
    cols = ["AUC-ROC", "Accuracy", "F1 macro", "Precision(y=1)", "Доходность стратегии"]
    heat_data = metrics_df[cols].astype(float)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heat_data,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Значение"},
    )
    ax.set_title("Сравнение метрик базовых моделей")
    ax.set_xlabel("Метрика")
    ax.set_ylabel("Модель")
    fig.tight_layout()
    save_fig(fig, "fig_33_metrics_heatmap", subfolder=SUBFOLDER)


def fig_34_roc_curves(preds_df: pd.DataFrame, y: pd.Series) -> None:
    """ROC-кривые всех 7 моделей на одном графике.

    Args:
        preds_df: DataFrame предсказанных вероятностей.
        y: Series истинных меток.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name in _model_names_from_predictions(preds_df):
        probs = preds_df[model_name].dropna()
        y_true = y.loc[probs.index].astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc_val = roc_auc_score(y_true, probs)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})", linewidth=1.8)
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Случайный классификатор")
    ax.set_title("ROC-кривые базовых моделей")
    ax.set_xlabel("Доля ложноположительных (FPR)")
    ax.set_ylabel("Доля истинноположительных (TPR)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig_34_roc_curves", subfolder=SUBFOLDER)


def fig_35_cumulative_returns(
    preds_df: pd.DataFrame,
    y: pd.Series,
    imoex_ret: pd.Series,
) -> None:
    """Накопленная доходность торговых стратегий всех 7 моделей и buy-and-hold.

    Args:
        preds_df: DataFrame предсказанных вероятностей.
        y: Series истинных меток.
        imoex_ret: Series месячных доходностей IMOEX.
    """
    common_idx = preds_df.dropna(how="all").index
    fig, ax = plt.subplots(figsize=(14, 7))

    for model_name in _model_names_from_predictions(preds_df):
        probs = preds_df[model_name].reindex(common_idx)
        y_pred = (probs >= 0.5).astype(int)
        strategy_ret = align_strategy_returns(probs.index, imoex_ret) * y_pred.values
        cum_ret = (1 + strategy_ret.fillna(0)).cumprod()
        ax.plot(cum_ret.index, cum_ret.values, label=model_name, linewidth=1.5)

    bah = (1 + align_strategy_returns(common_idx, imoex_ret).fillna(0)).cumprod()
    ax.plot(bah.index, bah.values, "k--", linewidth=2.0, label="Пассивный портфель (B&H)")

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Накопленная доходность торговых стратегий (базовые модели)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Накопленная доходность (база = 1.0)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "fig_35_cumulative_returns", subfolder=SUBFOLDER)


def fig_36_auc_over_time(preds_df: pd.DataFrame, y: pd.Series) -> None:
    """Скользящий AUC-ROC каждой модели (окно 12 месяцев).

    Args:
        preds_df: DataFrame предсказанных вероятностей.
        y: Series истинных меток.
    """
    window = 12
    fig, ax = plt.subplots(figsize=(14, 6))

    for model_name in _model_names_from_predictions(preds_df):
        probs = preds_df[model_name].dropna().sort_index()
        y_true = y.loc[probs.index].astype(int)
        rolling_auc: dict[pd.Timestamp, float] = {}

        dates = probs.index.tolist()
        for i in range(window - 1, len(dates)):
            window_dates = dates[i - window + 1 : i + 1]
            p_w = probs.loc[window_dates]
            y_w = y_true.loc[window_dates]
            if y_w.nunique() < 2:
                continue
            try:
                rolling_auc[dates[i]] = roc_auc_score(y_w, p_w)
            except Exception:
                pass

        if rolling_auc:
            s = pd.Series(rolling_auc)
            ax.plot(s.index, s.values, label=model_name, linewidth=1.5)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="Случайный классификатор")
    ax.set_title("AUC-ROC базовых моделей во времени (скользящее окно 12 мес.)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("AUC-ROC")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "fig_36_auc_over_time", subfolder=SUBFOLDER)


def fig_37_radar_top3(metrics_df: pd.DataFrame) -> None:
    """Радарный график профиля метрик трёх лучших моделей.

    Args:
        metrics_df: DataFrame метрик.
    """
    top3 = metrics_df["AUC-ROC"].nlargest(3).index.tolist()
    metric_cols = ["AUC-ROC", "Accuracy", "F1 macro", "Precision(y=1)", "Доходность стратегии"]
    metric_labels = ["AUC-ROC", "Accuracy", "F1 macro", "Precision", "Доходность"]

    data = metrics_df.loc[top3, metric_cols].astype(float)

    col_min = data.min()
    col_max = data.max()
    normalized = (data - col_min) / (col_max - col_min + 1e-9)

    n_metrics = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"polar": True})

    for model_name in top3:
        values = normalized.loc[model_name].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2.0, label=model_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Профиль метрик трёх лучших моделей", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    save_fig(fig, "fig_37_radar_top3", subfolder=SUBFOLDER)


def fig_38_calibration(preds_df: pd.DataFrame, y: pd.Series) -> None:
    """Калибровочные кривые всех 7 моделей.

    Args:
        preds_df: DataFrame предсказанных вероятностей.
        y: Series истинных меток.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name in _model_names_from_predictions(preds_df):
        probs = preds_df[model_name].dropna()
        y_true = y.loc[probs.index].astype(int)
        try:
            fraction_pos, mean_pred = calibration_curve(y_true, probs, n_bins=8)
            ax.plot(mean_pred, fraction_pos, marker="o", label=model_name, linewidth=1.5)
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Идеальная калибровка")
    ax.set_title("Калибровочные кривые базовых моделей")
    ax.set_xlabel("Средняя предсказанная вероятность")
    ax.set_ylabel("Доля истинных положительных")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig_38_calibration", subfolder=SUBFOLDER)


def fig_39_training_time(timing: dict[str, float]) -> None:
    """Столбчатая диаграмма времени обучения каждой модели.

    Args:
        timing: Словарь {название модели: суммарное время, с}.
    """
    models = list(timing.keys())
    times = [timing[m] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, times, edgecolor="black", linewidth=0.8, color="#4878d0")

    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f} с",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Время обучения базовых моделей (walk-forward, секунды)")
    ax.set_xlabel("Модель")
    ax.set_ylabel("Время, с")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    save_fig(fig, "fig_39_training_time", subfolder=SUBFOLDER)


def fig_40_auc_vs_accuracy(metrics_df: pd.DataFrame) -> None:
    """Диаграмма рассеяния AUC-ROC vs Accuracy для 7 моделей.

    Args:
        metrics_df: DataFrame метрик.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name in metrics_df.index:
        row = metrics_df.loc[model_name]
        ax.scatter(row["Accuracy"], row["AUC-ROC"], s=100, zorder=5)
        ax.annotate(
            model_name,
            (row["Accuracy"], row["AUC-ROC"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    ax.set_title("AUC-ROC vs Точность (Accuracy) базовых моделей")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("AUC-ROC")
    fig.tight_layout()
    save_fig(fig, "fig_40_auc_vs_accuracy", subfolder=SUBFOLDER)


def fig_41_feature_importance(X: pd.DataFrame, y: pd.Series) -> None:
    """Важность признаков XGBoost (gain), обученного на полной обучающей выборке.

    Args:
        X: DataFrame признаков.
        y: Series целевой переменной.
    """
    train_end = pd.Timestamp(TEST_START) - pd.DateOffset(months=GAP_MONTHS + 1)
    train_idx = y.index[y.index <= train_end]

    X_train_raw = X.loc[train_idx].fillna(0).values.astype(float)
    y_train = y.loc[train_idx].values.astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    feat_names = list(X.columns)
    feat_labels = [FEATURE_LABELS.get(f, f) for f in feat_names]

    sorted_idx = np.argsort(importances)[-15:]
    top_labels = [feat_labels[i] for i in sorted_idx]
    top_vals = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_labels)), top_vals, color="#1f77b4", edgecolor="black", linewidth=0.6)
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels, fontsize=9)
    ax.set_title("Важность признаков лучшей базовой модели (XGBoost, gain)")
    ax.set_xlabel("Важность (gain)")
    ax.set_ylabel("Признак")
    fig.tight_layout()
    save_fig(fig, "fig_41_feature_importance", subfolder=SUBFOLDER)


def fig_42_monthly_returns(
    preds_df: pd.DataFrame,
    y: pd.Series,
    imoex_ret: pd.Series,
    metrics_df: pd.DataFrame,
) -> None:
    """Месячная доходность лучшей стратегии и пассивного портфеля (столбчатая).

    Args:
        preds_df: DataFrame предсказанных вероятностей.
        y: Series истинных меток.
        imoex_ret: Series месячных доходностей IMOEX.
        metrics_df: DataFrame метрик для определения лучшей модели.
    """
    best = _best_model_name(metrics_df)
    probs = preds_df[best].dropna()
    y_pred = (probs >= 0.5).astype(int)
    bah_ret = align_strategy_returns(probs.index, imoex_ret)
    strategy_ret = bah_ret * y_pred.values

    excess_ret = strategy_ret - bah_ret
    fig, ax = plt.subplots(figsize=(14, 5))
    colors_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in excess_ret.fillna(0)]
    ax.bar(excess_ret.index, excess_ret.fillna(0).values, color=colors_bar, edgecolor="none", width=20)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Месячная разница доходности: стратегия «{best}» минус B&H")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Разница доходности, д.е.")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    save_fig(fig, "fig_42_monthly_returns", subfolder=SUBFOLDER)


def fig_43_confusion_best(
    preds_df: pd.DataFrame,
    y: pd.Series,
    metrics_df: pd.DataFrame,
) -> None:
    """Матрица ошибок лучшей базовой модели.

    Args:
        preds_df: DataFrame предсказанных вероятностей.
        y: Series истинных меток.
        metrics_df: DataFrame метрик для определения лучшей модели.
    """
    best = _best_model_name(metrics_df)
    probs = preds_df[best].dropna()
    y_true = y.loc[probs.index].astype(int)
    y_pred = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Падение (0)", "Рост (1)"],
        yticklabels=["Падение (0)", "Рост (1)"],
        linewidths=0.5,
    )
    ax.set_title(f"Матрица ошибок лучшей базовой модели ({best})")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    fig.tight_layout()
    save_fig(fig, "fig_43_confusion_best", subfolder=SUBFOLDER)


def compute_tft_saliency(X: pd.DataFrame, y: pd.Series) -> tuple[list[str], list[str], np.ndarray]:
    """Быстрая proxy-карта важности TFT на стартовой train-выборке.

    Текущая SimpleTFT не возвращает attention weights. Для дипломной визуализации
    используем устойчивую proxy-оценку: AUC однофакторного сигнала по каждому
    признаку и лагу внутри последовательности.
    """
    global TFT_SALIENCY_CACHE
    if TFT_SALIENCY_CACHE is not None:
        return TFT_SALIENCY_CACHE

    train_end_period = pd.Period(TEST_START, freq="M") - (GAP_MONTHS + 1)
    train_idx = y.index[y.index.to_period("M") <= train_end_period]
    X_train_raw = X.loc[train_idx].fillna(0).values.astype(float)
    y_train = y.loc[train_idx].values.astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_seq, y_seq = make_sequences(X_train, y_train, LSTM_SEQUENCE_LENGTH)
    if len(X_seq) == 0:
        TFT_SALIENCY_CACHE = ([], [], np.empty((0, 0)))
        return TFT_SALIENCY_CACHE

    saliency = np.zeros((LSTM_SEQUENCE_LENGTH, X_train.shape[1]), dtype=float)
    for lag in range(LSTM_SEQUENCE_LENGTH):
        for feat_idx in range(X_train.shape[1]):
            signal = X_seq[:, lag, feat_idx]
            if np.nanstd(signal) == 0 or len(np.unique(y_seq)) < 2:
                saliency[lag, feat_idx] = 0.0
                continue
            try:
                auc = roc_auc_score(y_seq, signal)
                saliency[lag, feat_idx] = abs(auc - 0.5)
            except Exception:
                saliency[lag, feat_idx] = 0.0

    lag_labels = [f"t-{LSTM_SEQUENCE_LENGTH - i - 1}" if i < LSTM_SEQUENCE_LENGTH - 1 else "t" for i in range(LSTM_SEQUENCE_LENGTH)]
    feat_labels = [FEATURE_LABELS.get(f, f) for f in X.columns]
    TFT_SALIENCY_CACHE = (lag_labels, feat_labels, saliency)
    return TFT_SALIENCY_CACHE


def fig_40_tft_attention(X: pd.DataFrame, y: pd.Series) -> None:
    """Proxy-карта внимания TFT: абсолютный градиент по лагам и признакам."""
    lag_labels, feat_labels, saliency = compute_tft_saliency(X, y)
    fig, ax = plt.subplots(figsize=(14, 6))
    if saliency.size:
        top_idx = np.argsort(saliency.mean(axis=0))[-12:]
        sns.heatmap(
            saliency[:, top_idx],
            ax=ax,
            cmap="YlGnBu",
            xticklabels=[feat_labels[i] for i in top_idx],
            yticklabels=lag_labels,
            cbar_kws={"label": "|градиент|"},
        )
        ax.set_xlabel("Признак")
        ax.set_ylabel("Лаг")
    else:
        ax.text(0.5, 0.5, "Недостаточно данных для TFT", ha="center", va="center")
    ax.set_title("TFT: proxy-карта внимания по лагам и признакам")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    save_fig(fig, "fig_40_tft_attention", subfolder=SUBFOLDER)


def fig_41_tft_variable_importance(X: pd.DataFrame, y: pd.Series) -> None:
    """Важность переменных TFT как средний абсолютный градиент."""
    _lag_labels, feat_labels, saliency = compute_tft_saliency(X, y)
    fig, ax = plt.subplots(figsize=(10, 8))
    if saliency.size:
        importance = saliency.mean(axis=0)
        top_idx = np.argsort(importance)[-15:]
        ax.barh(
            range(len(top_idx)),
            importance[top_idx],
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([feat_labels[i] for i in top_idx], fontsize=9)
        ax.set_xlabel("Средний |градиент|")
        ax.set_ylabel("Признак")
    else:
        ax.text(0.5, 0.5, "Недостаточно данных для TFT", ha="center", va="center")
    ax.set_title("TFT: важность переменных")
    fig.tight_layout()
    save_fig(fig, "fig_41_tft_variable_importance", subfolder=SUBFOLDER)


def fig_42_optuna_convergence() -> None:
    """Сходимость Optuna: накопленный лучший score по trial."""
    history = pd.DataFrame(OPTUNA_HISTORY)
    fig, ax = plt.subplots(figsize=(12, 6))
    if not history.empty:
        for model_name, grp in history.dropna(subset=["value"]).groupby("Модель"):
            values = grp["value"].astype(float).cummax().reset_index(drop=True)
            ax.plot(values.index + 1, values.values, label=model_name, linewidth=1.5)
        ax.set_xlabel("Trial во всех настройках walk-forward")
        ax.set_ylabel("Лучший inner-CV score")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "История Optuna отсутствует", ha="center", va="center")
    ax.set_title("Optuna: сходимость подбора гиперпараметров")
    fig.tight_layout()
    save_fig(fig, "fig_42_optuna_convergence", subfolder=SUBFOLDER)


def fig_43_optuna_param_importance() -> None:
    """Приближённая важность параметров Optuna через связь с inner-CV score."""
    history = pd.DataFrame(OPTUNA_HISTORY)
    rows: list[dict[str, object]] = []
    if not history.empty and "value" in history:
        param_cols = [c for c in history.columns if c.startswith("param_")]
        for model_name, grp in history.dropna(subset=["value"]).groupby("Модель"):
            for col in param_cols:
                s = grp[col].dropna()
                if len(s) < 3:
                    continue
                aligned_value = grp.loc[s.index, "value"].astype(float)
                numeric = pd.to_numeric(s, errors="coerce")
                if numeric.isna().all():
                    numeric = pd.Series(pd.factorize(s.astype(str))[0], index=s.index)
                if numeric.nunique() < 2 or aligned_value.nunique() < 2:
                    continue
                corr = abs(float(pd.Series(numeric).corr(aligned_value, method="spearman")))
                if np.isfinite(corr):
                    rows.append(
                        {
                            "Модель": model_name,
                            "Параметр": col.replace("param_", ""),
                            "Важность": corr,
                        }
                    )

    imp = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 7))
    if not imp.empty:
        plot_df = imp.sort_values("Важность", ascending=False).head(15)
        labels = plot_df["Модель"] + ": " + plot_df["Параметр"]
        ax.barh(range(len(plot_df)), plot_df["Важность"], color="#4878d0", edgecolor="black")
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("|Spearman rho| с inner-CV score")
    else:
        ax.text(0.5, 0.5, "Недостаточно trial для оценки важности", ha="center", va="center")
    ax.set_title("Optuna: важность гиперпараметров")
    fig.tight_layout()
    save_fig(fig, "fig_43_optuna_param_importance", subfolder=SUBFOLDER)


# ──────────────────────────────────────────────────────────────────────────────
# Сохранение предсказаний
# ──────────────────────────────────────────────────────────────────────────────


def save_predictions(preds_df: pd.DataFrame) -> None:
    """Сохраняет предсказания всех моделей в parquet.

    Args:
        preds_df: DataFrame предсказанных вероятностей.
    """
    path = os.path.join(DATA_PROCESSED, "baseline_predictions.parquet")
    preds_df.to_parquet(path)
    log.info("Сохранено baseline_predictions.parquet: %s", str(preds_df.shape))


# ──────────────────────────────────────────────────────────────────────────────
# Таблица T17: парные t-тесты на точность
# ──────────────────────────────────────────────────────────────────────────────


def build_t17(
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    preds_df: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Вычисляет парные t-тесты по точности между всеми парами моделей.

    Args:
        splits: Список кортежей (train_idx, test_idx).
        preds_df: DataFrame предсказанных вероятностей.
        y: Series истинных меток.

    Returns:
        DataFrame с результатами t-тестов: пары моделей, t-статистика, p-значение.
    """
    model_names = _model_names_from_predictions(preds_df)
    fold_accs: dict[str, list[int]] = {m: [] for m in model_names}

    for _train_idx, test_idx in splits:
        if len(test_idx) == 0:
            continue
        key = test_idx[0]
        if key not in preds_df.index:
            continue
        for m in model_names:
            val = preds_df.loc[key, m]
            if pd.isna(val):
                continue
            pred = int(val >= 0.5)
            true_val = int(y.loc[key])
            fold_accs[m].append(int(pred == true_val))

    rows: list[dict[str, object]] = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            a1, a2 = fold_accs[m1], fold_accs[m2]
            min_len = min(len(a1), len(a2))
            if min_len > 5:
                stat, p = ttest_rel(a1[:min_len], a2[:min_len])
                rows.append(
                    {
                        "Модель 1": m1,
                        "Модель 2": m2,
                        "t-stat": round(float(stat), 3),
                        "p-value": round(float(p), 4),
                        "Значимо (5%)": "Да" if p < 0.05 else "Нет",
                    }
                )

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Главная функция
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Точка входа: запускает полный pipeline базовых моделей."""
    setup_plot_style()

    X, y, imoex_ret = load_data()
    log.info(
        "Данные: X=%s, y=%s, тестовый период: %s → %s",
        str(X.shape),
        str(y.shape),
        TEST_START,
        TEST_END,
    )

    preds_df, timing, hyperparams = run_walk_forward(X, y, imoex_ret)
    save_predictions(preds_df)

    metrics_df = compute_metrics(preds_df, y, imoex_ret)
    log.info("Метрики:\n%s", metrics_df.to_string())

    save_table(metrics_df, "T14_baseline_metrics")

    save_table(hyperparams, "T15_hyperparams")

    timing_df = pd.DataFrame(
        {
            "Модель": list(timing.keys()),
            "Время (с)": [round(v, 2) for v in timing.values()],
        }
    )
    save_table(timing_df, "T16_compute_time")

    splits = walk_forward_splits(X, y)
    t17 = build_t17(splits, preds_df, y)
    save_table(t17, "T17_pairwise_tests")

    fig_32_auc_bar(metrics_df)
    fig_33_metrics_heatmap(metrics_df)
    fig_34_roc_curves(preds_df, y)
    fig_35_cumulative_returns(preds_df, y, imoex_ret)
    fig_36_auc_over_time(preds_df, y)
    fig_37_radar_top3(metrics_df)
    fig_38_calibration(preds_df, y)
    fig_39_training_time(timing)
    fig_40_tft_attention(X, y)
    fig_41_tft_variable_importance(X, y)
    fig_42_optuna_convergence()
    fig_43_optuna_param_importance()

    log.info("Базовые модели завершены. Лучшая модель: %s", _best_model_name(metrics_df))


if __name__ == "__main__":
    main()
