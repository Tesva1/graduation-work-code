"""Шаг 8 (подготовка): запуск walk-forward HMM и сохранение режимов без look-ahead.

На каждом тестовом месяце HMM переобучается ТОЛЬКО на дневных данных
строго до начала месяца минус config.GAP_MONTHS. Запускается трижды:
  1) auto-BIC: число состояний выбирается на каждом шаге → walk_forward_regimes.parquet
  2) принудительно n=2 (основной вариант, согласован с главой 6) → walk_forward_regimes_n2.parquet
  3) принудительно n=3 (робастность, исходная типология: спокойный рост, стресс, паника) → walk_forward_regimes_n3.parquet

Запуск: python src/09a_walk_forward_regimes.py
"""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PROCESSED, TEST_END, TEST_START
from src.utils import walk_forward_regime_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("step_09a")


def _summary(df: pd.DataFrame, label: str) -> None:
    """Печатает краткую сводку по результатам walk-forward."""
    log.info("=== %s ===", label)
    log.info("Фолдов: %d", len(df))
    log.info("Распределение n_states_used:\n%s", df["n_states_used"].value_counts().sort_index().to_string())
    log.info(
        "Распределение regime_hard:\n%s",
        df["regime_hard"].value_counts().sort_index().to_string(),
    )
    failed = int((df["regime_hard"] == -1).sum())
    if failed:
        log.warning("Неудачных фолдов (regime_hard=-1): %d", failed)


def main() -> None:
    """Запускает оба варианта walk-forward HMM и сохраняет parquet-файлы."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    monthly_features = pd.read_parquet(os.path.join(DATA_PROCESSED, "monthly_features.parquet"))
    idx = monthly_features.index

    # 1) Авто-выбор n_states по BIC на каждом шаге.
    log.info("Запуск walk-forward HMM (n_states по BIC)...")
    df_auto = walk_forward_regime_predictions(idx, TEST_START, TEST_END, n_states=None)
    out_auto = os.path.join(DATA_PROCESSED, "walk_forward_regimes.parquet")
    df_auto.to_parquet(out_auto)
    log.info("Сохранено: %s (shape=%s)", out_auto, df_auto.shape)
    _summary(df_auto, "auto-BIC")

    # 2) Принудительно n_states=2 — основной вариант, согласован с главой 6.
    log.info("Запуск walk-forward HMM (forced n_states=2)...")
    df_n2 = walk_forward_regime_predictions(idx, TEST_START, TEST_END, n_states=2)
    out_n2 = os.path.join(DATA_PROCESSED, "walk_forward_regimes_n2.parquet")
    df_n2.to_parquet(out_n2)
    log.info("Сохранено: %s (shape=%s)", out_n2, df_n2.shape)
    _summary(df_n2, "forced n=2 (основной)")

    # 3) Принудительно n_states=3 — робастность, исходная 3-режимная типология.
    log.info("Запуск walk-forward HMM (forced n_states=3)...")
    df_n3 = walk_forward_regime_predictions(idx, TEST_START, TEST_END, n_states=3)
    out_n3 = os.path.join(DATA_PROCESSED, "walk_forward_regimes_n3.parquet")
    df_n3.to_parquet(out_n3)
    log.info("Сохранено: %s (shape=%s)", out_n3, df_n3.shape)
    _summary(df_n3, "forced n=3 (робастность)")


if __name__ == "__main__":
    main()
