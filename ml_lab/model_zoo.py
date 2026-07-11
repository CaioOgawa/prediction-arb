"""
model_zoo.py
Define todos os modelos disponíveis para comparação no ML Lab.

Organizado em dois grupos:
  - Baseline: Logistic Regression (referência mínima)
  - Boosting:  XGBoost e LightGBM (com versões calibradas para melhorar probabilidades)

LSTM e Transformer ficam reservados para a Fase 3 com dados de séries temporais (CLOB).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_baseline_models() -> dict:
    """Modelos de referência — rápidos de treinar, servem como piso de performance."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1_000,
            C=1.0,
            random_state=42,
        ),
        # LR calibrada — melhora a qualidade das probabilidades (Brier Score)
        "logistic_calibrated": CalibratedClassifierCV(
            LogisticRegression(max_iter=1_000, random_state=42),
            method="isotonic",
            cv=3,
        ),
    }


def get_boosting_models() -> dict:
    """
    Modelos de gradient boosting — geralmente top performers em dados tabulares.
    Inclui versões calibradas para predições probabilísticas mais confiáveis.
    """
    return {
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        # Versões calibradas — melhora calibração de probabilidade (Brier Score menor)
        "xgboost_calibrated": CalibratedClassifierCV(
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            method="isotonic",
            cv=3,
        ),
        "lightgbm_calibrated": CalibratedClassifierCV(
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            method="isotonic",
            cv=3,
        ),
    }


def get_all_models() -> dict:
    """Retorna todos os modelos para comparação completa."""
    models = {}
    models.update(get_baseline_models())
    models.update(get_boosting_models())
    return models
