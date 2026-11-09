"""Módulo de evaluación y benchmarking de modelos predictivos (Supervisados y No Supervisados)
para Mantenimiento Predictivo Industrial.
"""

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Modelos Supervisados
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Modelos No Supervisados (Detección de Anomalías)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Métricas de evaluación
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
)


def run_model_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    contamination: float = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Ejecuta una comparativa completa entre modelos supervisados y no supervisados.

    Args:
        X_train: Matriz de características de entrenamiento.
        y_train: Vector objetivo de entrenamiento (1 = fallo, 0 = normal).
        X_test: Matriz de características de test (conjunto ciego).
        y_test: Vector objetivo de test.
        contamination: Proporción esperada de anomalías (si es None, usa y_train.mean()).
        random_state: Semilla para reproducibilidad.

    Returns:
        pd.DataFrame con las métricas de rendimiento ordenadas por PR-AUC.
    """
    if contamination is None:
        contamination = float(y_train.mean())

    # Preparamos variables escaladas para modelos basados en distancias (SVM, LOF, LogReg)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ==============================================================
    # 1. MODELOS SUPERVISADOS (Clasificación con etiquetas de fallo)
    # ==============================================================
    supervised_models = {
        "Regresión Logística": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Árbol de Decisión": DecisionTreeClassifier(
            class_weight="balanced", max_depth=6, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=100, random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=random_state
        ),
    }

    for name, model in supervised_models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        results.append(
            {
                "Enfoque": "Supervisado",
                "Modelo": name,
                "PR-AUC": average_precision_score(y_test, y_prob),
                "ROC-AUC": roc_auc_score(y_test, y_prob),
                "Recall": recall_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
            }
        )

    # ==============================================================
    # 2. MODELOS NO SUPERVISADOS (Detección de Anomalías sin etiquetas)
    # ==============================================================

    # A) Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_test)
    iso_pred = (iso.predict(X_test) == -1).astype(int)
    results.append(
        {
            "Enfoque": "No Supervisado",
            "Modelo": "Isolation Forest",
            "PR-AUC": average_precision_score(y_test, iso_scores),
            "ROC-AUC": roc_auc_score(y_test, iso_scores),
            "Recall": recall_score(y_test, iso_pred),
            "Precision": precision_score(y_test, iso_pred),
            "F1-Score": f1_score(y_test, iso_pred),
        }
    )

    # B) One-Class SVM
    oc_svm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
    oc_svm.fit(X_train_scaled)
    svm_scores = -oc_svm.decision_function(X_test_scaled)
    svm_pred = (oc_svm.predict(X_test_scaled) == -1).astype(int)
    results.append(
        {
            "Enfoque": "No Supervisado",
            "Modelo": "One-Class SVM",
            "PR-AUC": average_precision_score(y_test, svm_scores),
            "ROC-AUC": roc_auc_score(y_test, svm_scores),
            "Recall": recall_score(y_test, svm_pred),
            "Precision": precision_score(y_test, svm_pred),
            "F1-Score": f1_score(y_test, svm_pred),
        }
    )

    # C) Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(
        n_neighbors=20, contamination=contamination, novelty=True
    )
    lof.fit(X_train_scaled)
    lof_scores = -lof.decision_function(X_test_scaled)
    lof_pred = (lof.predict(X_test_scaled) == -1).astype(int)
    results.append(
        {
            "Enfoque": "No Supervisado",
            "Modelo": "Local Outlier Factor (LOF)",
            "PR-AUC": average_precision_score(y_test, lof_scores),
            "ROC-AUC": roc_auc_score(y_test, lof_scores),
            "Recall": recall_score(y_test, lof_pred),
            "Precision": precision_score(y_test, lof_pred),
            "F1-Score": f1_score(y_test, lof_pred),
        }
    )

    df_results = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
    return df_results


def plot_model_benchmark(
    df_benchmark: pd.DataFrame,
) -> Tuple[plt.Figure, np.ndarray]:
    """Genera las gráficas comparativas del benchmark de modelos."""
    palette = {"Supervisado": "#2b5c8f", "No Supervisado": "#e67e22"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: PR-AUC
    sns.barplot(
        data=df_benchmark,
        x="PR-AUC",
        y="Modelo",
        hue="Enfoque",
        palette=palette,
        dodge=False,
        ax=axes[0],
    )
    axes[0].set_title(
        "Rendimiento en Desbalance: PR-AUC (Mayor es mejor)", fontweight="bold"
    )
    axes[0].set_xlim(0, 1.05)
    for i, v in enumerate(df_benchmark["PR-AUC"]):
        axes[0].text(v + 0.02, i, f"{v:.3f}", va="center", fontweight="bold")

    # Gráfico 2: Trade-off Recall vs Precision
    sns.scatterplot(
        data=df_benchmark,
        x="Recall",
        y="Precision",
        hue="Enfoque",
        style="Enfoque",
        palette=palette,
        s=220,
        ax=axes[1],
    )
    for _, row in df_benchmark.iterrows():
        axes[1].text(
            row["Recall"] + 0.015,
            row["Precision"] + 0.015,
            row["Modelo"],
            fontsize=9,
            fontweight="bold",
        )

    axes[1].set_title(
        "Compromiso Operativo: Recall vs Precision en Test", fontweight="bold"
    )
    axes[1].set_xlim(0.1, 1.0)
    axes[1].set_ylim(0.1, 1.05)

    plt.tight_layout()
    return fig, axes
