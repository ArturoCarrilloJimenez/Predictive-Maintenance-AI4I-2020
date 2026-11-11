"""Módulo de evaluación y benchmarking de modelos predictivos (Supervisados y No Supervisados)
para Mantenimiento Predictivo Industrial.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV, cross_validate
from sklearn.multioutput import MultiOutputClassifier
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
    classification_report,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
)


def get_supervised_models(random_state: int = 42) -> dict:
    """Retorna el catálogo estándar de modelos supervisados configurados para datos desbalanceados."""
    return {
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
    supervised_models = get_supervised_models(random_state=random_state)

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
    iso = IsolationForest(contamination=contamination,
                          random_state=random_state)
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


def run_supervised_cv_benchmark(
    models: dict, X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5
) -> pd.DataFrame:
    """Ejecuta 5-Fold Stratified CV sobre todos los modelos supervisados

    y reporta media ± desviación estándar.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
    }

    records = []
    for name, model in models.items():
        cv_res = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
        )

        records.append(
            {
                "Modelo": name,
                "PR-AUC (CV)": f"{np.mean(cv_res['test_pr_auc']):.4f} ± {np.std(cv_res['test_pr_auc']):.4f}",
                "ROC-AUC (CV)": f"{np.mean(cv_res['test_roc_auc']):.4f} ± {np.std(cv_res['test_roc_auc']):.4f}",
                "Recall (CV)": f"{np.mean(cv_res['test_recall']):.4f} ± {np.std(cv_res['test_recall']):.4f}",
                "Precision (CV)": f"{np.mean(cv_res['test_precision']):.4f} ± {np.std(cv_res['test_precision']):.4f}",
                "F1-Score (CV)": f"{np.mean(cv_res['test_f1']):.4f} ± {np.std(cv_res['test_f1']):.4f}",
                # Mantenemos el flotante para poder ordenar el DataFrame
                "_pr_auc_mean": np.mean(cv_res["test_pr_auc"]),
            }
        )

    df_cv = (
        pd.DataFrame(records)
        .sort_values("_pr_auc_mean", ascending=False)
        .drop(columns=["_pr_auc_mean"])
    )

    return df_cv


def categorization_error(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,  # Matriz 2D con columnas de fallos (TWF, HDF, PWF, OSF)
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    random_state: int = 42,
) -> dict:
    """Entrena y evalúa un motor multi-etiqueta para clasificar tipos de avería.

    Args:
        X_train: Features de entrenamiento (sensores + variables físicas).
        Y_train: DataFrame con las etiquetas de fallos específicos en train.
        X_test: Features de test.
        Y_test: DataFrame con las etiquetas de fallos específicos en test.
        random_state: Semilla aleatoria para reproducibilidad.

    Returns:
        dict con el modelo entrenado, predicciones binarias, probabilidades y métricas.
    """
    # 1. Definir el motor multi-salida con HistGradientBoosting balanceado
    base_clf = HistGradientBoostingClassifier(
        class_weight="balanced", random_state=random_state
    )

    # Envolverlo en TunedThresholdClassifierCV con 'f1'
    # Cada avería buscará automáticamente el umbral de probabilidad que maximice su F1-score
    tuned_estimator = TunedThresholdClassifierCV(
        estimator=base_clf,
        scoring="f1",
        cv=5,
        thresholds=100,  # Evalúa 100 cortes posibles entre 0.0 y 1.0
    )

    diagnostic_engine = MultiOutputClassifier(
        estimator=tuned_estimator,
        n_jobs=-1,
    )

    # 2. Ajuste del modelo sobre la matriz multi-etiqueta
    diagnostic_engine.fit(X_train, Y_train)

    # 3. Inferencia de clases discretas (0 o 1) y probabilidades
    y_pred = diagnostic_engine.predict(X_test)  # Array 2D (n_samples, n_fallos)

    # predict_proba en MultiOutputClassifier devuelve una lista de arrays:
    # [array_falla_1, array_falla_2, ...], cada uno con shape (n_samples, 2).
    # Tomamos la columna [:, 1] (probabilidad de fallo) de cada submodelo:
    list_probs = diagnostic_engine.predict_proba(X_test)
    y_prob = np.column_stack([p[:, 1] for p in list_probs])

    # 4. Evaluación individual por cada tipo de avería
    metrics_per_mode = {}
    print("=" * 65)
    print("   EVALUACIÓN DE DIAGNÓSTICO PRESCRIPTIVO MULTI-ETIQUETA")
    print("=" * 65)

    for i, col_name in enumerate(Y_train.columns):
        y_true_col = Y_test.iloc[:, i]
        y_pred_col = y_pred[:, i]
        y_prob_col = y_prob[:, i]

        pr_auc = average_precision_score(y_true_col, y_prob_col)
        report = classification_report(
            y_true_col, y_pred_col, output_dict=True, zero_division=0
        )

        metrics_per_mode[col_name] = {
            "pr_auc": pr_auc,
            "recall": report["1"]["recall"] if "1" in report else 0.0,
            "precision": report["1"]["precision"] if "1" in report else 0.0,
            "f1_score": report["1"]["f1-score"] if "1" in report else 0.0,
            "soporte_positivos": int(y_true_col.sum()),
        }

        print(f"\n--- Modo de Fallo: {col_name} ---")
        print(f"Casos reales en Test: {int(y_true_col.sum())}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(
            classification_report(y_true_col, y_pred_col, zero_division=0)
        )

    return {
        "model": diagnostic_engine,
        "y_pred": pd.DataFrame(y_pred, columns=Y_train.columns, index=X_test.index),
        "y_prob": pd.DataFrame(y_prob, columns=Y_train.columns, index=X_test.index),
        "metrics": pd.DataFrame(metrics_per_mode).T,
    }


def plot_model_benchmark(
    df_benchmark: pd.DataFrame,
) -> tuple[plt.Figure, np.ndarray]:
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
