import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]
TARGET_COLUMN = "Machine failure"


class IndustrialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Genera variables físicas a partir de la telemetría de sensores
    y elimina columnas de metadatos o generadoras de leakage.
    """

    def __init__(self, drop_leakage: bool = True):
        self.drop_leakage = drop_leakage

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Eliminar identificadores si existen
        cols_to_drop = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
        if self.drop_leakage:
            cols_to_drop.extend([c for c in LEAKAGE_COLUMNS if c in df.columns])

        df = df.drop(columns=cols_to_drop, errors="ignore")

        # 2. Ingeniería de variables de dominio físico
        # Delta térmico (Kelvin)
        df["temp_diff_k"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
        )

        # Potencia mecánica (Vatios: W = Torque [Nm] * omega [rad/s])
        omega_rad_s = df["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
        df["mechanical_power_w"] = df["Torque [Nm]"] * omega_rad_s

        # Esfuerzo acumulado (Fatiga: Torque * Minutos de desgaste)
        df["strain_factor"] = df["Torque [Nm]"] * df["Tool wear [min]"]

        # Intensidad térmica por ciclo rotacional
        df["thermal_intensity"] = (
            df["Process temperature [K]"] / df["Rotational speed [rpm]"]
        )

        # 3. Encoding de la variable 'Type'
        if "Type" in df.columns:
            type_mapping = {"L": 0, "M": 1, "H": 2}
            df["type_tier"] = df["Type"].map(type_mapping).astype(int)
            df = df.drop(columns=["Type"])

        return df