from typing import Dict  # noqa: UP035

import numpy as np
import pandas as pd
from mad_gui.plugins import BaseAlgorithm
from PySide2.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)


class CalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibración Fina y Corrección de Deriva (Time Drift)")
        self.resize(450, 220)

        layout = QVBoxLayout(self)

        info_label = QLabel(
            "<b>Ajuste de Sincronización entre Sensores:</b><br>"
            "• <b>Offset (ms):</b> Desplaza la señal de Matrix adelante (+) o atrás (-).<br>"
            "• <b>Deriva temporal (Drift %):</b> Corrige la desincronización acumulada a lo largo del tiempo.<br>"
            "<i>(Ej. +0.05% si Matrix se retrasa progresivamente respecto a SENS).</i>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Campo: Offset manual en milisegundos
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Offset manual (ms):"))
        self.spin_offset = QDoubleSpinBox()
        self.spin_offset.setRange(-3600000.0, 3600000.0)
        self.spin_offset.setValue(0.0)
        self.spin_offset.setSingleStep(10.0)
        self.spin_offset.setSuffix(" ms")
        offset_layout.addWidget(self.spin_offset)
        layout.addLayout(offset_layout)

        # Campo: Corrección de Drift
        drift_layout = QHBoxLayout()
        drift_layout.addWidget(QLabel("Corrección de Deriva (%):"))
        self.spin_drift = QDoubleSpinBox()
        self.spin_drift.setRange(-5.0, 5.0)
        self.spin_drift.setValue(0.0)
        self.spin_drift.setSingleStep(0.01)
        self.spin_drift.setDecimals(4)
        self.spin_drift.setSuffix(" %")
        drift_layout.addWidget(self.spin_drift)
        layout.addLayout(drift_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_values(self) -> tuple[float, float]:
        return self.spin_offset.value(), self.spin_drift.value()


class SensorCalibrationAlgorithm(BaseAlgorithm):
    @classmethod
    def name(cls) -> str:
        return "Calibrar Offset y Deriva (Matrix vs SENS)"

    @classmethod
    def description(cls) -> str:
        return "Permite desplazar y reescalar la señal de Parmay Matrix respecto a SENS."

    def process_data(self, data: Dict) -> Dict:  # noqa: UP006
        matrix_key = "Parmay Matrix (acceleration)"

        if matrix_key not in data:
            return data

        dialog = CalibrationDialog()
        if dialog.exec_() != QDialog.Accepted:
            return data

        offset_ms, drift_percent = dialog.get_values()

        # Acceso compatible con PlotData de mad-gui o diccionario estándar
        plot_data_matrix = data[matrix_key]
        if hasattr(plot_data_matrix, "data"):
            df_matrix = plot_data_matrix.data.copy()
            fs_matrix = plot_data_matrix.sampling_rate_hz
        else:
            df_matrix = plot_data_matrix["sensor_data"].copy()
            fs_matrix = plot_data_matrix["sampling_rate_hz"]

        n_samples = len(df_matrix)
        if n_samples == 0:
            return data

        t_original = np.arange(n_samples) / fs_matrix

        # 1. Aplicar factor de escala temporal (drift)
        drift_factor = 1.0 + (drift_percent / 100.0)
        t_corrected = t_original * drift_factor

        # 2. Aplicar desplazamiento temporal (offset)
        t_shifted = t_corrected + (offset_ms / 1000.0)

        # 3. Interpolar canales de aceleración
        df_new = pd.DataFrame(index=df_matrix.index)
        for col in df_matrix.columns:
            df_new[col] = np.interp(t_original, t_shifted, df_matrix[col])

        # Actualizar PlotData en memoria
        if hasattr(plot_data_matrix, "data"):
            plot_data_matrix.data = df_new
        else:
            plot_data_matrix["sensor_data"] = df_new

        return data

