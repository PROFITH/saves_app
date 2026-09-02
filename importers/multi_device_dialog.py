from datetime import datetime
from pathlib import Path
from typing import Optional
from PySide2.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class MultiDeviceLoadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAVES - Configuración y Carga Multi-Dispositivo")
        self.resize(740, 620)

        self.device_rows: list[dict] = []

        main_layout = QVBoxLayout(self)

        # 1. Cabecera: Selector de número de dispositivos
        header_box = QGroupBox("1. Dispositivos a Cargar")
        header_layout = QHBoxLayout(header_box)

        header_layout.addWidget(QLabel("Número de dispositivos:"))
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 10)
        self.spin_count.setValue(2)
        self.spin_count.valueChanged.connect(self._on_count_changed)
        header_layout.addWidget(self.spin_count)

        btn_add = QPushButton("+ Añadir dispositivo")
        btn_add.clicked.connect(lambda: self.spin_count.setValue(self.spin_count.value() + 1))
        header_layout.addWidget(btn_add)
        header_layout.addStretch()

        main_layout.addWidget(header_box)

        # 2. Área de lista de dispositivos con scroll
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        main_layout.addWidget(self.scroll_area)

        # 3. Sección de Porción Temporal / Ventana de Estudio
        time_box = QGroupBox("2. Porción Temporal a Plotear (Start / End Time)")
        time_layout = QVBoxLayout(time_box)

        self.check_crop = QCheckBox("Recortar porción temporal específica (D0, D8 o rango personalizado)")
        self.check_crop.setChecked(True)
        self.check_crop.toggled.connect(self._toggle_time_inputs)
        time_layout.addWidget(self.check_crop)

        time_form = QHBoxLayout()
        time_form.addWidget(QLabel("Inicio (Start):"))
        self.dt_start = QDateTimeEdit(datetime.now())
        self.dt_start.setCalendarPopup(True)
        time_form.addWidget(self.dt_start)

        time_form.addWidget(QLabel("Fin (End):"))
        self.dt_end = QDateTimeEdit(datetime.now())
        self.dt_end.setCalendarPopup(True)
        time_form.addWidget(self.dt_end)
        time_layout.addLayout(time_form)

        main_layout.addWidget(time_box)

        # Video file
        video_box = QGroupBox("3. Vídeo de Referencia (Opcional)")
        video_layout = QHBoxLayout(video_box)
        self.line_video = QLineEdit()
        self.line_video.setPlaceholderText("Ruta del archivo de vídeo (*.mp4, *.avi, *.mov)...")
        btn_video = QPushButton("Examinar...")
        btn_video.clicked.connect(self._browse_video)
        video_layout.addWidget(self.line_video)
        video_layout.addWidget(btn_video)
        main_layout.addWidget(video_box)

        # Sync file
        sync_box = QGroupBox("4. Archivo de Sincronización Previa (Opcional)")
        sync_layout = QHBoxLayout(sync_box)
        self.line_sync = QLineEdit()
        self.line_sync.setPlaceholderText("Ruta de sync.xlsx guardado previamente...")
        btn_sync = QPushButton("Examinar...")
        btn_sync.clicked.connect(self._browse_sync)
        sync_layout.addWidget(self.line_sync)
        sync_layout.addWidget(btn_sync)
        main_layout.addWidget(sync_box)

        # Accept/Cancel buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        main_layout.addWidget(btns)

        # Initialize
        self._sync_device_rows(self.spin_count.value())

    def _toggle_time_inputs(self, enabled: bool):
        self.dt_start.setEnabled(enabled)
        self.dt_end.setEnabled(enabled)

    def _on_count_changed(self, new_count: int):
        self._sync_device_rows(new_count)

    def _sync_device_rows(self, target_count: int):
        while len(self.device_rows) < target_count:
            idx = len(self.device_rows) + 1
            self._create_device_row(idx)
        while len(self.device_rows) > target_count:
            last = self.device_rows.pop()
            self.scroll_layout.removeWidget(last["widget"])
            last["widget"].deleteLater()

    def _create_device_row(self, device_num: int):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)

        row_top = QHBoxLayout()
        lbl_num = QLabel(f"<b>Dispositivo #{device_num}</b>")
        row_top.addWidget(lbl_num)

        combo_type = QComboBox()
        combo_type.addItems(["SENS Motion (*.csv)", "Parmay Matrix (*.bin)"])
        if device_num == 2:
            combo_type.setCurrentIndex(1)
        row_top.addWidget(QLabel("Tipo:"))
        row_top.addWidget(combo_type)
        row_top.addStretch()

        btn_del = QPushButton("Eliminar")
        btn_del.setStyleSheet("color: red;")
        btn_del.clicked.connect(lambda: self._remove_specific_row(card))
        row_top.addWidget(btn_del)
        layout.addLayout(row_top)

        row_bottom = QHBoxLayout()
        line_path = QLineEdit()
        line_path.setPlaceholderText("Selecciona el archivo de datos...")
        btn_browse = QPushButton("Examinar...")
        btn_browse.clicked.connect(lambda: self._browse_file(line_path, combo_type))

        row_bottom.addWidget(line_path)
        row_bottom.addWidget(btn_browse)
        layout.addLayout(row_bottom)

        self.scroll_layout.addWidget(card)
        self.device_rows.append({
            "widget": card,
            "type": combo_type,
            "path": line_path,
        })

    def _remove_specific_row(self, card_widget: QFrame):
        if len(self.device_rows) > 1:
            self.device_rows = [r for r in self.device_rows if r["widget"] != card_widget]
            self.scroll_layout.removeWidget(card_widget)
            card_widget.deleteLater()
            self.spin_count.blockSignals(True)
            self.spin_count.setValue(len(self.device_rows))
            self.spin_count.blockSignals(False)

    def _browse_file(self, line_edit: QLineEdit, combo: QComboBox):
        if combo.currentIndex() == 0:
            filter_str = "Archivos SENS (*.csv);;Todos (*.*)"
        else:
            filter_str = "Archivos Matrix (*.bin);;Todos (*.*)"
        f_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo de sensor", "", filter_str)
        if f_path:
            line_edit.setText(f_path)

    def _browse_video(self):
        v_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar vídeo", "", "Vídeos (*.mp4 *.avi *.mov);;Todos (*.*)")
        if v_path:
            self.line_video.setText(v_path)

    def get_configured_data(self) -> tuple[list[dict], Optional[tuple[float, float]], str]:
        devices = []
        for i, row in enumerate(self.device_rows, start=1):
            p = row["path"].text().strip()
            if p and Path(p).exists():
                stype = "SENS" if row["type"].currentIndex() == 0 else "Matrix"
                devices.append({
                    "id": i,
                    "type": stype,
                    "path": p,
                })

        time_window = None
        if self.check_crop.isChecked():
            t_start_ms = self.dt_start.dateTime().toSecsSinceEpoch() * 1000.0
            t_end_ms = self.dt_end.dateTime().toSecsSinceEpoch() * 1000.0
            time_window = (t_start_ms, t_end_ms)

        return devices, time_window, self.line_video.text().strip()

    def _browse_sync(self):
        s_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de sincronización", "", "Excel (*.xlsx);;Todos (*.*)"
        )
        if s_path:
            self.line_sync.setText(s_path)

    def get_configured_data(self) -> tuple[list[dict], Optional[tuple[float, float]], str, str]:
        devices = []
        for i, row in enumerate(self.device_rows, start=1):
            p = row["path"].text().strip()
            if p and Path(p).exists():
                stype = "SENS" if row["type"].currentIndex() == 0 else "Matrix"
                devices.append({"id": i, "type": stype, "path": p})

        time_window = None
        if self.check_crop.isChecked():
            t_start_ms = self.dt_start.dateTime().toSecsSinceEpoch() * 1000.0
            t_end_ms = self.dt_end.dateTime().toSecsSinceEpoch() * 1000.0
            time_window = (t_start_ms, t_end_ms)

        return (
            devices,
            time_window,
            self.line_video.text().strip(),
            self.line_sync.text().strip(),
        )