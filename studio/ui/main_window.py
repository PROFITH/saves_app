"""
Main window of SAVES Studio.

The work follows the order of the task: open the session folder (or drop
files on the window) with the technician's code, let the claps and jumps be
detected in every sensor (each clock's offset is found first), let the audio
of the video place the video, fix the events flagged in red, count the
steps, sit-to-stands or any other event on the video, mark each task
finished, and export. A banner always says what the next step is, and each
technician's work is saved apart in the session folder as it is done.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from importers.manifest_importer import ManifestImporter
from studio import redcap
from studio.discovery import (
    FORMAT_LABELS,
    VIDEO_SUFFIXES,
    classify_file,
    discover,
    sensor_entry_for,
)
from studio.engine import FAIR_RMSE_MS, GOOD_RMSE_MS, VIDEO, SyncEngine, analyze_signal
from studio.export import label_times, read_phases
from studio.project import (
    FINISHED,
    LOCATIONS,
    NOT_STARTED,
    STEPS,
    SYNC_TASK,
    TASK_STATE_TEXT,
    TASK_STATES,
    EventType,
    SyncProject,
    VideoEntry,
    normalize_technician,
    project_file,
)
from studio.signals import load_signal
from studio.ui.plots import SignalStack, event_color, sensor_color
from studio.ui.redcap_dialog import RedcapSettingsDialog, busy, saved_token, saved_url
from studio.ui.technician import EventTypeDialog, TechnicianDialog
from studio.ui.video_panel import VideoPanel
from studio.ui.workers import start_job
from studio.video import find_audio_events, has_audio

QUALITY_COLORS = {
    "reference": "#1565c0",
    "good": "#2e7d32",
    "fair": "#f9a825",
    "poor": "#c62828",
    "none": "#9e9e9e",
}
QUALITY_TEXT = {
    "reference": "Referencia",
    "good": "Bien",
    "fair": "Revisar",
    "poor": "Mal",
    "none": "Sin ajuste",
}
TASK_STATE_COLORS = {"not_started": "#9e9e9e", "started": "#ef8f00", "finished": "#2e7d32"}
VIDEO_SOURCE_TEXT = {
    "claps": "ajustada con las palmadas del vídeo",
    "one_clap": "ajustada con 1 palmada (marca otra para corregir la deriva)",
    "recording_start": "según la hora de inicio registrada (aproximada)",
    "session_start": "aproximada: no hay hora de inicio del vídeo",
    "none": "",
}


def clock(t: Optional[float]) -> str:  # noqa: UP045
    if t is None:
        return "—"
    return datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]


def residual_color(value: Optional[float]) -> QColor:  # noqa: UP045
    if value is None:
        return QColor("#9e9e9e")
    a = abs(value)
    if a <= GOOD_RMSE_MS:
        return QColor("#2e7d32")
    if a <= FAIR_RMSE_MS:
        return QColor("#ef8f00")
    return QColor("#c62828")


def _is_na(value) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and bool(np.isnan(value)))


def _short_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m %H:%M")
    except (TypeError, ValueError):
        return ""


def _location_group(location: str) -> str:
    """First word of a location («Calzado, izquierda» → «Calzado»)."""
    words = location.replace(",", " ").replace("(", " ").split()
    return words[0] if words else location


def readonly_item(text: str, color: Optional[QColor] = None, bold: bool = False) -> QTableWidgetItem:  # noqa: UP045
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    if color is not None:
        item.setForeground(color)
    if bold:
        font = item.font()
        font.setBold(True)
        item.setFont(font)
    return item


class MainWindow(QMainWindow):
    def __init__(self, settings: Optional[QSettings] = None):  # noqa: UP045
        super().__init__()
        self.setWindowTitle("SAVES Studio")
        self.resize(1600, 950)
        self.setAcceptDrops(True)
        # Remembered on this computer: the last technician's code and the
        # kinds of events created here.
        self.settings = settings if settings is not None else QSettings("PROFITH", "SAVES Studio")

        self.project: Optional[SyncProject] = None  # noqa: UP045
        self.engine: Optional[SyncEngine] = None  # noqa: UP045
        self.loading: dict[str, str] = {}   # sensor_id -> progress text
        self.errors: dict[str, str] = {}
        self._jobs = set()
        self._plotted: list[str] = []
        self._selected_event: Optional[str] = None  # noqa: UP045
        self._session_title = ""
        self._phases = None
        # Events counted on the video: the kind being counted, the undo
        # history of each kind ("add" | "remove", event) and the counts of
        # the kind on screen per activity (main phase, all phases).
        self._event_key = STEPS
        self._event_history: dict[str, list[tuple[str, dict]]] = {}
        self._event_counts: dict[tuple[int, int], tuple[int, int]] = {}
        self._task_combos: dict[str, QComboBox] = {}

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._save)

        self._build_actions()
        self._build_ui()
        self._build_shortcuts()
        self.refresh_all()

    # ==================================================================
    # Construction
    # ==================================================================

    def _build_actions(self) -> None:
        bar = QToolBar("Principal")
        bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(bar)

        self.act_open = QAction("📂 Abrir sesión…", self, shortcut=QKeySequence.Open, triggered=self.choose_folder)
        self.act_add = QAction("➕ Añadir archivos…", self, shortcut="Ctrl+Shift+O", triggered=self.choose_files)
        self.act_technician = QAction("👤 Técnico", self, triggered=self.change_technician)
        self.act_technician.setToolTip("Cambiar de técnico: cada uno guarda su trabajo aparte en la carpeta de la sesión")
        self.act_audio = QAction("🔊 Detectar palmadas y saltos en el audio", self, triggered=self.detect_audio)
        self.act_save = QAction("💾 Guardar", self, shortcut=QKeySequence.Save, triggered=self.save_now)
        self.act_save.setToolTip(
            "Tu trabajo se guarda solo después de cada cambio, en saves_project_<código>.json dentro de la "
            "carpeta de la sesión; este botón lo guarda ya (Ctrl+S). Al volver a abrir la carpeta con tu "
            "código, sigues donde lo dejaste."
        )
        self.act_export = QAction("📤 Exportar", self, shortcut="Ctrl+E", triggered=self.export)
        self.act_redcap = QAction("☁ Enviar a REDCap", self, triggered=self.send_to_redcap)
        self.act_redcap.setToolTip(
            "Envía a lab_activity en REDCap los pasos y las veces que se levanta en cada actividad, "
            "de los conteos finalizados y con tu código de técnico"
        )
        self.act_redcap_settings = QAction("🔑 Conexión con REDCap…", self, triggered=self.redcap_settings)
        self.act_redcap_settings.setToolTip(
            "La dirección de la API y tu token de REDCap (opcional: sin token, los conteos solo se guardan "
            "en los archivos exportados)"
        )
        self.act_xyz = QAction("Ejes X/Y/Z", self, checkable=True, triggered=self._toggle_xyz)
        self.act_xyz.setToolTip("Ver los tres ejes en lugar del módulo (vector magnitude)")
        for action in (self.act_open, self.act_add):
            bar.addAction(action)
        bar.addSeparator()
        bar.addAction(self.act_technician)
        bar.addSeparator()
        bar.addWidget(QLabel(" Referencia: "))
        self.reference_box = QComboBox()
        self.reference_box.setToolTip(
            "Reloj al que se ajustan los demás. Automática: el Polar que más eventos captó (ya está en la hora "
            "del ordenador, y el otro Polar completa lo que le falte); si no hay, el vídeo con 2 o más eventos "
            "marcados; si tampoco, el sensor con más eventos, llevado a la hora de los avisos del cronómetro."
        )
        self.reference_box.currentIndexChanged.connect(self._reference_changed)
        bar.addWidget(self.reference_box)
        bar.addSeparator()
        bar.addAction(self.act_audio)
        bar.addAction(self.act_xyz)
        self.plots_button = QToolButton()
        self.plots_button.setText("👁 Gráficos")
        self.plots_button.setToolTip("Elegir qué gráficos se ven (los ocultos siguen en la sincronización)")
        self.plots_button.setPopupMode(QToolButton.InstantPopup)
        self.plots_button.setFocusPolicy(Qt.NoFocus)
        self.plots_menu = QMenu(self.plots_button)
        self.plots_menu.aboutToShow.connect(self._fill_plots_menu)
        self.plots_button.setMenu(self.plots_menu)
        bar.addWidget(self.plots_button)
        bar.addSeparator()
        bar.addAction(self.act_save)
        self.saved_label = QLabel()
        self.saved_label.setStyleSheet("color:#555; padding:0 6px;")
        bar.addWidget(self.saved_label)
        bar.addSeparator()
        bar.addAction(self.act_export)
        bar.addAction(self.act_redcap)

        help_menu = self.menuBar().addMenu("Ayuda")
        help_menu.addAction("Abrir una sesión de demostración", self.open_demo)
        help_menu.addAction("Atajos de teclado", self.show_shortcuts)
        file_menu = self.menuBar().addMenu("Archivo")
        file_menu.addAction(self.act_open)
        file_menu.addAction(self.act_add)
        file_menu.addAction(self.act_technician)
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_export)
        file_menu.addAction(self.act_redcap)
        file_menu.addAction(self.act_redcap_settings)
        self.menuBar().insertMenu(help_menu.menuAction(), file_menu)

    def _build_ui(self) -> None:
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_welcome())
        self.stack.addWidget(self._build_workspace())

    def _build_welcome(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch()
        box = QFrame()
        box.setStyleSheet("QFrame{border:2px dashed #90a4ae; border-radius:12px; background:#fafafa;}")
        inner = QVBoxLayout(box)
        inner.setContentsMargins(40, 40, 40, 40)
        title = QLabel("<h2>Sincroniza vídeo y sensores offline</h2>")
        title.setAlignment(Qt.AlignCenter)
        text = QLabel(
            "Arrastra aquí la <b>carpeta de la sesión</b> que guardó lab_protocol_timer "
            "(manifiesto, vídeo y CSV del Polar) o ábrela con el botón.<br>"
            "Después arrastra los archivos de los sensores: Axivity <b>.cwa</b>, "
            "ActiGraph <b>.gt3x</b>, GENEActiv o Parmay Matrix <b>.bin</b> y SENS <b>.csv</b>."
        )
        text.setWordWrap(True)
        text.setAlignment(Qt.AlignCenter)
        buttons = QHBoxLayout()
        open_button = QPushButton("📂 Abrir carpeta de sesión")
        open_button.setStyleSheet("padding:10px 18px; font-size:14px;")
        open_button.clicked.connect(self.choose_folder)
        demo_button = QPushButton("Probar con una sesión de demostración")
        demo_button.setStyleSheet("padding:10px 18px;")
        demo_button.clicked.connect(self.open_demo)
        buttons.addStretch()
        buttons.addWidget(open_button)
        buttons.addWidget(demo_button)
        buttons.addStretch()
        for widget in (title, text):
            widget.setStyleSheet("border:none;")
            inner.addWidget(widget)
        inner.addLayout(buttons)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(box, 3)
        row.addStretch()
        layout.addLayout(row)
        layout.addStretch()
        return page

    def _build_workspace(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(6, 6, 6, 6)

        # Banner with the next step.
        self.banner = QFrame()
        self.banner.setStyleSheet("QFrame{background:#e3f2fd; border-radius:6px;}")
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(10, 6, 10, 6)
        self.banner_label = QLabel()
        self.banner_label.setWordWrap(True)
        self.banner_button = QPushButton()
        self.banner_button.setFocusPolicy(Qt.NoFocus)
        self.banner_button.clicked.connect(self._banner_clicked)
        banner_layout.addWidget(self.banner_label, 1)
        banner_layout.addWidget(self.banner_button)
        outer.addWidget(self.banner)
        self._banner_action = None

        main = QSplitter(Qt.Horizontal)
        outer.addWidget(main, 1)

        # Left: the technician's tasks, files and quality.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.tasks_box = QGroupBox("Tareas de este vídeo")
        tasks_layout = QVBoxLayout(self.tasks_box)
        self.tasks_table = QTableWidget(0, 3)
        self.tasks_table.setHorizontalHeaderLabels(["Tarea", "Estado", "Quién y cuándo"])
        self.tasks_table.verticalHeader().hide()
        self.tasks_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tasks_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tasks_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.tasks_table.setFocusPolicy(Qt.NoFocus)
        self.tasks_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tasks_layout.addWidget(self.tasks_table)
        tasks_hint = QLabel("Pasan a «Iniciada» solas al trabajar en ellas; «Finalizada» la marcas tú al terminar.")
        tasks_hint.setWordWrap(True)
        tasks_hint.setStyleSheet("color:#555;")
        tasks_layout.addWidget(tasks_hint)
        left_layout.addWidget(self.tasks_box, 0)
        files_box = QGroupBox("Archivos de la sesión")
        files_layout = QVBoxLayout(files_box)
        self.session_label = QLabel()
        self.session_label.setWordWrap(True)
        files_layout.addWidget(self.session_label)
        self.sensors_table = QTableWidget(0, 5)
        self.sensors_table.setHorizontalHeaderLabels(["", "Sensor", "Ubicación", "Hz", "Estado"])
        self.sensors_table.verticalHeader().hide()
        self.sensors_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.sensors_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.sensors_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sensors_table.setFocusPolicy(Qt.NoFocus)
        self.sensors_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sensors_table.customContextMenuRequested.connect(self._sensor_menu)
        files_layout.addWidget(self.sensors_table)
        add_button = QPushButton("➕ Añadir sensores o vídeo…")
        add_button.setFocusPolicy(Qt.NoFocus)
        add_button.clicked.connect(self.choose_files)
        files_layout.addWidget(add_button)
        left_layout.addWidget(files_box, 3)

        quality_box = QGroupBox("Calidad de la sincronización")
        quality_layout = QVBoxLayout(quality_box)
        self.quality_table = QTableWidget(0, 6)
        self.quality_table.setHorizontalHeaderLabels(["Sensor", "Estado", "Eventos", "Desfase", "Deriva", "Error"])
        self.quality_table.verticalHeader().hide()
        self.quality_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.quality_table.setFocusPolicy(Qt.NoFocus)
        self.quality_table.setToolTip(
            "Eventos: palmadas o saltos encontrados / esperados.\n"
            "Desfase: diferencia de reloj con la referencia en el primer evento.\n"
            "Deriva: lo que se adelanta o atrasa el reloj (ppm = µs por segundo).\n"
            "Error: RMSE de los eventos tras el ajuste (verde ≤ 25 ms, ámbar ≤ 80 ms)."
        )
        self.quality_table.cellDoubleClicked.connect(self._quality_double_clicked)
        quality_layout.addWidget(self.quality_table)
        self.video_quality = QLabel()
        self.video_quality.setWordWrap(True)
        quality_layout.addWidget(self.video_quality)
        left_layout.addWidget(quality_box, 2)
        main.addWidget(left)

        # Centre: plots over the claps table.
        centre = QSplitter(Qt.Vertical)
        self.plots = SignalStack()
        self.plots.clicked.connect(self._plot_clicked)
        self.plots.anchor_moved.connect(self._anchor_moved)
        self.plots.anchor_action.connect(self._anchor_action)
        self.plots.add_anchor.connect(self._anchor_moved)
        self.plots.plot_visibility.connect(self._plot_visibility)
        centre.addWidget(self.plots)
        self.tabs = QTabWidget()
        claps_box = QWidget()
        claps_layout = QVBoxLayout(claps_box)
        claps_layout.setContentsMargins(4, 4, 4, 4)
        claps_hint = QLabel("Doble clic: ir al evento · clic derecho: opciones")
        claps_hint.setStyleSheet("color:#555;")
        claps_layout.addWidget(claps_hint)
        self.claps_table = QTableWidget(0, 0)
        self.claps_table.verticalHeader().hide()
        self.claps_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.claps_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.claps_table.setFocusPolicy(Qt.NoFocus)
        self.claps_table.itemSelectionChanged.connect(self._clap_selected)
        self.claps_table.cellDoubleClicked.connect(lambda r, _c: self.focus_event(self._event_at(r)))
        self.claps_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.claps_table.customContextMenuRequested.connect(self._clap_menu)
        claps_layout.addWidget(self.claps_table)
        self.tabs.addTab(claps_box, "Palmadas y saltos")
        self.tabs.addTab(self._build_events_tab(), "🔢 Conteo")
        centre.addWidget(self.tabs)
        centre.setStretchFactor(0, 3)
        centre.setStretchFactor(1, 1)
        main.addWidget(centre)

        # Right: video.
        self.video = VideoPanel()
        self.video.frame_changed.connect(self._video_frame)
        self.video.mark_requested.connect(self.mark_video_clap)
        self.video.event_requested.connect(self.mark_event)
        main.addWidget(self.video)

        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 3)
        main.setStretchFactor(2, 1)
        main.setSizes([420, 720, 380])
        return page

    def _build_events_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        row.addWidget(QLabel("Qué cuentas:"))
        self.event_kind_box = QComboBox()
        self.event_kind_box.setFocusPolicy(Qt.NoFocus)  # S, A and D must reach the shortcuts
        self.event_kind_box.setToolTip("Lo que marcan S (y A / D) en el vídeo. Cada tipo es una tarea de la lista.")
        self.event_kind_box.currentIndexChanged.connect(self._event_kind_changed)
        row.addWidget(self.event_kind_box, 1)
        self.finish_kind_button = QPushButton()
        self.finish_kind_button.setFocusPolicy(Qt.NoFocus)
        self.finish_kind_button.clicked.connect(lambda: self.set_task_state(self._event_key, FINISHED))
        row.addWidget(self.finish_kind_button)
        new_kind = QPushButton("➕ Otro evento…")
        new_kind.setFocusPolicy(Qt.NoFocus)
        new_kind.setToolTip("Contar algo distinto en el vídeo (se añade a esta sesión con su propia tarea)")
        new_kind.clicked.connect(self.add_event_kind)
        row.addWidget(new_kind)
        self.remove_kind_button = QPushButton("Quitar")
        self.remove_kind_button.setFocusPolicy(Qt.NoFocus)
        self.remove_kind_button.setToolTip("Quitar este tipo de la sesión si no se cuenta en ella (solo sin eventos marcados)")
        self.remove_kind_button.clicked.connect(self.remove_event_kind)
        row.addWidget(self.remove_kind_button)
        layout.addLayout(row)
        self.events_hint = QLabel()
        self.events_hint.setWordWrap(True)
        self.events_hint.setStyleSheet("color:#555;")
        layout.addWidget(self.events_hint)
        self.events_table = QTableWidget(0, 5)
        self.events_table.verticalHeader().hide()
        self.events_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.events_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.events_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.events_table.setFocusPolicy(Qt.NoFocus)
        self.events_table.cellDoubleClicked.connect(self._events_row_clicked)
        layout.addWidget(self.events_table)
        return page

    def _build_shortcuts(self) -> None:
        bindings = {
            "Space": self.video.toggle_play,
            "Right": self.video.next_frame,
            "Left": self.video.previous_frame,
            "Shift+Right": lambda: self.video.step_seconds(1.0),
            "Shift+Left": lambda: self.video.step_seconds(-1.0),
            "Ctrl+Right": lambda: self.video.step_seconds(10.0),
            "Ctrl+Left": lambda: self.video.step_seconds(-10.0),
            "M": self.mark_video_clap,
            "N": lambda: self.select_relative(1),
            "P": lambda: self.select_relative(-1),
            "Delete": self.clear_video_clap,
            "X": lambda: self.act_xyz.trigger(),
            "S": lambda: self.mark_event(""),
            "A": lambda: self.mark_event("L"),
            "D": lambda: self.mark_event("R"),
            "Backspace": self.remove_event,
            "Ctrl+Z": self.undo_event,
        }
        for key, slot in bindings.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(slot)

    def show_shortcuts(self) -> None:
        QMessageBox.information(self, "Atajos de teclado", (
            "Espacio: reproducir / pausa\n"
            "← / →: fotograma anterior / siguiente\n"
            "Mayús + ← / →: 1 s atrás / adelante\n"
            "Ctrl + ← / →: 10 s atrás / adelante\n"
            "N / P: evento (palmada o salto) siguiente / anterior\n"
            "M: marcar el evento en el vídeo (fotograma actual)\n"
            "Supr: borrar la marca del vídeo del evento elegido\n"
            "S: marcar lo que cuentas (pestaña «Conteo») en el fotograma actual;\n"
            "    A / D: lado izquierdo / derecho, si ese tipo los distingue\n"
            "Retroceso: borrar la marca más cercana · Ctrl+Z: deshacer el último cambio de ese tipo\n"
            "Ctrl+S: guardar ya (también se guarda solo) · Ctrl+E: exportar\n"
            "X: ver ejes X/Y/Z o el módulo\n"
            "Gráficas: rueda para zoom, arrastrar para desplazarse, clic para llevar el vídeo a ese instante,\n"
            "arrastrar una línea de evento para corregirla, clic derecho para más opciones."
        ))

    # ==================================================================
    # Opening sessions and files
    # ==================================================================

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Carpeta de la sesión")
        if folder:
            self.open_folder(folder)

    def choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Archivos de sensores o vídeo", self.project.folder if self.project else "",
            "Sesión (*.cwa *.gt3x *.bin *.csv *.json *.mp4 *.mov *.avi *.mkv);;Todos (*.*)",
        )
        if files:
            self.add_paths([Path(f) for f in files])

    def open_demo(self) -> None:
        from studio.demo import build_demo_session

        folder = Path(tempfile.gettempdir()) / "saves_studio_demo"
        if not (folder / "session_manifest.json").exists():
            self.statusBar().showMessage("Creando la sesión de demostración…")
            build_demo_session(folder)
        self.open_folder(folder)

    def open_folder(self, folder: str | Path, technician: Optional[str] = None,  # noqa: UP045
                    copy_from: Optional[str] = None) -> None:  # noqa: UP045
        """
        Open a session folder with the work of a technician.

        Without ``technician`` the technician is asked for their code; the
        work saved under that code is restored, or a new one starts (from
        the synchronization of technician ``copy_from`` if they choose so).
        """
        folder = Path(folder)
        if technician is None:
            dialog = TechnicianDialog(folder, str(self.settings.value("technician", "") or ""), self)
            if dialog.exec() != QDialog.Accepted:
                return
            code, copy_from = dialog.choice()
        else:
            code = normalize_technician(technician)
        # The work open now is saved before it is closed.
        self._save_timer.stop()
        self._save()
        self._close_session()
        found = discover(folder)
        own = SyncProject.exists_in(folder, code)
        project = self._saved_project(folder, code, copy_from)
        # New for this technician (not their own saved work).
        new = not (own and project is not None)
        if project is not None:
            known = {str(Path(s.path).resolve()) for s in project.sensors}
            for entry in found.sensors:
                if str(Path(entry.path).resolve()) not in known:
                    project.add_sensor(entry)
            if not project.videos:
                project.videos = found.videos
            project.manifest_path = project.manifest_path or found.manifest_path
        else:
            project = SyncProject(folder=str(folder), manifest_path=found.manifest_path, videos=found.videos,
                                  technician=code)
            for entry in found.sensors:
                project.add_sensor(entry)
        if new:
            for kind in self._remembered_kinds():
                if kind.get("new_sessions"):
                    project.add_event_type(str(kind["name"]), bool(kind.get("sides")))
        if code:
            self.settings.setValue("technician", code)
        self.project = project
        self._event_history = {}
        if self._event_key not in {k.key for k in project.event_types}:
            self._event_key = project.event_types[0].key if project.event_types else STEPS
        self.stack.setCurrentIndex(1)
        self._load_manifest()
        self._open_video()
        for entry in project.sensors:
            self._load_sensor(entry.sensor_id)
        self._schedule_save()
        self.refresh_all()
        if not project.sensors:
            self.statusBar().showMessage(
                "No hay archivos de sensores en la carpeta: cópialos en ella o arrástralos a la ventana."
            )
        # First try on its own: the claps and landings in the video's sound.
        if self.engine is not None and self.project.videos and not self.project.video_events:
            QTimer.singleShot(0, lambda: self.detect_audio(auto=True))

    def _saved_project(self, folder: Path, code: str, copy_from: Optional[str]) -> Optional[SyncProject]:  # noqa: UP045
        """The work saved for ``code`` in ``folder``, or None to start a new one."""
        path = folder / project_file(code)
        if path.exists():
            try:
                return SyncProject.load(folder, code)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # Never overwrite it: set it aside and start again.
                backup = path.with_name(f"{path.name}.ilegible-{datetime.now():%Y%m%d-%H%M%S}")
                try:
                    path.replace(backup)
                    where = f"Se ha apartado como {backup.name}"
                except OSError:
                    where = "No se ha podido apartar"
                QMessageBox.warning(self, "Trabajo guardado ilegible", (
                    f"No se pudo leer {path.name}:\n{exc}\n\n{where} y la sesión se abre de cero."
                ))
                return None
        if not code:
            return None
        if copy_from is not None:
            try:
                source = SyncProject.load(folder, copy_from)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                QMessageBox.warning(self, "Sincronización", f"No se pudo leer el trabajo de {copy_from}:\n{exc}")
                return None
            project = SyncProject(folder=str(folder), technician=code)
            project.copy_sync_from(source)
            return project
        if not SyncProject.technicians_in(folder) and SyncProject.exists_in(folder):
            # Work saved before technicians had codes: it becomes this technician's.
            try:
                project = SyncProject.load(folder)
            except (OSError, ValueError, KeyError, TypeError):
                return None
            project.technician = code
            if project.has_sync_work():
                project.work_on(SYNC_TASK)
            for kind in project.event_types:
                if project.events.get(kind.key):
                    project.work_on(kind.key)
            return project
        return None

    def change_technician(self) -> None:
        """Reopen the session with another technician's code."""
        if self.project is not None:
            self.open_folder(self.project.folder)

    def add_paths(self, paths: list[Path]) -> None:
        if self.project is None:
            folder = next((p for p in paths if p.is_dir()), None)
            if folder is not None:
                self.open_folder(folder)
                paths = [p for p in paths if p != folder]
            else:
                manifest = next((p for p in paths if p.name == "session_manifest.json"), None)
                self.open_folder((manifest or paths[0]).parent)
            if self.project is None:
                return  # The technician did not give a code.
        expanded = []
        for p in paths:
            expanded.extend(sorted(x for x in p.iterdir() if x.is_file()) if p.is_dir() else [p])
        added_video = False
        before = (len(self.project.sensors), len(self.project.videos))
        for p in expanded:
            if p.name == "session_manifest.json":
                self.project.manifest_path = str(p)
                self._load_manifest()
            elif p.suffix.lower() in VIDEO_SUFFIXES:
                if all(Path(v.path).resolve() != p.resolve() for v in self.project.videos):
                    self.project.videos.append(VideoEntry(path=str(p)))
                    added_video = True
            else:
                fmt = classify_file(p)
                if fmt is None:
                    continue
                entry = self.project.add_sensor(sensor_entry_for(p, fmt))
                if entry.sensor_id not in self.loading and (self.engine is None or entry.sensor_id not in self.engine.signals):
                    self._load_sensor(entry.sensor_id)
        if added_video and self.video.reader is None:
            self._open_video()
        if (len(self.project.sensors), len(self.project.videos)) != before:
            self.project.work_on(SYNC_TASK)
        self._schedule_save()
        self.refresh_all()

    def _close_session(self) -> None:
        self.video.close_video()
        self.project, self.engine = None, None
        self._phases = None
        self.loading, self.errors = {}, {}
        self._plotted = []
        self._session_title = ""
        self.saved_label.setText("")
        self.plots.set_engine(None)

    def _load_manifest(self) -> None:
        if not self.project or not self.project.manifest_path:
            self.engine = None
            return
        try:
            session = ManifestImporter().load(self.project.manifest_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Manifiesto no válido", f"No se pudo leer el manifiesto:\n{exc}")
            self.engine = None
            return
        old = self.engine
        self.engine = SyncEngine(self.project, session)
        self._phases = None
        if old is not None:
            for sid, signal in old.signals.items():
                self._analyze_in_background(sid, signal)
        participants = ", ".join(p.participant_id for p in session.participants)
        self._session_title = session.session_id + (f" ({participants})" if participants else "")

    def _open_video(self) -> None:
        if not self.project or not self.project.videos:
            return
        try:
            self.video.open(self.project.videos[0].path)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"No se pudo abrir el vídeo: {exc}", 8000)
            return
        if self.engine is not None and self.video.reader is not None:
            self.engine.video_duration_s = self.video.reader.duration_s

    def _load_sensor(self, sensor_id: str) -> None:
        entry = self.project.sensor(sensor_id)
        project_folder = self.project.folder
        timezone = self.project.timezone
        self.loading[sensor_id] = "En cola"
        self.errors.pop(sensor_id, None)

        def work(progress):
            signal = load_signal(entry.path, entry.sensor_format, project_folder, timezone, progress)
            engine = self.engine
            if engine is None:
                return signal, {}, {}, None
            progress("Buscando palmadas y saltos")
            anchors, coverage, search = analyze_signal(engine.session, signal, engine.pad_seconds, engine.min_prominence)
            return signal, anchors, coverage, search

        self._run(work, lambda result: self._sensor_loaded(sensor_id, result),
                  lambda error: self._sensor_failed(sensor_id, error),
                  lambda text: self._sensor_progress(sensor_id, text))

    def _analyze_in_background(self, sensor_id: str, signal) -> None:
        engine = self.engine
        self.loading[sensor_id] = "Buscando palmadas"

        def work(_progress):
            anchors, coverage, search = analyze_signal(engine.session, signal, engine.pad_seconds, engine.min_prominence)
            return signal, anchors, coverage, search

        self._run(work, lambda result: self._sensor_loaded(sensor_id, result),
                  lambda error: self._sensor_failed(sensor_id, error))

    def _run(self, work, on_done, on_error, on_progress=None) -> None:
        holder = {}

        def done(result):
            self._jobs.discard(holder.get("job"))
            on_done(result)

        def failed(error):
            self._jobs.discard(holder.get("job"))
            on_error(error)

        holder["job"] = start_job(work, done, failed, on_progress)
        self._jobs.add(holder["job"])

    def _sensor_progress(self, sensor_id: str, text: str) -> None:
        self.loading[sensor_id] = text
        self.refresh_sensors_table()

    def _sensor_loaded(self, sensor_id: str, result) -> None:
        self.loading.pop(sensor_id, None)
        if self.project is None or self.engine is None:
            return
        try:
            self.project.sensor(sensor_id)
        except KeyError:
            return  # Removed while loading.
        signal, anchors, coverage, search = result
        self.engine.ingest(sensor_id, signal, anchors, coverage, search)
        self.engine.refit()
        self.refresh_all()

    def _sensor_failed(self, sensor_id: str, error: str) -> None:
        self.loading.pop(sensor_id, None)
        self.errors[sensor_id] = error
        self.refresh_all()

    # ==================================================================
    # Drag and drop
    # ==================================================================

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        if self.project is None and len(paths) == 1 and paths[0].is_dir():
            self.open_folder(paths[0])
        else:
            self.add_paths(paths)

    # ==================================================================
    # Refresh
    # ==================================================================

    def refresh_all(self) -> None:
        if self.project is None:
            self.stack.setCurrentIndex(0)
            for action in (self.act_add, self.act_audio, self.act_export, self.act_xyz, self.act_save,
                           self.act_technician):
                action.setEnabled(False)
            self.act_technician.setText("👤 Técnico")
            self.plots_button.setEnabled(False)
            self.reference_box.setEnabled(False)
            self.setWindowTitle("SAVES Studio")
            return
        self.stack.setCurrentIndex(1)
        self.act_add.setEnabled(True)
        self.act_save.setEnabled(True)
        self.act_technician.setEnabled(True)
        self.plots_button.setEnabled(self.engine is not None)
        has_engine = self.engine is not None
        self.act_audio.setEnabled(has_engine and self.video.reader is not None)
        self.act_export.setEnabled(has_engine and bool(self.engine.signals) if has_engine else False)
        self.act_xyz.setEnabled(True)
        self.reference_box.setEnabled(True)
        self.refresh_title()
        self.refresh_reference_box()
        self.refresh_session_label()
        self.refresh_tasks()
        self.refresh_sensors_table()
        self.refresh_quality()
        self.refresh_claps_table()
        self.refresh_plots()
        self.refresh_events()
        self.refresh_video_label()
        self.refresh_banner()

    def refresh_title(self) -> None:
        code = self.project.technician
        self.act_technician.setText(f"👤 {code}" if code else "👤 Técnico")
        title = "SAVES Studio" + (f" — {self._session_title}" if self._session_title else "")
        self.setWindowTitle(title + (f" · técnico {code}" if code else ""))

    def refresh_tasks(self) -> None:
        """The checklist: one row per task, with a state the technician can change."""
        if self.project is None:
            return
        project = self.project
        keys = project.task_keys()
        table = self.tasks_table
        if list(self._task_combos) != keys:
            # The rows change only when a kind of event is added or removed;
            # otherwise the combos are kept (one may be emitting right now).
            table.setRowCount(0)
            table.setRowCount(len(keys))
            self._task_combos = {}
            for row, key in enumerate(keys):
                combo = QComboBox()
                combo.setFocusPolicy(Qt.NoFocus)
                for value in TASK_STATES:
                    combo.addItem(TASK_STATE_TEXT[value], value)
                combo.currentIndexChanged.connect(
                    lambda index, k=key, c=combo: self.set_task_state(k, c.itemData(index))
                )
                table.setCellWidget(row, 1, combo)
                self._task_combos[key] = combo
        for row, key in enumerate(keys):
            state = project.task_state(key)
            table.setItem(row, 0, readonly_item(project.task_name(key), bold=state != FINISHED))
            combo = self._task_combos[key]
            combo.blockSignals(True)
            combo.setCurrentIndex(TASK_STATES.index(state))
            combo.blockSignals(False)
            combo.setStyleSheet(f"QComboBox{{color:{TASK_STATE_COLORS[state]}; font-weight:bold;}}")
            task = project.tasks.get(key, {})
            who = ""
            if state != NOT_STARTED:
                when = _short_time(task.get("finished_at") if state == FINISHED else task.get("started_at", ""))
                who = " · ".join(x for x in (task.get("by", ""), when) if x)
                if task.get("copied_from"):
                    who += f" (copiada de {task['copied_from']})"
                if state == FINISHED and key in redcap.COUNT_FIELDS and project.events.get(key):
                    if not redcap.needs_sending(project, key):
                        who += f" · en REDCap {_short_time(task.get('redcap_sent_at', ''))}"
                    elif self.redcap_configured():
                        who += " · falta enviarlo a REDCap"
            table.setItem(row, 2, readonly_item(who))
        table.resizeRowsToContents()
        height = table.horizontalHeader().height() + sum(table.rowHeight(r) for r in range(len(keys))) + 2 * table.frameWidth()
        table.setFixedHeight(height)
        pending = len(project.pending_tasks())
        who = f" · técnico {project.technician}" if project.technician else ""
        self.tasks_box.setTitle(
            f"Tareas de este vídeo{who} · " + (f"{pending} sin finalizar" if pending else "todas finalizadas")
        )

    def set_task_state(self, key: str, state: str) -> None:
        """The technician sets the state of a task (only they finish one)."""
        if self.project is None or key not in self.project.task_keys():
            return
        if state == self.project.task_state(key):
            return
        if state == FINISHED and not self._confirm_finished(key):
            self.refresh_tasks()
            return
        self.project.set_task_state(key, state)
        self.statusBar().showMessage(f"«{self.project.task_name(key)}»: {TASK_STATE_TEXT[state].lower()}", 4000)
        self._schedule_save()
        self.refresh_tasks()
        self.refresh_events()
        self.refresh_banner()

    def _confirm_finished(self, key: str) -> bool:
        """Ask before finishing a task that does not look done."""
        if key == SYNC_TASK:
            if self.engine is None:
                return self._confirm("Sincronización", "La sesión no tiene manifiesto, así que no hay sincronización. "
                                     "¿Marcarla como finalizada igualmente?")
            bad = [sid for sid in self.engine.active_sensors() if self.engine.status(sid).quality in ("none", "poor")]
            if self.loading:
                bad = bad or ["(aún se están leyendo archivos)"]
            if bad:
                return self._confirm("Sincronización", (
                    f"Hay sensores sin ajustar o en rojo: {', '.join(bad)}.\n\n¿Marcar la sincronización como finalizada igualmente?"
                ))
            return True
        if not self.project.events.get(key):
            name = self.project.event_type(key).lower_name
            return self._confirm("Conteo", (
                f"No hay ninguna marca de {name}.\n\n¿Marcar el conteo como finalizado igualmente "
                "(por ejemplo, porque en este vídeo no hay ninguno)?"
            ))
        return True

    def _confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def refresh_reference_box(self) -> None:
        self.reference_box.blockSignals(True)
        self.reference_box.clear()
        self.reference_box.addItem("Automática", "auto")
        for entry in self.project.sensors:
            self.reference_box.addItem(entry.sensor_id, entry.sensor_id)
        self.reference_box.addItem("Vídeo (palmadas marcadas)", VIDEO)
        index = self.reference_box.findData(self.project.reference)
        self.reference_box.setCurrentIndex(max(index, 0))
        if self.engine is not None and self.project.reference == "auto":
            self.reference_box.setItemText(0, f"Automática: {self.engine.reference_label()}")
        self.reference_box.blockSignals(False)

    def refresh_session_label(self) -> None:
        lines = []
        if self.engine is not None:
            s = self.engine.session
            lines.append(
                f"<b>{s.session_id}</b> · {clock(s.start_time.timestamp())}–{clock(s.end_time.timestamp())[:-4]}"
                f" · {len(s.activities)} actividades · {len(s.sync_windows)} palmadas o saltos"
            )
        else:
            lines.append("<span style='color:#c62828'>Falta session_manifest.json: arrástralo a la ventana.</span>")
        if self.project.videos:
            v = self.project.videos[0]
            extra = "" if v.recording_start_unix_s is not None else " (sin hora de inicio)"
            lines.append(f"🎥 {Path(v.path).name}{extra}")
        else:
            lines.append("🎥 Sin vídeo")
        self.session_label.setText("<br>".join(lines))

    def refresh_sensors_table(self) -> None:
        if self.project is None:
            return
        table = self.sensors_table
        table.blockSignals(True)
        table.setRowCount(len(self.project.sensors))
        all_ids = [s.sensor_id for s in self.project.sensors]
        for row, entry in enumerate(self.project.sensors):
            check = QCheckBox()
            check.setChecked(entry.enabled)
            check.setToolTip("Incluir en la sincronización")
            check.toggled.connect(lambda on, sid=entry.sensor_id: self._toggle_sensor(sid, on))
            holder = QWidget()
            h = QHBoxLayout(holder)
            h.setContentsMargins(4, 0, 0, 0)
            h.addWidget(check)
            table.setCellWidget(row, 0, holder)
            name = readonly_item(entry.sensor_id, QColor(sensor_color(all_ids.index(entry.sensor_id))), bold=True)
            name.setToolTip(f"{FORMAT_LABELS.get(entry.sensor_format, entry.sensor_format)}\n{entry.path}\n{entry.serial}")
            table.setItem(row, 1, name)
            combo = QComboBox()
            combo.addItems(LOCATIONS)
            if entry.location not in LOCATIONS:
                combo.addItem(entry.location)  # From the manifest (e.g. with side).
            combo.setCurrentText(entry.location)
            combo.setFocusPolicy(Qt.NoFocus)
            combo.currentTextChanged.connect(lambda text, sid=entry.sensor_id: self._location_changed(sid, text))
            table.setCellWidget(row, 2, combo)
            signal = self.engine.signals.get(entry.sensor_id) if self.engine else None
            table.setItem(row, 3, readonly_item(f"{signal.sampling_rate_hz:g}" if signal else ""))
            if entry.sensor_id in self.loading:
                status = readonly_item(f"⏳ {self.loading[entry.sensor_id]}…", QColor("#1565c0"))
            elif entry.sensor_id in self.errors:
                status = readonly_item("✗ Error al leer", QColor("#c62828"))
                status.setToolTip(self.errors[entry.sensor_id])
            elif signal is not None:
                hours = signal.duration_s / 3600
                span = f"{hours:.1f} h" if hours >= 1 else f"{signal.duration_s / 60:.0f} min"
                status = readonly_item(f"✓ {span} · {FORMAT_LABELS.get(entry.sensor_format, '')}")
            else:
                status = readonly_item("—")
            table.setItem(row, 4, status)
        table.blockSignals(False)

    def refresh_quality(self) -> None:
        table = self.quality_table
        if self.engine is None:
            table.setRowCount(0)
            self.video_quality.setText("")
            return
        ids = self.engine.active_sensors()
        table.setRowCount(len(ids))
        all_ids = [s.sensor_id for s in self.project.sensors]
        for row, sid in enumerate(ids):
            st = self.engine.status(sid)
            table.setItem(row, 0, readonly_item(sid, QColor(sensor_color(all_ids.index(sid))), bold=True))
            state = readonly_item(f"● {QUALITY_TEXT[st.quality]}", QColor(QUALITY_COLORS[st.quality]), bold=True)
            state.setToolTip(st.note)
            table.setItem(row, 1, state)
            table.setItem(row, 2, readonly_item(f"{st.n_anchors}/{st.n_windows}"))
            table.setItem(row, 3, readonly_item("" if st.offset_ms is None else f"{st.offset_ms / 1000:+.3f} s"))
            table.setItem(row, 4, readonly_item("" if st.drift_ppm is None else f"{st.drift_ppm:+.0f} ppm"))
            table.setItem(row, 5, readonly_item("" if st.rmse_ms is None else f"{st.rmse_ms:.1f} ms",
                                                residual_color(st.rmse_ms) if st.rmse_ms is not None else None, bold=True))
        if self.project.videos:
            n = len(self.project.video_events)
            residuals = self.engine.video_residuals_ms()
            text = f"🎥 Vídeo: {n} palmada(s) marcada(s); hora {VIDEO_SOURCE_TEXT[self.engine.video_fit_source]}"
            if residuals:
                rmse = (sum(r * r for r in residuals.values()) / len(residuals)) ** 0.5
                text += f". Error {rmse:.0f} ms (1 fotograma = {1000 / self.video.reader.fps:.0f} ms)" if self.video.reader else ""
            self.video_quality.setText(text)
        else:
            self.video_quality.setText("")

    def refresh_claps_table(self) -> None:
        table = self.claps_table
        if self.engine is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        ids = self.engine.active_sensors()
        has_video = bool(self.project.videos)
        headers = ["Evento", "Tipo", "Actividad", "Fase"] + (["Vídeo"] if has_video else []) + ids
        table.blockSignals(True)
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        windows = self.engine.session.sync_windows
        table.setRowCount(len(windows))
        names = {a.activity_index: a.activity_name for a in self.engine.session.activities}
        video_res = self.engine.video_residuals_ms()
        phase_text = {"static_prep": "antes", "static_post": "después", "transition_lap": "transición", "activity_main": "durante"}
        for row, window in enumerate(windows):
            eid = window.event_id
            table.setItem(row, 0, readonly_item(eid, bold=True))
            table.setItem(row, 1, readonly_item("salto" if window.sync_kind == "jump" else "palmada"))
            table.setItem(row, 2, readonly_item(names.get(window.activity_index) or f"Actividad {window.activity_index}"))
            table.setItem(row, 3, readonly_item(phase_text.get(window.phase, window.phase)))
            col = 4
            if has_video:
                if eid in self.project.video_events:
                    r = video_res.get(eid)
                    text = "✓" + (f" {r:+.0f} ms" if r is not None else "")
                    table.setItem(row, col, readonly_item(text, residual_color(r) if r is not None else QColor("#1565c0")))
                else:
                    table.setItem(row, col, readonly_item("marcar", QColor("#9e9e9e")))
                col += 1
            for sid in ids:
                state = self.engine.anchor_state(sid, eid)
                r = self.engine.residual_ms(sid, eid)
                if state == "missing":
                    item = readonly_item("✗ no hallada", QColor("#c62828"))
                elif state == "excluded":
                    item = readonly_item("⊘ excluida", QColor("#9e9e9e"))
                else:
                    mark = "✋" if state == "manual" else "✓"
                    if sid == self.engine.resolved_reference():
                        item = readonly_item(f"{mark} ref.", QColor("#1565c0"))
                    else:
                        item = readonly_item(f"{mark} {r:+.0f} ms" if r is not None else mark, residual_color(r))
                table.setItem(row, col, item)
                col += 1
        table.resizeColumnsToContents()
        table.blockSignals(False)
        if self._selected_event:
            self._select_row(self._selected_event, focus=False)

    def refresh_plots(self) -> None:
        if self.engine is None:
            self.plots.set_engine(None)
            self._plotted = []
            return
        hidden = set(self.project.hidden_plots)
        shown = [sid for sid in self.engine.active_sensors() if sid not in hidden]
        if shown != self._plotted or self.plots.engine is not self.engine:
            view = self.plots.view_range() if self.plots.plots else None
            self.plots.set_engine(self.engine)
            if view:
                self.plots.focus(*view)
            self._plotted = list(shown)
        else:
            self.plots.refresh_overview()
            self.plots.refresh_markers()
            self.plots.refresh_data()
        self._video_frame(self.video.current_time)

    def refresh_video_label(self) -> None:
        self._video_frame(self.video.current_time)

    # ------------------------------------------------------------------
    # Next-step banner
    # ------------------------------------------------------------------

    def refresh_banner(self) -> None:
        text, button, action, tone = self._next_step()
        colors = {"info": "#e3f2fd", "warn": "#fff8e1", "ok": "#e8f5e9", "error": "#ffebee"}
        self.banner.setStyleSheet(f"QFrame{{background:{colors[tone]}; border-radius:6px;}}")
        self.banner_label.setText(text)
        self.banner_button.setVisible(bool(button))
        self.banner_button.setText(button or "")
        self._banner_action = action

    def _banner_clicked(self) -> None:
        if self._banner_action:
            self._banner_action()

    def _next_step(self):
        if self.engine is None:
            return ("<b>Falta el manifiesto de la sesión</b> (session_manifest.json). Arrástralo a la ventana.",
                    "Añadir archivos…", self.choose_files, "error")
        if not self.project.sensors:
            return ("<b>Añade los sensores:</b> arrastra los archivos .cwa (Axivity), .gt3x (ActiGraph), .bin (GENEActiv o Parmay Matrix) o .csv (SENS).",
                    "Añadir archivos…", self.choose_files, "info")
        if self.loading:
            return (f"⏳ Cargando {len(self.loading)} archivo(s) y buscando palmadas y saltos. La primera vez tarda más; "
                    "luego se abre desde la caché.", None, None, "info")
        ref = self.engine.resolved_reference()
        if ref is None:
            return ("<b>No hay referencia.</b> Elige un sensor como referencia arriba o marca al menos 2 palmadas o saltos en el vídeo.",
                    None, None, "warn")
        for sid in self.engine.active_sensors():
            st = self.engine.status(sid)
            if st.quality == "none":
                missing = next((w.event_id for w in self.engine.session.sync_windows
                                if self.engine.anchor_state(sid, w.event_id) == "missing"), None)
                return (f"<b>{sid}</b> no comparte 2 palmadas o saltos con la referencia. Revisa sus ventanas: "
                        "arrastra la línea o haz clic derecho en la señal para poner el evento a mano.",
                        "Ir a la primera" if missing else None,
                        (lambda e=missing: self.focus_event(e)) if missing else None, "warn")
        for sid in self.engine.active_sensors():
            st = self.engine.status(sid)
            if st.quality == "poor" or (st.quality == "fair" and st.n_pairs >= 3):
                worst = self.engine.worst_event(sid)
                r = self.engine.residual_ms(sid, worst)
                return (f"<b>Revisa {sid}:</b> el evento {worst} se desvía {r:+.0f} ms. Corrígelo arrastrando su línea "
                        "o exclúyelo (clic derecho).", "Ir al evento", lambda e=worst: self.focus_event(e), "warn")
        if self.project.videos and ref != VIDEO and self.engine.video_fit_source != "claps":
            pending = next((w.event_id for w in self.engine.session.sync_windows
                            if w.event_id not in self.project.video_events), None)
            audio = " o pulsa «Detectar palmadas y saltos en el audio»" if self.video.reader else ""
            return (f"<b>Marca las palmadas o saltos en el vídeo</b> (al menos 2){audio}. Al marcar uno, se pasa solo al siguiente.",
                    "Empezar", lambda e=pending: self.focus_event(e), "info")
        pending = self.project.pending_tasks()
        if SYNC_TASK in pending:
            return ("✅ <b>La sincronización está bien.</b> Revisa la tabla de calidad y, cuando hayas terminado, "
                    "márcala como finalizada.",
                    "✓ Sincronización finalizada", lambda: self.set_task_state(SYNC_TASK, FINISHED), "ok")
        if pending:
            # The kind being counted first, if it is still open.
            key = self._event_key if self._event_key in pending else pending[0]
            kind = self.project.event_type(key)
            n = len(self.project.events.get(key, []))
            so_far = f" Llevas {n}." if n else ""
            return (f"<b>Falta el conteo de {kind.lower_name}</b> "
                    f"({TASK_STATE_TEXT[self.project.task_state(key)].lower()}).{so_far} Cuenta en la pestaña «Conteo» "
                    "y márcalo como finalizado al terminar; si esta sesión no lo tiene, quítalo con «Quitar».",
                    "Ir al conteo", lambda k=key: self.show_event_kind(k), "info")
        unsent = [k for k in redcap.countable_kinds(self.project) if redcap.needs_sending(self.project, k)]
        if unsent and self.redcap_configured():
            names = " y ".join(self.project.event_type(k).lower_name for k in unsent)
            return (f"✅ <b>Todas las tareas están finalizadas.</b> Envía a REDCap el conteo de {names} "
                    "y exporta los resultados.", "☁ Enviar a REDCap", self.send_to_redcap, "ok")
        return ("✅ <b>Todas las tareas están finalizadas.</b> Exporta los resultados.",
                "📤 Exportar", self.export, "ok")

    # ==================================================================
    # Claps: selection, video marks and corrections
    # ==================================================================

    def _event_at(self, row: int) -> Optional[str]:  # noqa: UP045
        item = self.claps_table.item(row, 0)
        return item.text() if item else None

    def _select_row(self, event_id: str, focus: bool = True) -> None:
        for row in range(self.claps_table.rowCount()):
            if self._event_at(row) == event_id:
                self.claps_table.blockSignals(True)
                self.claps_table.selectRow(row)
                self.claps_table.blockSignals(False)
                self.claps_table.scrollToItem(self.claps_table.item(row, 0))
                break
        self._selected_event = event_id
        if focus:
            self.focus_event(event_id)

    def _clap_selected(self) -> None:
        rows = self.claps_table.selectionModel().selectedRows()
        if rows:
            event_id = self._event_at(rows[0].row())
            if event_id != self._selected_event:
                self._selected_event = event_id
                self.focus_event(event_id)

    def select_relative(self, step: int) -> None:
        if self.engine is None or not self.engine.session.sync_windows:
            return
        ids = [w.event_id for w in self.engine.session.sync_windows]
        if self._selected_event in ids:
            index = (ids.index(self._selected_event) + step) % len(ids)
        else:
            index = 0 if step > 0 else len(ids) - 1
        self._select_row(ids[index])

    def focus_event(self, event_id: Optional[str]) -> None:  # noqa: UP045
        """Zoom the plots on a sync window and take the video just before its clap."""
        if self.engine is None or event_id is None:
            return
        if self._selected_event != event_id:
            self._select_row(event_id, focus=False)
        t0, t1 = self.engine.window_display_range(event_id)
        self.plots.focus(t0 - 2.0, t1 + 2.0)
        if self.video.reader is not None:
            if event_id in self.project.video_events:
                self.video.seek(self.project.video_events[event_id])
            else:
                clap = self.engine.display_time_of_anchor(event_id)
                target = (clap - 0.7) if clap is not None else t0
                self.video.seek(float(self.engine.display_to_video(target)))

    def mark_video_clap(self) -> None:
        if self.engine is None or self.video.current_time is None:
            return
        event_id = self._selected_event
        if event_id is None:
            display = float(self.engine.video_to_display(self.video.current_time))
            event_id = self.plots._nearest_window(display).event_id
        self.project.video_events[event_id] = float(self.video.current_time)
        self._changed(message=f"Palmada {event_id} marcada en el vídeo a {self.video.current_time:.3f} s")
        # Go straight to the next clap still unmarked in the video.
        pending = [w.event_id for w in self.engine.session.sync_windows if w.event_id not in self.project.video_events]
        if pending:
            ids = [w.event_id for w in self.engine.session.sync_windows]
            after = [e for e in pending if ids.index(e) > ids.index(event_id)]
            self._select_row((after or pending)[0])

    def clear_video_clap(self) -> None:
        if self.project and self._selected_event in self.project.video_events:
            del self.project.video_events[self._selected_event]
            self._changed()

    def _anchor_moved(self, sensor_id: str, event_id: str, display_s: float) -> None:
        if self.engine is None:
            return
        native = float(self.engine.display_to_sensor(sensor_id, display_s))
        self.project.set_manual_anchor(sensor_id, event_id, native)
        self._changed(message=f"Palmada {event_id} de {sensor_id} corregida a mano")

    def _anchor_action(self, sensor_id: str, event_id: str, action: str) -> None:
        if action == "exclude":
            self.project.set_excluded(sensor_id, event_id, True)
        elif action == "include":
            self.project.set_excluded(sensor_id, event_id, False)
        elif action == "auto":
            self.project.clear_manual_anchor(sensor_id, event_id)
        elif action == "focus":
            self.focus_event(event_id)
            return
        self._changed()

    def _clap_menu(self, pos) -> None:
        index = self.claps_table.indexAt(pos)
        if not index.isValid() or self.engine is None:
            return
        event_id = self._event_at(index.row())
        header = self.claps_table.horizontalHeaderItem(index.column()).text()
        menu = QMenu(self)
        menu.addAction("Ir a este evento", lambda: self.focus_event(event_id))
        if header == "Vídeo":
            if self.video.current_time is not None:
                menu.addAction("Marcar en el fotograma actual del vídeo", lambda: (self._select_row(event_id, focus=False), self.mark_video_clap()))
            if event_id in self.project.video_events:
                menu.addAction("Borrar la marca del vídeo", lambda: (self.project.video_events.pop(event_id, None), self._changed()))
        elif header in self.engine.signals:
            state = self.engine.anchor_state(header, event_id)
            if state in ("auto", "manual"):
                menu.addAction(f"Excluir del ajuste de {header}", lambda: self._anchor_action(header, event_id, "exclude"))
            if state == "excluded":
                menu.addAction("Volver a incluir", lambda: self._anchor_action(header, event_id, "include"))
            if state == "manual":
                menu.addAction("Volver a la detección automática", lambda: self._anchor_action(header, event_id, "auto"))
            if self.video.current_time is not None:
                menu.addAction(
                    f"Poner el evento de {header} en el instante del vídeo",
                    lambda: self._anchor_moved(header, event_id, float(self.engine.video_to_display(self.video.current_time))),
                )
        menu.exec(self.claps_table.viewport().mapToGlobal(pos))

    def _quality_double_clicked(self, row: int, _col: int) -> None:
        sid = self.quality_table.item(row, 0).text()
        worst = self.engine.worst_event(sid) if self.engine else None
        if worst:
            self.focus_event(worst)

    # ==================================================================
    # Video and plots interaction
    # ==================================================================

    def _video_frame(self, video_s: Optional[float]) -> None:  # noqa: UP045
        if self.engine is None or video_s is None:
            self.plots.set_cursor(None)
            self.video.set_host_text("")
            return
        display = float(self.engine.video_to_display(video_s))
        self.plots.set_cursor(display, follow=self.video.playing)
        self.video.set_host_text(f"Hora de sesión {clock(display)}  ({VIDEO_SOURCE_TEXT[self.engine.video_fit_source]})")
        self.video.set_events_text(self._events_text(display))

    def _plot_clicked(self, display_s: float) -> None:
        if self.engine is not None and self.video.reader is not None:
            self.video.pause()
            self.video.seek(float(self.engine.display_to_video(display_s)))

    def _toggle_xyz(self, enabled: bool) -> None:
        self.plots.set_xyz(enabled)

    # ==================================================================
    # Sensors list edits
    # ==================================================================

    def _toggle_sensor(self, sensor_id: str, enabled: bool) -> None:
        self.project.sensor(sensor_id).enabled = enabled
        self._changed()

    def _location_changed(self, sensor_id: str, location: str) -> None:
        self.project.sensor(sensor_id).location = location
        self._plotted = []  # Titles show the location.
        self._changed()

    def _sensor_menu(self, pos) -> None:
        row = self.sensors_table.indexAt(pos).row()
        if row < 0:
            return
        sid = self.project.sensors[row].sensor_id
        menu = QMenu(self)
        menu.addAction("Usar como referencia", lambda: self._set_reference(sid))
        if sid in self.project.hidden_plots:
            menu.addAction("Mostrar su gráfico", lambda: self._set_plot_hidden(sid, False))
        else:
            menu.addAction("Ocultar su gráfico", lambda: self._set_plot_hidden(sid, True))
        menu.addAction("Volver a leer el archivo", lambda: self._load_sensor(sid))
        menu.addAction("Abrir la carpeta del archivo", lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(self.project.sensor(sid).path).parent))))
        menu.addSeparator()
        menu.addAction("Quitar de la sesión", lambda: self._remove_sensor(sid))
        menu.exec(self.sensors_table.viewport().mapToGlobal(pos))

    def _remove_sensor(self, sensor_id: str) -> None:
        self.project.remove_sensor(sensor_id)
        if self.engine is not None:
            self.engine.drop_signal(sensor_id)
        self.loading.pop(sensor_id, None)
        self._changed()

    def _set_reference(self, value: str) -> None:
        self.project.reference = value
        self._changed()

    # ------------------------------------------------------------------
    # Which plots are shown
    # ------------------------------------------------------------------

    def _plot_candidates(self) -> list[str]:
        return self.engine.active_sensors() if self.engine is not None else []

    def _fill_plots_menu(self) -> None:
        menu = self.plots_menu
        menu.clear()
        ids = self._plot_candidates()
        if not ids:
            menu.addAction("Aún no hay gráficos").setEnabled(False)
            return
        hidden = set(self.project.hidden_plots)
        for sid in ids:
            action = menu.addAction(f"{sid} · {self.project.sensor(sid).location}")
            action.setCheckable(True)
            action.setChecked(sid not in hidden)
            action.toggled.connect(lambda on, s=sid: self._set_plot_hidden(s, not on))
        groups: dict[str, list[str]] = {}
        for sid in ids:
            groups.setdefault(_location_group(self.project.sensor(sid).location), []).append(sid)
        if len(groups) > 1:
            menu.addSeparator()
            for group, members in groups.items():
                menu.addAction(f"Ver solo: {group} ({len(members)})", lambda m=members: self._show_only(m))
        menu.addSeparator()
        menu.addAction("Mostrar todos", lambda: self._show_only(ids)).setEnabled(bool(hidden & set(ids)))

    def _set_plot_hidden(self, sensor_id: str, hidden: bool) -> None:
        if self.project is None:
            return
        self.project.set_plot_hidden(sensor_id, hidden)
        self._schedule_save()
        self.refresh_plots()
        self.refresh_events()

    def _show_only(self, sensor_ids: list[str]) -> None:
        if self.project is None:
            return
        keep = set(sensor_ids)
        for sid in self._plot_candidates():
            self.project.set_plot_hidden(sid, sid not in keep)
        self._schedule_save()
        self.refresh_plots()
        self.refresh_events()

    def _plot_visibility(self, sensor_id: str, action: str) -> None:
        if action == "only":
            self._show_only([sensor_id])
        else:
            self._set_plot_hidden(sensor_id, True)

    def _reference_changed(self, _index: int) -> None:
        value = self.reference_box.currentData()
        if self.project is not None and value is not None and value != self.project.reference:
            self._set_reference(value)

    # ==================================================================
    # Events counted on the video (steps, sit-to-stands or any other)
    # ==================================================================

    def _session_phases(self):
        if self._phases is None and self.engine is not None:
            self._phases = read_phases(self.project.folder, self.engine.session)
        return self._phases

    def _activity_names(self) -> dict[int, str]:
        if self.engine is None:
            return {}
        return {a.activity_index: a.activity_name or f"Actividad {a.activity_index}" for a in self.engine.session.activities}

    def _event_kind(self) -> Optional[EventType]:  # noqa: UP045
        """The kind of event being counted (S, A and D mark it)."""
        if self.project is None or not self.project.event_types:
            return None
        try:
            return self.project.event_type(self._event_key)
        except KeyError:
            self._event_key = self.project.event_types[0].key
            return self.project.event_types[0]

    def show_event_kind(self, key: str) -> None:
        self._event_key = key
        self.tabs.setCurrentIndex(1)
        self.refresh_events()

    def _event_kind_changed(self, index: int) -> None:
        key = self.event_kind_box.itemData(index)
        if key and key != self._event_key:
            self._event_key = key
            self.refresh_events()

    def mark_event(self, side: str = "") -> None:
        """Event of the kind being counted at the frame on screen (keys S, A and D)."""
        kind = self._event_kind()
        if kind is None or self.video.current_time is None:
            return
        if side and not kind.sides:
            self.statusBar().showMessage(f"«{kind.name}» no distingue izquierda y derecha: marca con S", 4000)
            return
        event = self.project.add_event(kind.key, self.video.current_time, side)
        self._event_history.setdefault(kind.key, []).append(("add", event))
        side_text = {"L": " (izquierda)", "R": " (derecha)"}.get(side, "")
        self._events_changed(kind.key, f"{kind.name}: {len(self.project.events_of(kind.key))}, "
                                       f"el último a {event['video_s']:.2f} s{side_text}")

    def remove_event(self) -> None:
        kind = self._event_kind()
        if kind is None or self.video.current_time is None:
            return
        removed = self.project.remove_event_near(kind.key, self.video.current_time)
        if removed is None:
            self.statusBar().showMessage(f"No hay ninguna marca de {kind.lower_name} a menos de 0,5 s del fotograma actual", 4000)
            return
        self._event_history.setdefault(kind.key, []).append(("remove", removed))
        self._events_changed(kind.key, f"{kind.name}: borrada la marca de {removed['video_s']:.2f} s")

    def undo_event(self) -> None:
        kind = self._event_kind()
        if kind is None:
            return
        history = self._event_history.get(kind.key)
        if not history:
            self.statusBar().showMessage(f"No hay nada que deshacer en {kind.lower_name}", 3000)
            return
        action, event = history.pop()
        if action == "add":
            self.project.events[kind.key] = [e for e in self.project.events_of(kind.key) if e is not event]
        else:
            self.project.add_event(kind.key, event["video_s"], event.get("side", ""))
        self._events_changed(kind.key, f"{kind.name}: deshecho")

    def _events_changed(self, key: str, message: str) -> None:
        timeout = 3000
        if self.project.work_on(key) == "reopened":
            message += (f". La tarea «{self.project.task_name(key)}» vuelve a «Iniciada»: "
                        "márcala como finalizada cuando termines")
            timeout = 10000
        self.statusBar().showMessage(message, timeout)
        self._schedule_save()
        self.refresh_events()
        self.refresh_tasks()
        self.refresh_banner()

    def add_event_kind(self) -> None:
        """A new kind of event to count in this session."""
        if self.project is None:
            return
        dialog = EventTypeDialog(self._remembered_kinds(), self)
        if dialog.exec() != QDialog.Accepted:
            return
        name, sides, new_sessions = dialog.values()
        if not name:
            return
        kind = self.project.add_event_type(name, sides)
        self._remember_kind(kind.name, kind.sides, new_sessions)
        self._event_key = kind.key
        self.tabs.setCurrentIndex(1)
        self._schedule_save()
        self.refresh_all()
        self.statusBar().showMessage(f"Ahora cuentas «{kind.name}»: S en el fotograma de cada uno", 5000)

    def remove_event_kind(self) -> None:
        """Take a kind of event out of this session (only one with nothing marked)."""
        kind = self._event_kind()
        if kind is None:
            return
        try:
            self.project.remove_event_type(kind.key)
        except ValueError:
            self.statusBar().showMessage(f"«{kind.name}» tiene marcas: bórralas antes de quitarlo", 5000)
            return
        remembered = next((k for k in self._remembered_kinds() if k.get("name") == kind.name), None)
        if remembered and remembered.get("new_sessions") and self._confirm(
            "Sesiones nuevas", f"«{kind.name}» se añade a las sesiones nuevas de este ordenador. ¿Dejar de añadirlo?"
        ):
            self._remember_kind(kind.name, kind.sides, False)
        self._event_key = self.project.event_types[0].key if self.project.event_types else STEPS
        self._event_history.pop(kind.key, None)
        self._schedule_save()
        self.refresh_all()
        self.statusBar().showMessage(f"«{kind.name}» quitado de esta sesión", 4000)

    def _remembered_kinds(self) -> list[dict]:
        """Kinds of events created on this computer: name, sides and whether new sessions get them."""
        raw = self.settings.value("event_types", "[]")
        try:
            kinds = json.loads(raw if isinstance(raw, str) else "[]")
        except ValueError:
            return []
        return [k for k in kinds if isinstance(k, dict) and k.get("name")]

    def _remember_kind(self, name: str, sides: bool, new_sessions: bool) -> None:
        kinds = [k for k in self._remembered_kinds() if k.get("name") != name]
        kinds.append({"name": name, "sides": sides, "new_sessions": new_sessions})
        self.settings.setValue("event_types", json.dumps(kinds[-30:], ensure_ascii=False))

    def refresh_events(self) -> None:
        if self.project is None:
            return
        kind = self._event_kind()
        box = self.event_kind_box
        box.blockSignals(True)
        box.clear()
        for k in self.project.event_types:
            n = len(self.project.events.get(k.key, []))
            box.addItem(f"{k.name} ({n})" if n else k.name, k.key)
        box.setCurrentIndex(max(box.findData(self._event_key), 0))
        box.blockSignals(False)
        if kind is None:
            self.events_hint.setText("No hay nada que contar en esta sesión. Añade lo que quieras contar con «➕ Otro evento…».")
            self.remove_kind_button.setEnabled(False)
            self.finish_kind_button.setVisible(False)
            self.events_table.setRowCount(0)
            self.plots.set_events([])
            self.tabs.setTabText(1, "🔢 Conteo")
            self._video_frame(self.video.current_time)
            return
        times = self.project.event_times(kind.key)
        self.remove_kind_button.setEnabled(not times)
        finished = self.project.task_state(kind.key) == FINISHED
        self.finish_kind_button.setVisible(True)
        self.finish_kind_button.setEnabled(not finished)
        self.finish_kind_button.setText("✓ Conteo finalizado" if finished else "Finalizar el conteo")
        self.finish_kind_button.setToolTip(
            "Cámbialo en la lista de tareas si tienes que volver a él" if finished
            else f"Marca como finalizada la tarea «{self.project.task_name(kind.key)}»"
        )
        sides = " (<b>A</b> izquierda, <b>D</b> derecha si las distingues)" if kind.sides else ""
        self.events_hint.setText(
            f"Reproduce el vídeo despacio (0,25× o 0,5×) y pulsa <b>S</b> en {kind.mark_hint}{sides}. "
            "<b>Retroceso</b> borra la marca más cercana y <b>Ctrl+Z</b> deshace. "
            "Doble clic en una actividad: ir a su inicio."
        )
        color = event_color(kind.key)
        self.video.set_event_type(kind.button_text, kind.sides, color)
        self.tabs.setTabText(1, f"🔢 Conteo · {kind.name} ({len(times)})" if times else f"🔢 Conteo · {kind.name}")
        if self.engine is None:
            self.plots.set_events([], color)
            self.events_table.setRowCount(0)
            return
        display = [float(t) for t in np.atleast_1d(self.engine.video_to_display(times))] if times else []
        self.plots.set_events(display, color)
        phases = self._session_phases()
        names = self._activity_names()
        self._event_counts = {}
        if display:
            labels = label_times(np.asarray(display, dtype=float), phases, names)
            for act, att, phase in zip(labels["activity_index"], labels["attempt"], labels["phase"]):
                if _is_na(act):
                    continue
                key = (int(act), 1 if _is_na(att) else int(att))
                main, total = self._event_counts.get(key, (0, 0))
                self._event_counts[key] = (main + (phase == "activity_main"), total + 1)
        steps = kind.key == STEPS
        table = self.events_table
        table.setHorizontalHeaderLabels(["Actividad", "En la actividad", "Con las fases quietas",
                                         "Cadencia" if steps else "Por minuto", "Duración"])
        activities = self.engine.session.activities
        table.setRowCount(len(activities))
        for row, activity in enumerate(activities):
            index, attempt = activity.activity_index, activity.attempt
            main_phases = (
                phases[(phases["activity_index"] == index) & (phases["attempt"] == attempt)
                       & (phases["phase"] == "activity_main")]
                if phases is not None else []
            )
            main_s = float((main_phases["end_s"] - main_phases["start_s"]).sum()) if len(main_phases) else activity.duration_seconds
            n_main, n_all = self._event_counts.get((index, attempt), (0, 0))
            rate = ""
            if main_s > 0 and n_main:
                rate = f"{60.0 * n_main / main_s:.0f} pasos/min" if steps else f"{60.0 * n_main / main_s:.1f}/min"
            name = names.get(index, "") + (f" (intento {attempt})" if attempt > 1 else "")
            item = readonly_item(name, bold=bool(n_all))
            item.setData(Qt.UserRole, row)
            table.setItem(row, 0, item)
            table.setItem(row, 1, readonly_item(str(n_main) if n_all else "", QColor(color), bold=True))
            table.setItem(row, 2, readonly_item(str(n_all) if n_all else ""))
            table.setItem(row, 3, readonly_item(rate))
            table.setItem(row, 4, readonly_item(f"{main_s:.0f} s"))
        self._video_frame(self.video.current_time)

    def _events_row_clicked(self, row: int, _col: int) -> None:
        item = self.events_table.item(row, 0)
        if item is None or self.engine is None:
            return
        row = item.data(Qt.UserRole)
        activities = self.engine.session.activities
        if not isinstance(row, int) or not 0 <= row < len(activities):
            return
        activity = activities[row]
        start, end = activity.start_time.timestamp(), activity.end_time.timestamp()
        self.plots.focus(start - 5.0, min(end, start + 60.0))
        if self.video.reader is not None:
            self.video.pause()
            self.video.seek(float(self.engine.display_to_video(start)))

    def _events_text(self, display: float) -> str:
        """Events so far of the kind being counted and those of the activity on screen."""
        kind = self._event_kind()
        if kind is None or self.engine is None:
            return ""
        n = len(self.project.events.get(kind.key, []))
        if not n:
            return f"{kind.name}: ninguno todavía (S en {kind.mark_hint})"
        phases = self._session_phases()
        if phases is None or not len(phases):
            return f"{kind.name}: {n}"
        here = label_times(np.asarray([display], dtype=float), phases, self._activity_names())
        index, attempt = here["activity_index"][0], here["attempt"][0]
        if _is_na(index):
            return f"{kind.name}: {n} en total"
        main, total = self._event_counts.get((int(index), 1 if _is_na(attempt) else int(attempt)), (0, 0))
        return (f"{kind.name}: {n} en total · «{here['activity'][0]}»: "
                f"{main} en la actividad ({total} con las fases quietas)")

    # ==================================================================
    # Audio, export, saving
    # ==================================================================

    def detect_audio(self, auto: bool = False) -> None:
        """Find the claps and landings in the sound of the video."""
        if self.engine is None or not self.project.videos:
            return
        path = self.project.videos[0].path
        try:
            sound = has_audio(path)
        except Exception:  # noqa: BLE001 - unreadable video
            sound = False
        if not sound:
            if not auto:
                QMessageBox.information(self, "El vídeo no tiene sonido", (
                    "Este vídeo se grabó sin audio, así que las palmadas y los saltos hay que marcarlos a mano.\n\n"
                    "Para las próximas sesiones, graba con audio en lab_protocol_timer "
                    "y se detectarán solos."
                ))
            return
        if self.engine.video_fit is None:
            self.engine.refit()  # the video is placed on the session clock
        # How far the video may be from where it is placed: little when the
        # timer logged its start or claps are marked, the whole video when not.
        if self.engine.video_fit_source in ("claps", "one_clap"):
            search_s = 3.0
        elif self.engine.video_fit_source == "recording_start":
            search_s = 10.0
        else:
            search_s = max(float(self.engine.video_duration_s or 0.0), 3600.0)
        margin = 3.0
        windows, expected = {}, {}
        for w in self.engine.session.sync_windows:
            if w.expected_time is not None:
                # Around the cue: from just before the go to 3.5 s after it.
                cue = w.cue_time.timestamp()
                windows[w.event_id] = (
                    float(self.engine.display_to_video(cue - 1.0)),
                    float(self.engine.display_to_video(cue + 3.5)),
                )
                expected[w.event_id] = float(self.engine.display_to_video(w.expected_time.timestamp()))
            else:
                windows[w.event_id] = (
                    float(self.engine.display_to_video(w.window_start.timestamp() - margin)),
                    float(self.engine.display_to_video(w.window_end.timestamp() + margin)),
                )
        self.statusBar().showMessage("Buscando palmadas y saltos en el audio del vídeo…")

        def done(match):
            if self.project is None:
                return
            found = match.found
            new = {e: t for e, t in found.items() if e not in self.project.video_events}
            self.project.video_events.update(new)
            # Found on its own when the session opens: not the technician's work yet.
            self._changed(None if auto else SYNC_TASK)
            kept = len(found) - len(new)
            msg = f"Audio: {len(found)} de {len(windows)} palmadas o saltos encontrados, {len(new)} añadidos"
            if kept:
                msg += f" ({kept} ya estaban marcados a mano y no se tocan)"
            if match.searched and abs(match.shift_s) >= 1.0:
                msg += f"; el vídeo estaba {abs(match.shift_s):.1f} s {'después' if match.shift_s > 0 else 'antes'} de lo previsto"
            if len(found) < len(windows):
                msg += ". Revisa los que faltan en la columna «Vídeo»."
            self.statusBar().showMessage(msg, 15000)

        def failed(error):
            if auto:
                self.statusBar().showMessage(f"No se pudo analizar el audio: {error}", 8000)
            else:
                QMessageBox.warning(self, "Audio", f"No se pudo analizar el audio:\n{error}")

        self._run(lambda _p: find_audio_events(path, windows, expected_video_s=expected, search_s=search_s),
                  done, failed)

    def export_folder(self) -> Path:
        """saves_sync/<technician> in the session folder (saves_sync/ without technician)."""
        base = Path(self.project.folder) / "saves_sync"
        return base / self.project.technician if self.project.technician else base

    def export(self) -> None:
        if self.engine is None or not self.engine.signals:
            return
        self._save_timer.stop()
        self._save()
        folder = str(self.export_folder())
        pending = self.project.pending_tasks()
        box = QMessageBox(self)
        box.setWindowTitle("Exportar")
        text = f"Se exportará la sesión sincronizada y etiquetada en:\n{folder}"
        if pending:
            box.setIcon(QMessageBox.Warning)
            lines = "\n".join(
                f"• {self.project.task_name(k)}: {TASK_STATE_TEXT[self.project.task_state(k)].lower()}" for k in pending
            )
            text = (f"Faltan tareas por finalizar:\n{lines}\n\nSi exportas ahora, se anotará en tasks.csv "
                    f"y en el LEEME.\n\n{text}")
        box.setText(text)
        aligned = QCheckBox("Incluir también la grabación completa de cada sensor, alineada (archivos grandes)")
        box.setCheckBox(aligned)
        export_button = box.addButton("Exportar igualmente" if pending else "Exportar", QMessageBox.AcceptRole)
        other_button = box.addButton("Otra carpeta…", QMessageBox.ActionRole)
        cancel_button = box.addButton("Cancelar", QMessageBox.RejectRole)
        box.setDefaultButton(cancel_button if pending else export_button)
        box.exec()
        if box.clickedButton() is other_button:
            folder = QFileDialog.getExistingDirectory(self, "Carpeta de resultados", str(self.project.folder))
            if not folder:
                return
        elif box.clickedButton() is not export_button:
            return
        export_aligned = aligned.isChecked()
        engine = self.engine
        self.statusBar().showMessage("Exportando…")

        def done(paths):
            self.statusBar().showMessage(f"Exportados {len(paths)} archivos en {folder}", 8000)
            box = QMessageBox(self)
            box.setWindowTitle("Exportación terminada")
            box.setText(f"Se han guardado {len(paths)} archivos en:\n{folder}")
            open_button = box.addButton("Abrir carpeta", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Close)
            box.exec()
            if box.clickedButton() is open_button:
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

        self._run(lambda _p: engine.export(folder, export_aligned), done,
                  lambda e: QMessageBox.warning(self, "Exportar", f"No se pudo exportar:\n{e}"))

    def redcap_configured(self) -> bool:
        """A REDCap token is set (in Studio or in REDCAP_TOKEN); REDCap is optional."""
        return bool(saved_token(self.settings) or redcap.env_token())

    def redcap_settings(self) -> None:
        """«Conexión con REDCap»: the API address and the token of this Windows user."""
        dialog = RedcapSettingsDialog(self.settings, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        if dialog.removed:
            message = "Token de REDCap quitado: los conteos se guardan solo en los archivos exportados."
        elif self.redcap_configured():
            message = f"Conexión con REDCap guardada ({saved_url(self.settings)})."
        else:
            message = "Sin token de REDCap: los conteos se guardan solo en los archivos exportados."
        self.statusBar().showMessage(message, 8000)
        if self.project is not None:
            self.refresh_tasks()
            self.refresh_banner()

    def _redcap_token(self, csv_path: Optional[Path]) -> tuple[str, bool]:  # noqa: UP045
        """The technician's REDCap token (set in Studio, else REDCAP_TOKEN, else asked now) and whether it is the saved one."""
        token = saved_token(self.settings)
        if token:
            return token, True
        if redcap.env_token():
            return redcap.env_token(), False
        dialog = RedcapSettingsDialog(self.settings, csv_path=csv_path or redcap.CSV_NAME, parent=self)
        if dialog.exec() != QDialog.Accepted or dialog.removed:
            return "", False
        return dialog.typed_token(), bool(saved_token(self.settings))

    def _redcap_failed(self, error: Exception, csv_path: Optional[Path], saved: bool = False) -> None:  # noqa: UP045
        extra = ""
        if isinstance(error, redcap.TokenRejected) and saved:
            self.settings.remove("redcap_token")
            extra = "\n\nEl token guardado se ha borrado: la próxima vez se pedirá de nuevo."
        if csv_path is not None:
            extra += f"\n\nLos conteos están en {csv_path}; se pueden subir con Data Import Tool de REDCap."
        QMessageBox.warning(self, "REDCap", f"No se pudo enviar a REDCap.\n\n{error}{extra}")

    def send_to_redcap(self) -> None:
        """Sends the finished counts to the lab_activity instance of each attempt in REDCap."""
        if self.engine is None or self.project is None:
            return
        participants = list(self.engine.session.participants)
        if not participants:
            QMessageBox.warning(self, "REDCap", "El manifiesto de la sesión no dice de qué participante es.")
            return
        participant = participants[0]
        if len(participants) > 1:
            labels = [f"{p.participant_id} · {p.redcap_event_name}" for p in participants]
            choice, ok = QInputDialog.getItem(self, "REDCap", "¿De qué participante son los conteos?", labels, 0, False)
            if not ok:
                return
            participant = participants[labels.index(choice)]
        self._save_timer.stop()
        self._save()
        try:
            prepared = redcap.count_records(self.engine, participant)
        except redcap.RedcapError as error:
            QMessageBox.information(self, "REDCap", str(error))
            return
        left_out = "".join(f"\n• {line}" for line in prepared.left_out)
        if not prepared.records:
            QMessageBox.information(self, "REDCap", "No hay ningún conteo que enviar."
                                    + (f"\n\nSe quedan fuera:{left_out}" if left_out else ""))
            return
        try:
            csv_path = redcap.write_import_csv(prepared.records, self.export_folder() / redcap.CSV_NAME)
        except OSError:
            csv_path = None
        token, saved = self._redcap_token(csv_path)
        if not token:
            where = f" Los conteos están en {csv_path}." if csv_path is not None else ""
            self.statusBar().showMessage(f"No se ha enviado nada a REDCap.{where}", 10000)
            return
        url = saved_url(self.settings)
        try:
            with busy():
                info = redcap.project_info(token, url)
                if not info.longitudinal:
                    # A project without events: the visit only named the folder.
                    prepared.records = redcap.for_project(prepared.records, longitudinal=False)
                    if csv_path is not None:
                        redcap.write_import_csv(prepared.records, csv_path)
                existing = redcap.current_rows(token, participant, url=url, longitudinal=info.longitudinal)
        except redcap.RedcapError as error:
            self._redcap_failed(error, csv_path, saved)
            return
        except OSError as error:
            self._redcap_failed(redcap.RedcapError(str(error)), csv_path, saved)
            return
        problems = redcap.check_instances(prepared, existing)
        if problems:
            QMessageBox.warning(self, "REDCap", "No se envía nada:\n\n" + "\n".join(f"• {p}" for p in problems[:8]))
            return
        others = redcap.counts_of_others(prepared, existing)

        lines = []
        for key in prepared.kinds:
            n = prepared.sent_counts.get(key, 0)
            lines.append(f"• {self.project.event_type(key).name}: {n} actividad{'es' if n != 1 else ''}")
        visit = f" ({participant.redcap_event_name})" if info.longitudinal else ""
        project_name = f" del proyecto «{info.title}»" if info.title else ""
        text = (f"Se enviará a lab_activity{project_name} en REDCap, registro {participant.participant_id}"
                f"{visit}, lo contado por {prepared.technician}:\n" + "\n".join(lines))
        if left_out:
            text += f"\n\nNo se envían:{left_out}"
        box = QMessageBox(self)
        box.setWindowTitle("Enviar a REDCap")
        if others:
            box.setIcon(QMessageBox.Warning)
            who = "; ".join(f"{self.project.event_type(k).lower_name} de {', '.join(sorted(codes))}"
                            for k, codes in others.items())
            text += f"\n\nEn REDCap ya hay conteos de otro técnico ({who}). Si envías, se sustituyen por los tuyos."
        box.setText(text)
        send_button = box.addButton("Sustituir y enviar" if others else "Enviar", QMessageBox.AcceptRole)
        cancel_button = box.addButton("Cancelar", QMessageBox.RejectRole)
        box.setDefaultButton(cancel_button if others else send_button)
        box.exec()
        if box.clickedButton() is not send_button:
            return
        try:
            with busy():
                redcap.import_records(token, prepared.records, url=url)
        except redcap.RedcapError as error:
            self._redcap_failed(error, csv_path, saved)
            return
        redcap.mark_sent(self.project, prepared.kinds)
        self._save()
        self.refresh_tasks()
        self.refresh_banner()
        self.statusBar().showMessage(
            f"Enviado a REDCap: {len(prepared.records)} actividades de {participant.participant_id}", 8000)

    def _changed(self, task: Optional[str] = SYNC_TASK, message: str = "") -> None:  # noqa: UP045
        """Something of the synchronization changed (``task``: the task it belongs to, if the technician did it)."""
        timeout = 4000
        if task is not None and self.project is not None and self.project.work_on(task) == "reopened":
            message = (f"{message}. " if message else "") + (
                f"La tarea «{self.project.task_name(task)}» vuelve a «Iniciada» porque has cambiado algo: "
                "márcala como finalizada cuando termines")
            timeout = 10000
        if message:
            self.statusBar().showMessage(message, timeout)
        if self.engine is not None:
            self.engine.refit()
        self._schedule_save()
        self.refresh_all()

    def _schedule_save(self) -> None:
        self._save_timer.start()

    def save_now(self) -> None:
        self._save_timer.stop()
        if self._save():
            self.statusBar().showMessage(f"Guardado en {self.project.project_path.name} (carpeta de la sesión)", 5000)

    def _save(self) -> bool:
        if self.project is None:
            return False
        try:
            path = self.project.save()
        except OSError as exc:
            self.statusBar().showMessage(f"No se pudo guardar el proyecto: {exc}", 8000)
            self.saved_label.setText("⚠ Sin guardar")
            self.saved_label.setToolTip(str(exc))
            return False
        self.saved_label.setText(f"Guardado {datetime.now():%H:%M:%S}")
        self.saved_label.setToolTip(str(path))
        return True

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_timer.stop()
        self._save()
        self.video.close_video()
        super().closeEvent(event)
