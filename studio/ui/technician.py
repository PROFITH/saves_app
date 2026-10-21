"""
Dialogs about who does the work: the technician's code when a session is
opened, and a new kind of event to count on the video.

Each technician keeps their work apart in the session folder
(``saves_project_<code>.json``), so two technicians can review the same
video independently and compare.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from studio.project import SYNC_TASK, SyncProject, normalize_technician, project_file


class TechnicianDialog(QDialog):
    """Asks for the technician's code and says what will be opened."""

    def __init__(self, folder: str | Path, last_code: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Técnico")
        self.setMinimumWidth(520)
        self.folder = Path(folder)
        self.others: dict[str, SyncProject] = {}
        self.unreadable: list[str] = []
        for code in SyncProject.technicians_in(self.folder):
            try:
                self.others[code] = SyncProject.load(self.folder, code)
            except (OSError, ValueError, KeyError, TypeError):
                self.unreadable.append(code)
        self.legacy = SyncProject.exists_in(self.folder) and not self.others and not self.unreadable

        layout = QVBoxLayout(self)
        intro = QLabel(
            f"<b>{self.folder.parent.name}/{self.folder.name}</b><br>Escribe tu código de técnico (tus iniciales, p. ej. JHM). "
            "Tu trabajo se guarda aparte en la carpeta de la sesión: puedes dejarlo a medias y seguir otro día, "
            "y otro técnico puede revisar el mismo vídeo por su cuenta."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Código:"))
        self.code_edit = QLineEdit(normalize_technician(last_code))
        self.code_edit.setMaxLength(12)
        self.code_edit.setPlaceholderText("JHM")
        self.code_edit.setValidator(QRegularExpressionValidator(QRegularExpression("[A-Za-z0-9_-]{0,12}"), self))
        self.code_edit.textEdited.connect(self._upper)
        self.code_edit.textChanged.connect(self._update)
        row.addWidget(self.code_edit, 1)
        layout.addLayout(row)

        if self.others or self.unreadable:
            lines = [f"<b>{code}</b>: {project.progress_summary()}" for code, project in self.others.items()]
            lines += [f"<b>{code}</b>: <span style='color:#c62828'>archivo ilegible</span>" for code in self.unreadable]
            others = QLabel("Trabajo guardado en esta carpeta:<br>" + "<br>".join(lines))
            others.setWordWrap(True)
            others.setStyleSheet("color:#444;")
            layout.addWidget(others)

        copy_row = QHBoxLayout()
        self.copy_check = QCheckBox("Empezar con la sincronización de")
        self.copy_box = QComboBox()
        for code, project in self.others.items():
            if project.has_sync_work() or project.task_state(SYNC_TASK) != "not_started":
                self.copy_box.addItem(code, code)
        self.copy_check.setToolTip(
            "Copia las palmadas marcadas y las correcciones de ese técnico; los conteos no se copian. "
            "Déjalo sin marcar para una revisión independiente."
        )
        self.copy_check.toggled.connect(self._update)
        copy_row.addWidget(self.copy_check)
        copy_row.addWidget(self.copy_box)
        copy_row.addStretch()
        layout.addLayout(copy_row)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#1565c0;")
        layout.addWidget(self.status)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Abrir")
        self.buttons.button(QDialogButtonBox.Cancel).setText("Cancelar")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update()
        self.code_edit.setFocus()
        self.code_edit.selectAll()

    @property
    def code(self) -> str:
        return normalize_technician(self.code_edit.text())

    def _upper(self, text: str) -> None:
        if text != text.upper():
            position = self.code_edit.cursorPosition()
            self.code_edit.setText(text.upper())
            self.code_edit.setCursorPosition(position)

    def _update(self, *_args) -> None:
        code = self.code
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(code))
        mine = code in self.others or code in self.unreadable
        can_copy = bool(code) and not mine and self.copy_box.count() > 0
        self._can_copy = can_copy
        self.copy_check.setVisible(can_copy)
        self.copy_box.setVisible(can_copy)
        self.copy_box.setEnabled(self.copy_check.isChecked())
        if not code:
            text = "Escribe tu código para continuar."
        elif code in self.unreadable:
            text = "Tu archivo de trabajo no se puede leer: se apartará con otro nombre y empezarás de cero."
        elif mine:
            saved = (self.folder / project_file(code)).stat().st_mtime
            text = f"Seguirás donde lo dejaste (guardado el {datetime.fromtimestamp(saved):%d/%m %H:%M})."
        elif self.legacy:
            text = "Seguirás con el trabajo guardado en esta carpeta antes de los códigos de técnico."
        elif can_copy and self.copy_check.isChecked():
            text = (f"Empezarás con la sincronización de {self.copy_box.currentText()}; "
                    "los conteos los haces tú desde cero.")
        else:
            text = "Empezarás de cero en esta carpeta."
        self.status.setText(text)

    def choice(self) -> tuple[str, Optional[str]]:  # noqa: UP045
        """The code, and whose synchronization to copy (None: none)."""
        code = self.code
        if self._can_copy and self.copy_check.isChecked() and self.copy_box.currentData():
            return code, str(self.copy_box.currentData())
        return code, None


class EventTypeDialog(QDialog):
    """A new kind of event to count on the video."""

    def __init__(self, remembered: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Otro evento para contar")
        self.setMinimumWidth(460)
        self.remembered = {str(r.get("name", "")): r for r in remembered if r.get("name")}
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Nombre de lo que vas a contar en el vídeo (por ejemplo «Escalones» o «Saltos»). "
            "Se añade a esta sesión con su propia tarea en la lista."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.name_box = QComboBox()
        self.name_box.setEditable(True)
        self.name_box.addItems(list(self.remembered))
        self.name_box.setCurrentText("")
        self.name_box.lineEdit().setMaxLength(40)
        self.name_box.lineEdit().setPlaceholderText("Escalones")
        self.name_box.currentTextChanged.connect(self._update)
        layout.addWidget(self.name_box)
        self.sides_check = QCheckBox("Distinguir izquierda y derecha (teclas A y D)")
        layout.addWidget(self.sides_check)
        self.new_sessions_check = QCheckBox("Añadirlo también a las sesiones nuevas que se abran en este ordenador")
        layout.addWidget(self.new_sessions_check)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Añadir")
        self.buttons.button(QDialogButtonBox.Cancel).setText("Cancelar")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update("")

    def _update(self, text: str) -> None:
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(text.strip()))
        known = self.remembered.get(text.strip())
        if known is not None:
            self.sides_check.setChecked(bool(known.get("sides")))
            self.new_sessions_check.setChecked(bool(known.get("new_sessions")))

    def values(self) -> tuple[str, bool, bool]:
        """Name, whether it has sides and whether new sessions get it."""
        name = " ".join(self.name_box.currentText().split())
        return name, self.sides_check.isChecked(), self.new_sessions_check.isChecked()
