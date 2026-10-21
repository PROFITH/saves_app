"""The REDCap API token of the technician, asked once per computer."""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

TOKEN_PATTERN = re.compile(r"^[0-9A-Fa-f]{32}$")


class RedcapTokenDialog(QDialog):
    """Asks for the token; without one, the counts stay in a file to import by hand."""

    def __init__(self, csv_path: str | Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Token de REDCap")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Para enviar los conteos hace falta tu token de la API de REDCap (en el proyecto, "
            "<i>API → Request API token</i>, con permiso de importación). Son 32 letras y números."
            f"<br><br>Sin token, los conteos quedan en <b>{Path(csv_path).name}</b>, en tu carpeta de "
            "exportación, y se pueden subir con <i>Data Import Tool</i>."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText("Token de la API")
        self.token_edit.textChanged.connect(self._update)
        layout.addWidget(self.token_edit)
        self.remember_check = QCheckBox("Recordarlo en este ordenador (solo para tu usuario de Windows)")
        self.remember_check.setChecked(True)
        layout.addWidget(self.remember_check)
        self.status = QLabel()
        self.status.setStyleSheet("color:#c62828;")
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Usar este token")
        self.buttons.button(QDialogButtonBox.Cancel).setText("Cancelar")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update("")

    def _update(self, text: str) -> None:
        token = text.strip()
        valid = bool(TOKEN_PATTERN.match(token))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(valid)
        self.status.setText("" if valid or not token else "Un token de REDCap tiene 32 letras (A-F) y números.")

    def values(self) -> tuple[str, bool]:
        """The token and whether to keep it on this computer."""
        return self.token_edit.text().strip(), self.remember_check.isChecked()
