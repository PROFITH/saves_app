from typing import ClassVar, Optional
from mad_gui.plot_tools.labels.base_label import BaseRegionLabel

BASE_DESCRIPTIONS: dict[str, Optional[dict]] = {
    "Actividad Personalizada": None,
    **{f"Activity {i}": None for i in range(1, 21)},
    "Reposo": ["Sedestacion", "Bipedestacion", "Decubito"],
    "Locomocion": ["Caminar", "Correr", "Subir escaleras", "Bajar escaleras"],
    "Impacto / Salto": None,
    "Transicion Postural": None,
    "Artefacto / No-Wear": None,
}


class SensActivityLabel(BaseRegionLabel):
    name = "SENS (acceleration)"
    min_height = 0
    max_height = 1
    descriptions: ClassVar[dict[str, Optional[dict]]] = BASE_DESCRIPTIONS


class MatrixActivityLabel(BaseRegionLabel):
    name = "Matrix (acceleration)"
    min_height = 0
    max_height = 1
    descriptions: ClassVar[dict[str, Optional[dict]]] = BASE_DESCRIPTIONS


class Sens1ActivityLabel(BaseRegionLabel):
    name = "SENS #1"
    min_height = 0
    max_height = 1
    descriptions: ClassVar[dict[str, Optional[dict]]] = BASE_DESCRIPTIONS


class Matrix1ActivityLabel(BaseRegionLabel):
    name = "Matrix #1"
    min_height = 0
    max_height = 1
    descriptions: ClassVar[dict[str, Optional[dict]]] = BASE_DESCRIPTIONS


class Matrix2ActivityLabel(BaseRegionLabel):
    name = "Matrix #2"
    min_height = 0
    max_height = 1
    descriptions: ClassVar[dict[str, Optional[dict]]] = BASE_DESCRIPTIONS