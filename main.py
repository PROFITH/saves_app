from mad_gui import start_gui

from importers.combined_importer import UniversalStudyImporter
from importers.matrix_importer import MatrixImporter
from importers.sens_importer import SensImporter
from labels.activity_labels import (
    Matrix1ActivityLabel,
    Matrix2ActivityLabel,
    MatrixActivityLabel,
    Sens1ActivityLabel,
    SensActivityLabel,
)


def main():
    start_gui(
        plugins=[
            UniversalStudyImporter,
            SensImporter,
            MatrixImporter,
        ],
        labels=[
            SensActivityLabel,
            MatrixActivityLabel,
            Sens1ActivityLabel,
            Matrix1ActivityLabel,
            Matrix2ActivityLabel,
        ],
    )


if __name__ == "__main__":
    main()