from mad_gui import BaseImporter

from readers.cwa import read_cwa


class AxivityImporter(BaseImporter):
    """
    Importer for Axivity AX3 / AX6 binary files (.cwa).

    Native timestamps are kept as ``timestamp_unix_ms`` (UTC). The device
    clock is assumed to have been set in Europe/Madrid local time.
    """

    loadable_file_type = "*.cwa"

    @classmethod
    def name(cls) -> str:
        """Return the human-readable importer name."""
        return "Axivity AX3/AX6 (*.cwa)"

    def load_sensor_data(self, file_path: str) -> dict:
        """Load an Axivity .cwa file."""
        recording = read_cwa(file_path)

        return {
            "Axivity (acceleration)": {
                "sensor_data": recording.data,
                "sampling_rate_hz": recording.sampling_rate_hz,
            }
        }
