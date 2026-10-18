from mad_gui import BaseImporter

from readers.sens import read_sens


class SensImporter(BaseImporter):
    """
    Importer for SENS accelerometer CSV files.

    The importer preserves the native sensor timestamps so that the
    synchronization pipeline can later relate sensor events to video time.
    """

    loadable_file_type = "*.csv"

    @classmethod
    def name(cls) -> str:
        """Return the human-readable importer name."""
        return "SENS file"

    def load_sensor_data(self, file_path: str) -> dict:
        """
        Load a SENS accelerometer CSV file.

        Parameters
        ----------
        file_path
            Path to the SENS CSV file.

        Returns
        -------
        dict
            Dictionary containing the accelerometer data and estimated
            sampling frequency.

        Notes
        -----
        The original ``unixts`` timestamps are preserved in the output
        dataframe as ``timestamp_unix_ms``. These timestamps are part of
        the sensor's native time domain and must not be replaced by a
        reconstructed time axis.
        """
        # Parsing lives in readers.sens so it can run without the GUI.
        recording = read_sens(file_path)

        return {
            "SENS motion (acceleration)": {
                "sensor_data": recording.data,
                "sampling_rate_hz": recording.sampling_rate_hz,
            }
        }