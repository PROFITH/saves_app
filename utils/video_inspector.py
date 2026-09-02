from pathlib import Path
from typing import Union

import cv2


class VideoInspector:
    """
    Inspect the temporal and basic technical properties of a video file.

    This class is intentionally limited to metadata inspection and frame
    timestamp inspection. It does not perform synchronization or annotation.
    """

    def inspect(self, file_path: Union[str, Path]) -> dict:  # noqa: UP007
        """
        Inspect the basic properties of a video file.

        Parameters
        ----------
        file_path
            Path to the video file.

        Returns
        -------
        dict
            Basic video metadata.

        Raises
        ------
        FileNotFoundError
            If the video file does not exist.
        ValueError
            If OpenCV cannot open the video.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Video path is not a file: {path}"
            )

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            capture.release()
            raise ValueError(
                f"Could not open video file: {path}"
            )

        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

            duration_seconds = None

            if fps > 0 and frame_count >= 0:
                duration_seconds = frame_count / fps

            return {
                "file_path": str(path.resolve()),
                "file_name": path.name,
                "fps": fps,
                "frame_count": int(frame_count),
                "duration_seconds": duration_seconds,
                "width": int(width),
                "height": int(height),
                "codec": self._get_codec(capture),
                "backend": capture.getBackendName(),
            }

        finally:
            capture.release()

    def inspect_frames(
        self,
        file_path: Union[str, Path],  # noqa: UP007
        frame_indices: list[int],
    ) -> list[dict]:
        """
        Inspect the timestamps of selected video frames.

        Parameters
        ----------
        file_path
            Path to the video file.

        frame_indices
            Frame numbers to inspect. Frame numbering starts at zero.

        Returns
        -------
        list[dict]
            One dictionary per requested frame containing:

            - frame_index
            - timestamp_seconds
            - timestamp_milliseconds
            - reported_fps

        Notes
        -----
        OpenCV reports the current frame position through
        CAP_PROP_POS_MSEC. This value is useful for investigating the
        temporal structure of the video, but should not yet be assumed
        to be the authoritative presentation timestamp of the MP4.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {path}"
            )

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            capture.release()
            raise ValueError(
                f"Could not open video file: {path}"
            )

        try:
            frame_count = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            fps = capture.get(cv2.CAP_PROP_FPS)

            results = []

            for frame_index in frame_indices:
                if frame_index < 0 or frame_index >= frame_count:
                    raise ValueError(
                        f"Frame index {frame_index} is outside the "
                        f"valid range 0-{frame_count - 1}."
                    )

                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_index,
                )

                success, _ = capture.read()

                if not success:
                    raise ValueError(
                        f"Could not read frame {frame_index}."
                    )

                timestamp_ms = capture.get(
                    cv2.CAP_PROP_POS_MSEC
                )

                results.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp_ms / 1000.0,
                        "timestamp_milliseconds": timestamp_ms,
                        "reported_fps": fps,
                    }
                )

            return results

        finally:
            capture.release()

    @staticmethod
    def _get_codec(capture: cv2.VideoCapture) -> str:
        """
        Return the four-character video codec reported by OpenCV.
        """
        codec_value = int(
            capture.get(cv2.CAP_PROP_FOURCC)
        )

        if codec_value == 0:
            return "unknown"

        codec = "".join(
            chr((codec_value >> (8 * i)) & 0xFF)
            for i in range(4)
        )

        return codec.strip() or "unknown"


    def inspect_frame_sequence(
        self,
        file_path: Union[str, Path],  # noqa: UP007
        start_frame: int = 0,
        number_of_frames: int = 100,
    ) -> list[dict]:
        """
        Inspect timestamps while reading consecutive frames sequentially.

        Parameters
        ----------
        file_path
            Path to the video file.

        start_frame
            Frame at which to start reading.

        number_of_frames
            Number of consecutive frames to inspect.

        Returns
        -------
        list[dict]
            Frame index and timestamp information for each decoded frame.

        Notes
        -----
        Frames are decoded sequentially rather than accessed using random
        frame seeking. This provides a more reliable assessment of the
        temporal structure reported by the video decoder.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {path}"
            )

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            capture.release()
            raise ValueError(
                f"Could not open video file: {path}"
            )

        try:
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                start_frame,
            )

            results = []

            for frame_offset in range(number_of_frames):
                success, _ = capture.read()

                if not success:
                    break

                frame_index = start_frame + frame_offset

                timestamp_ms = capture.get(
                    cv2.CAP_PROP_POS_MSEC
                )

                results.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp_ms / 1000.0,
                        "timestamp_milliseconds": timestamp_ms,
                    }
                )

            return results

        finally:
            capture.release()

    def inspect_time_range(
        self,
        file_path: Union[str, Path],  # noqa: UP007
        start_seconds: float,
        end_seconds: float,
    ) -> list[dict]:
        """
        Inspect all decoded frames within a temporal interval.

        Parameters
        ----------
        file_path
            Path to the video file.

        start_seconds
            Start of the interval in video time.

        end_seconds
            End of the interval in video time.

        Returns
        -------
        list[dict]
            One dictionary per decoded frame containing:

            - frame_index
            - timestamp_seconds
            - timestamp_milliseconds

        Notes
        -----
        Frames are decoded sequentially. The timestamps reported by the video
        decoder are used instead of calculating time from frame index and FPS.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Video file does not exist: {path}"
            )

        if start_seconds < 0:
            raise ValueError(
                "start_seconds must be greater than or equal to zero."
            )

        if end_seconds <= start_seconds:
            raise ValueError(
                "end_seconds must be greater than start_seconds."
            )

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            capture.release()
            raise ValueError(
                f"Could not open video file: {path}"
            )

        try:
            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                start_seconds * 1000.0,
            )

            results = []

            while True:
                success, _ = capture.read()

                if not success:
                    break

                frame_index = int(
                    capture.get(cv2.CAP_PROP_POS_FRAMES)
                ) - 1

                timestamp_ms = capture.get(
                    cv2.CAP_PROP_POS_MSEC
                )

                timestamp_seconds = timestamp_ms / 1000.0

                if timestamp_seconds > end_seconds:
                    break

                if timestamp_seconds >= start_seconds:
                    results.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp_seconds,
                            "timestamp_milliseconds": timestamp_ms,
                        }
                    )

            return results

        finally:
            capture.release()