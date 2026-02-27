from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvc
from typing_extensions import Self

from pupil_labs.neon_usb.pyrav4l2 import Device, v4l2

from ..frame import Frame
from ..v4lstream import V4lStream
from .camera import CameraNotFoundError, CameraSpec


class CameraBackend(ABC):
    def __init__(self, spec: CameraSpec):
        self.spec = spec

    @abstractmethod
    def get_frame(self) -> Frame:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.close()


class UVCBackend(CameraBackend):
    _uvc_capture: uvc.Capture

    def __init__(self, spec: CameraSpec, extended_controls: Any = None):
        super().__init__(spec)

        self._uvc_capture = None
        self.spec = spec
        self.extended_controls = extended_controls
        self.exposure_controls = None

        connected_devices = uvc.device_list()
        uid = None
        for device_info in connected_devices:
            if (device_info["idVendor"], device_info["idProduct"]) == (
                self.spec.vendor_id,
                self.spec.product_id,
            ):
                uid = device_info["uid"]
                break

        if uid is None:
            raise CameraNotFoundError(self.spec.name)

        capture = uvc.Capture(uid, self.extended_controls)
        capture.bandwidth_factor = self.spec.bandwidth_factor

        mode_matched = False
        for mode in capture.available_modes:
            if (mode.width, mode.height, mode.fps) == (
                self.spec.width,
                self.spec.height,
                self.spec.fps,
            ):
                mode_matched = True
                capture.frame_mode = mode
                self._uvc_capture = capture

                break

        if not mode_matched:
            capture.close()
            raise OSError(
                f"None of the available modes matched: {capture.available_modes}!"
            )

    def get_frame(self) -> Frame:
        if self._uvc_capture is None:
            raise OSError("Camera not initialized!")

        frame = self._uvc_capture.get_frame(timeout=2.0)
        assert frame is not None
        return Frame(frame.img, frame.timestamp, frame.index)

    def close(self) -> None:
        if self._uvc_capture is not None:
            self._uvc_capture.close()
            del self._uvc_capture
        self._uvc_capture = None


class V4l2Backend(CameraBackend):
    def __init__(self, spec: CameraSpec):
        super().__init__(spec)

        self.camera_reinit_timeout = 3
        self.device = None
        self.frame_counter = -1

        errors = {}
        for device_path in Path("/dev/").glob("video*"):
            try:
                device = Device(device_path)
            except (AttributeError, FileNotFoundError, PermissionError) as e:
                errors[device_path] = e
                continue

            if self.spec.name in device.device_name and device.is_video_capture_capable:
                formats = [
                    (color_format, frame_size, frame_interval)
                    for color_format, frame_sizes in device.available_formats.items()
                    for frame_size in frame_sizes
                    for frame_interval in device.get_available_frame_intervals(
                        color_format, frame_size
                    )
                ]
                for color_format, frame_size, frame_interval in formats:
                    fps = frame_interval.denominator / frame_interval.numerator
                    if (frame_size.width, frame_size.height, fps) == (
                        self.spec.width,
                        self.spec.height,
                        self.spec.fps,
                    ):
                        device.set_format(color_format, frame_size)
                        device.set_frame_interval(frame_interval)
                        self.device = device
                        self.stream = V4lStream(self.device)
                        self.stream.open()
                        self._fd = open(self.device.path)  # noqa: SIM115
                        self.color_format, _ = self.device.get_format()

                        break

                else:
                    raise OSError("None of the available modes matched!")

        if self.device is None:
            raise CameraNotFoundError(self.spec.name)

    def get_frame(self) -> Frame:
        frame = self.stream.get_frame()
        assert frame is not None, "Failed to get frame from stream!"
        buffer, time_ns = frame
        if buffer is None:
            raise TimeoutError

        if self.color_format.pixelformat == v4l2.V4L2_PIX_FMT_GREY:
            pixels = np.frombuffer(buffer, dtype=np.uint8).reshape([
                self.spec.height,
                self.spec.width,
            ])
        elif self.color_format.pixelformat == v4l2.V4L2_PIX_FMT_MJPEG:
            pixels = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)

        self.frame_counter += 1

        return Frame(pixels, time_ns / 1e9, self.frame_counter)

    def close(self) -> None:
        self._fd.close()