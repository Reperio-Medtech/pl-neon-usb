import os
import time
from typing import NamedTuple

import numpy as np

from pupil_labs.neon_usb.cameras.backend import UVCBackend, V4l2Backend
from pupil_labs.neon_usb.cameras.camera import Camera, CameraSpec
from pupil_labs.neon_usb.frame import Frame
from pupil_labs.neon_usb.pyrav4l2.controls import Menu
from pupil_labs.neon_usb.usb_utils import get_calibration


class SceneIntrinsics(NamedTuple):
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    exterinsics_affine_matrix: np.ndarray


NEON_SCENE_CAMERA_SPEC = CameraSpec(
    name="Neon Scene Camera v1",
    vendor_id=0x0BDA,
    product_id=0x3036,
    width=1600,
    height=1200,
    fps=30,
    bandwidth_factor=1.2,
)

# V4L2 spec for scene camera — resolution/fps configurable via env vars
# to allow experimenting with lower bandwidth modes.
#   NEON_SCENE_V4L2_WIDTH   (default: 800)
#   NEON_SCENE_V4L2_HEIGHT  (default: 600)
#   NEON_SCENE_V4L2_FPS     (default: 30)
_v4l2_width = int(os.getenv("NEON_SCENE_V4L2_WIDTH", "800"))
_v4l2_height = int(os.getenv("NEON_SCENE_V4L2_HEIGHT", "600"))
_v4l2_fps = int(os.getenv("NEON_SCENE_V4L2_FPS", "30"))

NEON_SCENE_CAMERA_V4L2_SPEC = CameraSpec(
    name="Neon Scene Camera v1",
    vendor_id=0x0BDA,
    product_id=0x3036,
    width=_v4l2_width,
    height=_v4l2_height,
    fps=_v4l2_fps,
    bandwidth_factor=0.8,
)


class SceneCamera(Camera):
    """Provides an interface for handling the Neon scene camera.

    The class is assuming that no more than one Neon device is connected to the
    computer at the same time.
    """

    def __init__(self, spec: CameraSpec = NEON_SCENE_CAMERA_SPEC) -> None:
        """Initialize the scene camera of the connected Neon device.

        The camera stream will be started right away. If the object fails to grab
        frames, it will automatically try to reinitialize.
        """
        super().__init__(NEON_SCENE_CAMERA_SPEC, UVCBackend)

        assert isinstance(self.backend, UVCBackend)
        self.uvc_controls = {
            c.display_name: c for c in self.backend._uvc_capture.controls
        }
        camera_parameters = {
            "Backlight Compensation": 2,
            "Brightness": 0,
            "Contrast": 32,
            "Gain": 64,
            "Hue": 0,
            "Saturation": 64,
            "Sharpness": 50,
            "Gamma": 300,
            "Auto Exposure Mode": 1,
            "Absolute Exposure Time": 250,
        }
        for key, value in camera_parameters.items():
            try:
                self.uvc_controls[key].value = value
            except KeyError:
                print(f"Setting {key} to {value} failed: Unknown control. Known ")

    @staticmethod
    def get_intrinsics() -> SceneIntrinsics:
        """Retrieve the scene camera intrinsics of the Neon device

        Returns:
            Tuple containing camera matrix and distortion coefficients of scene camera.

        """
        calib_data = get_calibration()
        return SceneIntrinsics(
            calib_data.scene_camera_matrix,
            calib_data.scene_distortion_coefficients,
            calib_data.scene_extrinsics_affine_matrix,
        )

    @property
    def exposure(self) -> int:
        value = self.uvc_controls["Absolute Exposure Time"].value
        assert isinstance(value, int)
        return value

    @exposure.setter
    def exposure(self, value: int) -> None:
        self.uvc_controls["Absolute Exposure Time"].value = value

class SceneCameraV4l2(Camera):
    """Scene camera using V4L2 backend"""

    def __init__(self, spec: CameraSpec = NEON_SCENE_CAMERA_V4L2_SPEC) -> None:
        
        super().__init__(spec, V4l2Backend)

        assert isinstance(self.backend, V4l2Backend)
        # Build a dictionary of V4L2 controls by name for easy access
        self.v4l2_controls = {ctrl.name: ctrl for ctrl in self.backend.device.controls}

        # Set default camera parameters similar to UVC version
        default_params = {
            "Brightness": 0,
            "Contrast": 32,
            "Saturation": 64,
            "Hue": 0,
            "Gamma": 300,
            "Gain": 64,
            "Sharpness": 50,
            "Backlight Compensation": 2,
            "Auto Exposure": 1,  # Manual mode (menu index 1)
            "Exposure Time, Absolute": 250,
        }
        for key, value in default_params.items():
            try:
                self._set_v4l2_control(key, value)
            except Exception as e:
                print(f"Setting {key} to {value} failed: {e}")

    def _set_v4l2_control(self, name: str, value: int) -> None:
        """Set a V4L2 control by name, handling both menu and integer controls."""
        if name not in self.v4l2_controls:
            return
        control = self.v4l2_controls[name]
        if isinstance(control, Menu):
            # For menu controls, find the item matching the requested index
            item = next((i for i in control.items if i.index == value), None)
            if item is not None:
                self.backend.device.set_control_value(control, item)
        else:
            self.backend.device.set_control_value(control, value)

    def get_frame(self) -> Frame:
        """Get a frame using software timestamps instead of hardware timestamps."""
        frame = super().get_frame()
        # Replace hardware timestamp with software timestamp to avoid visual artifacts
        return Frame(frame.img, time.time(), frame.index)

    @staticmethod
    def get_intrinsics() -> SceneIntrinsics:
        """Retrieve the scene camera intrinsics of the Neon device

        Returns:
            Tuple containing camera matrix and distortion coefficients of scene camera.

        """
        calib_data = get_calibration()
        return SceneIntrinsics(
            calib_data.scene_camera_matrix,
            calib_data.scene_distortion_coefficients,
            calib_data.scene_extrinsics_affine_matrix,
        )

    @property
    def exposure(self) -> int:
        """Get current exposure time."""
        control_name = "Exposure Time, Absolute"
        if control_name in self.v4l2_controls:
            value = self.backend.device.get_control_value(self.v4l2_controls[control_name])
            assert isinstance(value, int)
            return value
        raise AttributeError("Exposure control not available")

    @exposure.setter
    def exposure(self, value: int) -> None:
        """Set exposure time (manual mode)."""
        # Ensure we're in manual exposure mode (menu index 1 = Manual Mode)
        self._set_v4l2_control("Auto Exposure", 1)
        self._set_v4l2_control("Exposure Time, Absolute", value)