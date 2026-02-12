from typing import NamedTuple

import numpy as np

from pupil_labs.neon_usb.cameras.backend import UVCBackend
from pupil_labs.neon_usb.cameras.camera import Camera, CameraSpec
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
    bandwidth_factor=0.6,
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
