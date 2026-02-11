from typing import NamedTuple

import numpy as np

from pupil_labs.neon_usb.cameras.backend import UVCBackend, V4l2Backend
from pupil_labs.neon_usb.cameras.camera import Camera, CameraSpec
from pupil_labs.neon_usb.usb_utils import get_calibration
from pupil_labs.neon_usb.pyrav4l2.controls import Menu, MenuItem, IntegerMenuItem, Control


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


class SceneCamera(Camera):
    """Provides an interface for handling the Neon scene camera.

    The class is assuming that no more than one Neon device is connected to the
    computer at the same time.
    """

    def __init__(
        self,
        spec: CameraSpec = NEON_SCENE_CAMERA_SPEC,
        backend_class: type[UVCBackend | V4l2Backend] = UVCBackend,
    ) -> None:
        """Initialize the scene camera of the connected Neon device.

        The camera stream will be started right away. If the object fails to grab
        frames, it will automatically try to reinitialize.
        """
        super().__init__(spec, backend_class)

        self.uvc_controls = None
        if isinstance(self.backend, UVCBackend):
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

        self._v4l2_exposure_control: Control | None = None
        self._v4l2_auto_exposure_control: Control | None = None
        if isinstance(self.backend, V4l2Backend):
            self._v4l2_exposure_control = self._find_v4l2_control(
                (
                    "Exposure (Absolute)",
                    "Exposure, Absolute",
                    "Exposure Absolute",
                    "exposure_time_absolute",
                )
            )
            self._v4l2_auto_exposure_control = self._find_v4l2_control(
                ("Exposure, Auto", "Exposure Auto", "auto_exposure")
            )
            self._configure_v4l2_exposure_defaults()

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
        if isinstance(self.backend, UVCBackend):
            assert self.uvc_controls is not None
            value = self.uvc_controls["Absolute Exposure Time"].value
            assert isinstance(value, int)
            return value
        if isinstance(self.backend, V4l2Backend):
            if self._v4l2_exposure_control is None:
                raise RuntimeError("V4L2 exposure control not available.")
            value = self.backend.device.get_control_value(self._v4l2_exposure_control)
            assert isinstance(value, int)
            return value
        raise RuntimeError("Unsupported backend for exposure control.")

    @exposure.setter
    def exposure(self, value: int) -> None:
        if isinstance(self.backend, UVCBackend):
            assert self.uvc_controls is not None
            self.uvc_controls["Absolute Exposure Time"].value = value
            return
        if isinstance(self.backend, V4l2Backend):
            if self._v4l2_exposure_control is None:
                raise RuntimeError("V4L2 exposure control not available.")
            self.backend.device.set_control_value(self._v4l2_exposure_control, value)
            return
        raise RuntimeError("Unsupported backend for exposure control.")

    def _find_v4l2_control(self, names: tuple[str, ...]) -> Control | None:
        if not isinstance(self.backend, V4l2Backend):
            return None
        def _norm(value: str) -> str:
            return value.casefold().replace(" ", "_").replace(",", "_")

        names_cf = {_norm(name) for name in names}
        for ctrl in self.backend.device.controls:
            if _norm(ctrl.name) in names_cf:
                return ctrl
        for ctrl in self.backend.device.controls:
            for name in names_cf:
                if name in _norm(ctrl.name):
                    return ctrl
        return None

    def _configure_v4l2_exposure_defaults(self) -> None:
        if not isinstance(self.backend, V4l2Backend):
            return
        if self._v4l2_auto_exposure_control is not None:
            auto_ctrl = self._v4l2_auto_exposure_control
            if isinstance(auto_ctrl, Menu):
                self._set_v4l2_menu_value(auto_ctrl, ("manual",))
            else:
                try:
                    self.backend.device.set_control_value(auto_ctrl, 1)
                except Exception:
                    pass
        if self._v4l2_exposure_control is not None:
            try:
                self.backend.device.set_control_value(self._v4l2_exposure_control, 250)
            except Exception:
                pass

    def _set_v4l2_menu_value(self, control: Menu, keywords: tuple[str, ...]) -> None:
        keywords_cf = tuple(k.casefold() for k in keywords)
        for item in control.items:
            name = ""
            if isinstance(item, MenuItem):
                name = item.name
            elif isinstance(item, IntegerMenuItem):
                name = str(item.value)
            if any(k in name.casefold() for k in keywords_cf):
                self.backend.device.set_control_value(control, item)
                return


class SceneCameraUVC(SceneCamera):
    def __init__(self, spec: CameraSpec = NEON_SCENE_CAMERA_SPEC) -> None:
        super().__init__(spec, UVCBackend)


class SceneCameraV4l2(SceneCamera):
    def __init__(self, spec: CameraSpec = NEON_SCENE_CAMERA_SPEC) -> None:
        super().__init__(spec, V4l2Backend)
