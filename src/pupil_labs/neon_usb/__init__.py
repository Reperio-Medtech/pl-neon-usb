"""pupil_labs.neon_usb package.

Library for connecting to Neon via USB
"""

from __future__ import annotations

import importlib.metadata

from pupil_labs.neon_usb.cameras.camera import CameraNotFoundError
from pupil_labs.neon_usb.cameras.eye import EyeCameraUVC, EyeCameraV4l2
from pupil_labs.neon_usb.cameras.scene import SceneCamera
from pupil_labs.neon_usb.device import Device
from pupil_labs.neon_usb.frame import Frame
from pupil_labs.neon_usb.queue_utils import get_all_items, image_receiver

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__: list[str] = [
    "CameraNotFoundError",
    "Device",
    "EyeCameraUVC",
    "EyeCameraV4l2",
    "Frame",
    "SceneCamera",
    "__version__",
    "get_all_items",
    "image_receiver",
]
