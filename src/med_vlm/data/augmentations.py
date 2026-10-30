from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image as PILImage
from PIL import ImageEnhance


@dataclass
class TrainingAugmentor:
    """Online data augmentation applied per-batch during *training only*.

    All transforms are spatially safe for medical VQA: no horizontal or
    vertical flips are included because they would corrupt laterality answers
    (e.g. "Which side is the lesion on?").

    Pipeline (all steps individually toggleable):
    1. Random rotation   – small angle to simulate patient positioning variance.
    2. Random brightness – simulates exposure/illumination differences.
    3. Random contrast   – simulates scanner contrast variance.

    Args:
        rotation_degrees: Max absolute rotation in degrees (±). Default 5.
        brightness_range: (min, max) factor for random brightness. Default
            (0.85, 1.15). Set to (1.0, 1.0) to disable.
        contrast_range: (min, max) factor for random contrast. Default
            (0.85, 1.15). Set to (1.0, 1.0) to disable.
        seed: Optional fixed seed for a private :class:`random.Random` instance
            (useful for reproducible tests). Does **not** affect global state.
    """

    rotation_degrees: float = 5.0
    brightness_range: tuple[float, float] = field(default_factory=lambda: (0.85, 1.15))
    contrast_range: tuple[float, float] = field(default_factory=lambda: (0.85, 1.15))
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def __call__(self, img: PILImage.Image) -> PILImage.Image:
        # Random rotation
        if self.rotation_degrees > 0:
            angle = self._rng.uniform(-self.rotation_degrees, self.rotation_degrees)
            img = img.rotate(angle, resample=PILImage.BILINEAR, expand=False)

        # Random brightness
        lo, hi = self.brightness_range
        if lo != hi or lo != 1.0:
            factor = self._rng.uniform(lo, hi)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Random contrast
        lo, hi = self.contrast_range
        if lo != hi or lo != 1.0:
            factor = self._rng.uniform(lo, hi)
            img = ImageEnhance.Contrast(img).enhance(factor)

        return img

    @classmethod
    def from_config(cls, cfg: dict) -> TrainingAugmentor:
        """Build a ``TrainingAugmentor`` from a config dict (e.g. from YAML).

        Example::

            augmentation:
              enabled: true
              rotation_degrees: 5
              brightness_range: [0.85, 1.15]
              contrast_range: [0.85, 1.15]
        """
        br = cfg.get("brightness_range", [0.85, 1.15])
        cr = cfg.get("contrast_range", [0.85, 1.15])
        return cls(
            rotation_degrees=float(cfg.get("rotation_degrees", 5.0)),
            brightness_range=(float(br[0]), float(br[1])),
            contrast_range=(float(cr[0]), float(cr[1])),
        )


def build_training_augmentor(cfg: dict) -> Callable[[PILImage.Image], PILImage.Image]:
    """Factory: returns an augmentor callable from a config dict, or a no-op if
    ``cfg["enabled"]`` is False or absent."""
    if not cfg.get("enabled", False):
        return lambda img: img
    return TrainingAugmentor.from_config(cfg)
