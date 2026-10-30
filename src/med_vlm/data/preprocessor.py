from __future__ import annotations

from dataclasses import dataclass, field

import cv2

# 15/09/2026: workers do datasets.map (fork) morreram com o pool de threads do cv2 herdado;
# CLAHE/denoise sao deterministicos por pixel, o numero de threads nao muda o resultado.
cv2.setNumThreads(int(__import__("os").environ.get("MED_VLM_CV2_THREADS", "1")))
import numpy as np
from PIL import Image as PILImage


@dataclass
class ImagePreprocessor:
    """Static image enhancement applied once at dataset load time (cached by
    HuggingFace ``datasets.map``).

    Pipeline (all steps are individually toggleable):
    1. Noise removal  – applied *before* CLAHE so noise isn't amplified.
    2. CLAHE          – local contrast enhancement.

    For radiology images (grayscale stored as RGB) CLAHE is applied to the
    grayscale channel and the result is replicated back to three channels.
    For colour images (e.g. H&E pathology) CLAHE is applied to the Lightness
    channel in LAB colour-space, preserving hue and saturation.

    Args:
        denoise: Whether to apply noise-removal before CLAHE.
        denoise_method: ``"gaussian"`` (mild, σ=1) or ``"bilateral"``
            (edge-preserving, good for MRI/CT).
        clahe: Whether to apply CLAHE contrast enhancement.
        clahe_clip_limit: Threshold for contrast limiting (default 2.0).
        clahe_tile_grid_size: Grid size for histogram equalisation tiles.
    """

    denoise: bool = False
    denoise_method: str = "gaussian"
    clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = field(default_factory=lambda: (8, 8))

    def __post_init__(self) -> None:
        if self.denoise_method not in ("gaussian", "bilateral"):
            raise ValueError(
                f"denoise_method must be 'gaussian' or 'bilateral', "
                f"got {self.denoise_method!r}"
            )

    def _is_grayscale(self, arr: np.ndarray) -> bool:
        """Return True if all three RGB channels are identical (true grayscale)."""
        return (
            arr.ndim == 3
            and arr.shape[2] == 3
            and np.array_equal(arr[:, :, 0], arr[:, :, 1])
            and np.array_equal(arr[:, :, 0], arr[:, :, 2])
        )

    def _apply_denoise(self, arr: np.ndarray) -> np.ndarray:
        if self.denoise_method == "gaussian":
            return cv2.GaussianBlur(arr, (0, 0), sigmaX=1.0)
        # bilateral: d=-1 lets OpenCV pick based on sigmaColor
        if arr.ndim == 2:
            return cv2.bilateralFilter(arr, d=5, sigmaColor=35, sigmaSpace=35)
        return cv2.bilateralFilter(arr, d=5, sigmaColor=35, sigmaSpace=35)

    def _apply_clahe_gray(self, arr: np.ndarray) -> np.ndarray:
        """Apply CLAHE to a 2-D uint8 grayscale array."""
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )
        return clahe.apply(arr)

    def __call__(self, img: PILImage.Image) -> PILImage.Image:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)

        grayscale = self._is_grayscale(arr)

        if grayscale:
            # Work on the single luminance channel
            gray = arr[:, :, 0]

            if self.denoise:
                gray = self._apply_denoise(gray)

            if self.clahe:
                gray = self._apply_clahe_gray(gray)

            arr = np.stack([gray, gray, gray], axis=2)
        else:
            # Colour image (pathology): work in LAB to preserve stain colours
            if self.denoise:
                arr = self._apply_denoise(arr)

            if self.clahe:
                lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                l_ch = self._apply_clahe_gray(l_ch)
                lab = cv2.merge([l_ch, a_ch, b_ch])
                arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return PILImage.fromarray(arr)

    @classmethod
    def from_config(cls, cfg: dict) -> ImagePreprocessor:
        """Build an ``ImagePreprocessor`` from a config dict (e.g. from YAML).

        Example::

            preprocessing:
              denoise: true
              denoise_method: "gaussian"
              clahe: true
              clahe_clip_limit: 2.0
              clahe_tile_grid_size: [8, 8]
        """
        tile = cfg.get("clahe_tile_grid_size", [8, 8])
        return cls(
            denoise=bool(cfg.get("denoise", False)),
            denoise_method=str(cfg.get("denoise_method", "gaussian")),
            clahe=bool(cfg.get("clahe", False)),
            clahe_clip_limit=float(cfg.get("clahe_clip_limit", 2.0)),
            clahe_tile_grid_size=(int(tile[0]), int(tile[1])),
        )
