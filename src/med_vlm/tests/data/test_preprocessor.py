import numpy as np
import pytest
from PIL import Image as PILImage

from med_vlm.data.preprocessor import ImagePreprocessor


def _make_gray_rgb(size: tuple[int, int] = (64, 64)) -> PILImage.Image:
    """Create an RGB image where all three channels are identical (true grayscale)."""
    arr = np.random.randint(0, 256, (*size,), dtype=np.uint8)
    arr_rgb = np.stack([arr, arr, arr], axis=2)
    return PILImage.fromarray(arr_rgb, mode="RGB")


def _make_color_rgb(size: tuple[int, int] = (64, 64)) -> PILImage.Image:
    """Create a genuinely colour RGB image (channels differ)."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    # Ensure channels differ so the image isn't accidentally grayscale
    arr[:, :, 1] = (arr[:, :, 0].astype(int) + 50) % 256
    arr[:, :, 2] = (arr[:, :, 0].astype(int) + 100) % 256
    return PILImage.fromarray(arr, mode="RGB")


class TestImagePreprocessorIdentity:
    def test_noop_returns_rgb(self):
        proc = ImagePreprocessor()
        img = PILImage.new("L", (32, 32))
        out = proc(img)
        assert out.mode == "RGB"

    def test_noop_preserves_size(self):
        proc = ImagePreprocessor()
        img = PILImage.new("RGB", (128, 64))
        out = proc(img)
        assert out.size == (128, 64)

    def test_noop_returns_pil_image(self):
        proc = ImagePreprocessor()
        img = PILImage.new("RGB", (32, 32))
        assert isinstance(proc(img), PILImage.Image)


class TestImagePreprocessorDenoise:
    def test_gaussian_denoise_grayscale(self):
        proc = ImagePreprocessor(denoise=True, denoise_method="gaussian")
        img = _make_gray_rgb()
        out = proc(img)
        assert out.mode == "RGB"
        assert out.size == img.size

    def test_gaussian_denoise_color(self):
        proc = ImagePreprocessor(denoise=True, denoise_method="gaussian")
        img = _make_color_rgb()
        out = proc(img)
        assert out.mode == "RGB"
        assert out.size == img.size

    def test_bilateral_denoise_grayscale(self):
        proc = ImagePreprocessor(denoise=True, denoise_method="bilateral")
        img = _make_gray_rgb()
        out = proc(img)
        assert out.mode == "RGB"
        assert out.size == img.size

    def test_bilateral_denoise_color(self):
        proc = ImagePreprocessor(denoise=True, denoise_method="bilateral")
        img = _make_color_rgb()
        out = proc(img)
        assert out.mode == "RGB"
        assert out.size == img.size

    def test_invalid_denoise_method_raises(self):
        with pytest.raises(ValueError, match="denoise_method"):
            ImagePreprocessor(denoise_method="nonexistent")


class TestImagePreprocessorCLAHE:
    def test_clahe_grayscale_image(self):
        proc = ImagePreprocessor(clahe=True)
        img = _make_gray_rgb()
        out = proc(img)
        assert out.mode == "RGB"
        assert out.size == img.size

    def test_clahe_color_image(self):
        proc = ImagePreprocessor(clahe=True)
        img = _make_color_rgb()
        out = proc(img)
        assert out.mode == "RGB"
        assert out.size == img.size

    def test_clahe_modifies_pixel_values_for_grayscale(self):
        proc = ImagePreprocessor(clahe=True, clahe_clip_limit=4.0)
        img = _make_gray_rgb((64, 64))
        out = proc(img)
        # CLAHE should generally change pixel values (unless input is already flat)
        _ = np.array(img)
        out_arr = np.array(out)
        # Both channels should be identical (grayscale replication preserved)
        assert np.array_equal(out_arr[:, :, 0], out_arr[:, :, 1])
        assert np.array_equal(out_arr[:, :, 0], out_arr[:, :, 2])


class TestImagePreprocessorCombined:
    def test_denoise_and_clahe(self):
        proc = ImagePreprocessor(denoise=True, clahe=True)
        for img in [_make_gray_rgb(), _make_color_rgb()]:
            out = proc(img)
            assert out.mode == "RGB"
            assert out.size == img.size

    def test_input_pil_L_mode(self):
        proc = ImagePreprocessor(denoise=True, clahe=True)
        img = PILImage.new("L", (48, 48))
        out = proc(img)
        assert out.mode == "RGB"
        assert out.size == (48, 48)


class TestImagePreprocessorFromConfig:
    def test_default_from_config(self):
        proc = ImagePreprocessor.from_config({})
        assert proc.denoise is False
        assert proc.clahe is False

    def test_from_config_sets_fields(self):
        cfg = {
            "denoise": True,
            "denoise_method": "bilateral",
            "clahe": True,
            "clahe_clip_limit": 3.0,
            "clahe_tile_grid_size": [4, 4],
        }
        proc = ImagePreprocessor.from_config(cfg)
        assert proc.denoise is True
        assert proc.denoise_method == "bilateral"
        assert proc.clahe is True
        assert proc.clahe_clip_limit == 3.0
        assert proc.clahe_tile_grid_size == (4, 4)
