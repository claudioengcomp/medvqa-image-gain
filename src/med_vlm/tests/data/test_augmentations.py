import numpy as np
from PIL import Image as PILImage

from med_vlm.data.augmentations import TrainingAugmentor, build_training_augmentor


def _rgb_image(size: tuple[int, int] = (64, 64)) -> PILImage.Image:
    return PILImage.new("RGB", size, color=(128, 64, 200))


class TestTrainingAugmentor:
    def test_returns_pil_image(self):
        aug = TrainingAugmentor(seed=0)
        out = aug(_rgb_image())
        assert isinstance(out, PILImage.Image)

    def test_preserves_mode(self):
        aug = TrainingAugmentor(seed=0)
        out = aug(_rgb_image())
        assert out.mode == "RGB"

    def test_preserves_size(self):
        img = _rgb_image((80, 50))
        aug = TrainingAugmentor(rotation_degrees=10, seed=0)
        out = aug(img)
        assert out.size == (80, 50)

    def test_no_rotation_when_zero_degrees(self):
        aug = TrainingAugmentor(
            rotation_degrees=0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            seed=0,
        )
        img = _rgb_image()
        out = aug(img)
        assert np.array_equal(np.array(out), np.array(img))

    def test_deterministic_with_seed(self):
        img = _rgb_image()
        aug1 = TrainingAugmentor(rotation_degrees=5, seed=42)
        aug2 = TrainingAugmentor(rotation_degrees=5, seed=42)
        out1 = aug1(img)
        out2 = aug2(img)
        assert np.array_equal(np.array(out1), np.array(out2))


class TestBuildTrainingAugmentor:
    def test_disabled_returns_identity(self):
        fn = build_training_augmentor({"enabled": False})
        img = _rgb_image()
        assert fn(img) is img

    def test_empty_config_returns_identity(self):
        fn = build_training_augmentor({})
        img = _rgb_image()
        assert fn(img) is img

    def test_enabled_returns_callable(self):
        fn = build_training_augmentor({"enabled": True})
        assert callable(fn)

    def test_enabled_returns_pil_image(self):
        fn = build_training_augmentor(
            {
                "enabled": True,
                "rotation_degrees": 5,
                "brightness_range": [0.9, 1.1],
                "contrast_range": [0.9, 1.1],
            }
        )
        out = fn(_rgb_image())
        assert isinstance(out, PILImage.Image)

    def test_from_config_sets_rotation(self):
        aug = TrainingAugmentor.from_config({"rotation_degrees": 15})
        assert aug.rotation_degrees == 15.0

    def test_from_config_sets_ranges(self):
        aug = TrainingAugmentor.from_config(
            {"brightness_range": [0.7, 1.3], "contrast_range": [0.8, 1.2]}
        )
        assert aug.brightness_range == (0.7, 1.3)
        assert aug.contrast_range == (0.8, 1.2)
