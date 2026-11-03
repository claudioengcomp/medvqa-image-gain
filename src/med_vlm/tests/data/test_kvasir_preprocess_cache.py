from med_vlm.data.loader import _kvasir_preprocess_cache_path


class _FakePreprocessor:
    """Stable repr simulating ImagePreprocessor (dataclass) without needing
    the cv2/numpy weights in this pure test."""

    def __init__(self, denoise=True, clahe=True):
        self.denoise = denoise
        self.clahe = clahe

    def __repr__(self):
        return f"_FakePreprocessor(denoise={self.denoise}, clahe={self.clahe})"


def test_returns_none_when_preprocessor_is_none(tmp_path):
    arrow = tmp_path / "vqa_kvasir_train_clean.arrow"
    arrow.write_bytes(b"fake")
    assert _kvasir_preprocess_cache_path(str(arrow), None) is None


def test_returns_none_when_arrow_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.arrow"
    assert _kvasir_preprocess_cache_path(str(missing), _FakePreprocessor()) is None


def test_same_inputs_produce_same_path(tmp_path):
    arrow = tmp_path / "vqa_kvasir_train_clean.arrow"
    arrow.write_bytes(b"fake")
    pre = _FakePreprocessor()
    path_a = _kvasir_preprocess_cache_path(str(arrow), pre)
    path_b = _kvasir_preprocess_cache_path(str(arrow), pre)
    assert path_a == path_b
    assert path_a is not None


def test_different_preprocessor_config_produces_different_path(tmp_path):
    arrow = tmp_path / "vqa_kvasir_train_clean.arrow"
    arrow.write_bytes(b"fake")
    path_denoise_on = _kvasir_preprocess_cache_path(str(arrow), _FakePreprocessor(denoise=True))
    path_denoise_off = _kvasir_preprocess_cache_path(str(arrow), _FakePreprocessor(denoise=False))
    assert path_denoise_on != path_denoise_off


def test_different_arrow_mtime_produces_different_path(tmp_path):
    arrow = tmp_path / "vqa_kvasir_train_clean.arrow"
    arrow.write_bytes(b"fake")
    pre = _FakePreprocessor()
    path_before = _kvasir_preprocess_cache_path(str(arrow), pre)

    import os
    import time
    time.sleep(0.01)
    os.utime(arrow, None)  # sets mtime to "now"

    path_after = _kvasir_preprocess_cache_path(str(arrow), pre)
    assert path_before != path_after


def test_path_lives_under_preprocessed_cache_dir(tmp_path):
    arrow = tmp_path / "vqa_kvasir_val_clean.arrow"
    arrow.write_bytes(b"fake")
    path = _kvasir_preprocess_cache_path(str(arrow), _FakePreprocessor())
    assert "preprocessed_cache" in path
    assert path.endswith(".arrow")
