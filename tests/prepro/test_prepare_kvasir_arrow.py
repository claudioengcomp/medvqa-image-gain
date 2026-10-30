import json
import urllib.error
from unittest.mock import patch

import pytest

from harness.prepro.prepare_kvasir_arrow import (
    _download_image_bytes,
    _placeholder_image_bytes,
    build_arrow_dataframe,
    build_image_records,
    classify_question,
    decompose_original_field,
)


def test_decompose_original_field_parses_json_list():
    original = json.dumps([
        {"q": "Are there any abnormalities in the image? Check all that are present.", "a": "polyp"},
        {"q": "How many polyps are in the image?", "a": "1"},
    ])
    items = decompose_original_field(original)
    assert items == [
        {"question": "Are there any abnormalities in the image? Check all that are present.", "answer": "polyp"},
        {"question": "How many polyps are in the image?", "answer": "1"},
    ]


def test_decompose_original_field_empty_or_malformed_returns_empty_list():
    assert decompose_original_field("") == []
    assert decompose_original_field("not json") == []


def test_classify_question_covers_all_known_templates():
    closed_examples = [
        "Is there text?",
        "How many instrumnets are in the image?",
        "What type of polyp is present?",
        "Are there any abnormalities in the image? Check all that are present.",
        "What is the size of the polyp?",
        "Where in the image is the abnormality?",
        "Have all polyps been removed?",
        "What color is the abnormality?",
        # achados só no split de treino, em execução real (21-22/08/2026) --
        # ver docstring do módulo.
        "Does this image contain any finding?",
        "What color is the anatomical landmark? If more than one separate with ;",
    ]
    for q in closed_examples:
        assert classify_question(q) == "closed", q


def test_classify_question_unknown_template_raises():
    with pytest.raises(ValueError):
        classify_question("some question never seen in the known templates")


def test_build_image_records_captures_pathology_label_separately():
    rows = [{
        "img_id": "img1",
        "image": "http://example.com/img1.jpg",
        "original": json.dumps([
            {"q": "Are there any abnormalities in the image? Check all that are present.", "a": "polyp"},
            {"q": "Is there text?", "a": "no"},
        ]),
    }]
    records = build_image_records(rows)
    assert len(records) == 1
    assert records[0]["pathology_label"] == "polyp"
    assert "Is there text?" in records[0]["questions"]


def test_build_image_records_pathology_label_none_when_question_absent():
    rows = [{
        "img_id": "img2",
        "image": "http://example.com/img2.jpg",
        "original": json.dumps([{"q": "Is there text?", "a": "no"}]),
    }]
    records = build_image_records(rows)
    assert records[0]["pathology_label"] is None


def test_build_image_records_deduplicates_repeated_atomic_questions_across_rows():
    # a mesma imagem aparece em varias linhas do x1 (perguntas compostas
    # diferentes), cada uma decompoe algumas das mesmas perguntas atomicas
    rows = [
        {"img_id": "img1", "image": "http://x/1.jpg",
         "original": json.dumps([{"q": "Is there text?", "a": "no"}, {"q": "How many polyps are in the image?", "a": "1"}])},
        {"img_id": "img1", "image": "http://x/1.jpg",
         "original": json.dumps([{"q": "Is there text?", "a": "no"}])},
    ]
    records = build_image_records(rows)
    assert len(records) == 1
    assert records[0]["questions"].count("Is there text?") == 1


def test_build_image_records_skips_images_with_no_recognized_closed_question():
    rows = [{"img_id": "img1", "image": "http://x/1.jpg", "original": json.dumps([])}]
    assert build_image_records(rows) == []


def test_build_arrow_dataframe_produces_expected_schema_columns():
    records = [{
        "img_id": "img1", "image_url": "http://x/1.jpg",
        "questions": ["Is there text?", "How many polyps are in the image?"],
        "answers": ["no", "1"],
        "pathology_label": "polyp",
    }]

    def fake_download(_url: str) -> bytes:
        return b"FAKEBYTES"

    df = build_arrow_dataframe(records, split="test", qid_offset=100, download_fn=fake_download)
    assert list(df.columns) == [
        "image", "questions", "answers", "answer_labels", "answer_scores",
        "image_id", "question_id", "answer_type", "split",
    ]
    assert df.iloc[0]["image_id"] == "kvasir/img1"
    assert df.iloc[0]["question_id"] == [100, 101]
    assert df.iloc[0]["answers"] == [["no"], ["1"]]
    assert df.iloc[0]["answer_type"] == [0, 0]
    assert df.iloc[0]["split"] == "test"


def test_download_image_bytes_retries_on_transient_error_then_succeeds():
    # Achado real (Colab, 21/08/2026): ConnectionResetError no meio de
    # ~4.000 downloads sequenciais derrubava o script inteiro sem essa
    # retentativa.
    mock_resp = type("Resp", (), {
        "__enter__": lambda self: self, "__exit__": lambda self, *a: None,
        "read": lambda self: b"FAKEBYTES",
    })()
    calls = {"n": 0}

    def flaky_urlopen(url, timeout=30):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("connection reset by peer")
        return mock_resp

    with patch("harness.prepro.prepare_kvasir_arrow.urllib.request.urlopen", side_effect=flaky_urlopen), \
         patch("harness.prepro.prepare_kvasir_arrow.time.sleep"):
        result = _download_image_bytes("http://example.com/img.jpg")

    assert result == b"FAKEBYTES"
    assert calls["n"] == 3


def test_download_image_bytes_raises_after_exhausting_retries():
    def always_fails(url, timeout=30):
        raise urllib.error.URLError("connection reset by peer")

    with patch("harness.prepro.prepare_kvasir_arrow.urllib.request.urlopen", side_effect=always_fails), \
         patch("harness.prepro.prepare_kvasir_arrow.time.sleep"):
        with pytest.raises(urllib.error.URLError):
            _download_image_bytes("http://example.com/img.jpg", max_retries=3)


def test_placeholder_image_bytes_ignores_url_and_returns_valid_jpeg():
    from PIL import Image
    import io

    b1 = _placeholder_image_bytes("http://example.com/a.jpg")
    b2 = _placeholder_image_bytes("http://example.com/completely-different-url.jpg")
    assert b1 == b2  # mesmos bytes sempre, independente da url

    img = Image.open(io.BytesIO(b1))
    assert img.format == "JPEG"
    assert img.size == (1, 1)


def test_build_arrow_dataframe_with_skip_images_never_calls_real_downloader():
    from harness.prepro.prepare_kvasir_arrow import _placeholder_image_bytes

    records = [{
        "img_id": "img1", "image_url": "http://example.com/1.jpg",
        "questions": ["Is there text?"], "answers": ["no"], "pathology_label": None,
    }]
    df = build_arrow_dataframe(records, split="train", download_fn=_placeholder_image_bytes)
    assert df.iloc[0]["image"] == _placeholder_image_bytes("qualquer-coisa")
