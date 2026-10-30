import math
from unittest.mock import Mock

from med_vlm.utils import (
    adalora_compute_total_steps,
    blip2_vqa_prompt,
    log_metrics,
    medvlm_vqa_prompt,
)


def test_log_metrics_print_only(capsys):
    log_metrics(metrics={"loss": 1.23, "lr": 0.0001}, step=100, print_to_console=True)
    captured = capsys.readouterr()
    assert "[step 100]" in captured.out
    assert "loss=1.2300" in captured.out
    assert "lr=0.0001" in captured.out


def test_log_metrics_with_writer():
    writer = Mock()
    log_metrics(
        metrics={"loss": 1.23, "lr": 0.0001},
        step=100,
        writer=writer,
        print_to_console=False,
    )
    writer.log.assert_called_once_with({"loss": 1.23, "lr": 0.0001}, step=100)


def test_log_metrics_with_prefix():
    writer = Mock()
    log_metrics(
        metrics={"loss": 1.23},
        step=100,
        writer=writer,
        prefix="train",
        print_to_console=False,
    )
    writer.log.assert_called_once_with({"train/loss": 1.23}, step=100)


def test_log_metrics_prefix_trailing_slash():
    writer = Mock()
    log_metrics(
        metrics={"loss": 1.23},
        step=100,
        writer=writer,
        prefix="train/",
        print_to_console=False,
    )
    writer.log.assert_called_once_with({"train/loss": 1.23}, step=100)


def test_adalora_compute_total_steps_matches_formula():
    num_examples, batch_size, grad_accum, epochs = 1000, 32, 4, 3
    expected = math.ceil(math.ceil(num_examples / batch_size) / grad_accum) * epochs
    assert (
        adalora_compute_total_steps(num_examples, batch_size, grad_accum, epochs)
        == expected
    )


def test_blip2_vqa_prompt():
    assert blip2_vqa_prompt("What is this?") == "Question: What is this?\nAnswer:"


def test_medvlm_vqa_prompt_without_answer():
    result = medvlm_vqa_prompt("What is this?")
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 2
    assert result[0]["content"][0]["type"] == "image"
    assert result[0]["content"][1]["type"] == "text"
    assert result[0]["content"][1]["text"] == "What is this?"


def test_medvlm_vqa_prompt_with_answer():
    result = medvlm_vqa_prompt("What is this?", answer="A medical image")
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"][0]["type"] == "text"
    assert result[1]["content"][0]["text"] == "A medical image"
