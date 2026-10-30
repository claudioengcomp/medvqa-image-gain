from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
import torch

from med_vlm.training.trainer import (
    TrainConfig,
    create_accelerator,
    create_optimizer,
    log_train_metrics,
    prepare_training,
    runningAverage,
    should_log,
    train_step,
    train_vqa_generation,
)


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.lr == 1e-4
    assert cfg.epochs == 1
    assert cfg.grad_accum_steps == 16
    assert cfg.max_grad_norm == 1.0
    assert cfg.log_every == 10


def test_train_config_custom_values():
    cfg = TrainConfig(
        lr=5e-5, epochs=3, grad_accum_steps=8, max_grad_norm=2.0, log_every=5
    )
    assert cfg.lr == 5e-5
    assert cfg.epochs == 3
    assert cfg.grad_accum_steps == 8
    assert cfg.max_grad_norm == 2.0
    assert cfg.log_every == 5


def test_train_config_frozen():
    cfg = TrainConfig()
    with pytest.raises(AttributeError):
        cfg.lr = 1e-3


def test_train_config_eval_every_zero_raises():
    with pytest.raises(ValueError, match="eval_every must be a positive integer"):
        TrainConfig(eval_every=0)


def test_train_config_eval_every_negative_raises():
    with pytest.raises(ValueError, match="eval_every must be a positive integer"):
        TrainConfig(eval_every=-5)


def test_train_config_eval_every_valid():
    cfg = TrainConfig(eval_every=100)
    assert cfg.eval_every == 100


def test_running_average_initialization():
    avg = runningAverage()
    assert avg.running_loss == 0.0
    assert avg.running_count == 0


def test_running_average_update():
    avg = runningAverage()
    avg.update(1.5)
    avg.update(2.5)
    assert avg.running_loss == 4.0
    assert avg.running_count == 2


def test_running_average_single_value():
    avg = runningAverage()
    avg.update(5.0)
    assert avg.average() == 5.0


def test_running_average_multiple_values():
    avg = runningAverage()
    avg.update(2.0)
    avg.update(4.0)
    avg.update(6.0)
    assert avg.average() == 4.0


def test_running_average_empty():
    avg = runningAverage()
    assert avg.average() == 0.0


def test_running_average_reset():
    avg = runningAverage()
    avg.update(5.0)
    avg.reset()
    assert avg.running_loss == 0.0
    assert avg.running_count == 0
    assert avg.average() == 0.0


def test_should_log_true():
    assert should_log(10, 10) is True
    assert should_log(20, 10) is True


def test_should_log_false():
    assert should_log(5, 10) is False
    assert should_log(15, 10) is False


def test_should_log_first_step():
    assert should_log(1, 10) is False


def test_log_train_metrics_not_main_process():
    accelerator = Mock()
    accelerator.is_main_process = False
    writer = Mock()

    result = log_train_metrics(
        accelerator=accelerator,
        writer=writer,
        step=1,
        loss=0.5,
        lr=1e-4,
    )

    assert result is None


@patch("med_vlm.training.trainer.log_metrics")
def test_log_train_metrics_main_process(mock_log_metrics):
    accelerator = Mock()
    accelerator.is_main_process = True
    writer = Mock()

    log_train_metrics(
        accelerator=accelerator,
        writer=writer,
        step=1,
        loss=0.5,
        lr=1e-4,
    )

    mock_log_metrics.assert_called_once()
    call_kwargs = mock_log_metrics.call_args[1]
    assert call_kwargs["metrics"] == {"loss": 0.5, "lr": 1e-4}
    assert call_kwargs["step"] == 1
    assert call_kwargs["writer"] == writer
    assert call_kwargs["prefix"] == "train"


@patch("med_vlm.training.trainer.Accelerator")
def test_create_accelerator_calls_accelerate(mock_acc):
    cfg = TrainConfig(grad_accum_steps=8)
    create_accelerator(cfg)
    mock_acc.assert_called_once_with(
        gradient_accumulation_steps=8,
        mixed_precision="bf16",
    )


def test_create_optimizer_default():
    model = Mock()
    model.parameters = Mock(return_value=[torch.nn.Parameter(torch.zeros(1))])
    cfg = TrainConfig(lr=1e-4)

    optimizer = create_optimizer(model, cfg, None)

    assert optimizer is not None
    assert optimizer.param_groups[0]["lr"] == cfg.lr


def test_create_optimizer_provided():
    model = Mock()
    cfg = TrainConfig()
    provided_optimizer = Mock()

    optimizer = create_optimizer(model, cfg, provided_optimizer)

    assert optimizer == provided_optimizer


def test_create_optimizer_uses_config_lr():
    model = Mock()
    model.parameters = Mock(return_value=[torch.nn.Parameter(torch.zeros(1))])
    cfg = TrainConfig(lr=5e-5)

    optimizer = create_optimizer(model, cfg, None)

    assert optimizer.param_groups[0]["lr"] == cfg.lr


def test_prepare_training_calls_prepare():
    accelerator = Mock()
    accelerator.prepare.return_value = ("m", "o", "dl")

    model, optimizer, train_dl = Mock(), Mock(), Mock()
    out = prepare_training(accelerator, model, optimizer, train_dl)

    accelerator.prepare.assert_called_once_with(model, optimizer, train_dl)
    assert out == ("m", "o", "dl")


def test_train_step_calls_optimizer_and_returns_loss():
    accelerator = Mock()
    accelerator.accumulate.return_value = nullcontext()
    accelerator.backward = Mock()
    accelerator.clip_grad_norm_ = Mock()

    model = Mock()
    model.parameters.return_value = []
    model.return_value = Mock(loss=torch.tensor(0.5))

    optimizer = Mock()
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}

    loss = train_step(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        batch=batch,
        max_grad_norm=1.0,
    )

    assert loss == 0.5
    accelerator.backward.assert_called_once()
    accelerator.clip_grad_norm_.assert_called_once()
    optimizer.step.assert_called_once()
    optimizer.zero_grad.assert_called_once()


def test_train_vqa_generation_eval_every_without_deps_raises():
    """eval_every set but
    eval_dl/processor/spec missing → ValueError before training."""
    cfg = TrainConfig(eval_every=10)
    with pytest.raises(ValueError, match="eval_every is set but"):
        train_vqa_generation(
            model=Mock(),
            train_dl=[],
            cfg=cfg,
            writer=None,
            # eval_dl, processor, spec intentionally omitted
        )


@patch(
    "med_vlm.training.trainer.torch.compile",
    side_effect=lambda model, **kwargs: model,
)
@patch("med_vlm.training.trainer.train_step")
@patch("med_vlm.training.trainer.prepare_training")
@patch("med_vlm.training.trainer.create_scheduler")
@patch("med_vlm.training.trainer.create_optimizer")
@patch("med_vlm.training.trainer.create_accelerator")
def test_train_vqa_generation_stops_at_max_steps(
    mock_create_accelerator,
    mock_create_optimizer,
    mock_create_scheduler,
    mock_prepare_training,
    mock_train_step,
    mock_compile,
):
    """max_steps deve encerrar o treino imediatamente ao ser atingido, mesmo
    quando epochs * len(train_dl) produziria naturalmente muito mais passos."""
    accelerator = Mock()
    accelerator.is_main_process = True
    mock_create_accelerator.return_value = accelerator

    train_dl = [{"input_ids": torch.tensor([[1, 2, 3]])} for _ in range(10)]

    model = Mock()
    optimizer = Mock()
    optimizer.param_groups = [{"lr": 1e-4}]
    mock_create_optimizer.return_value = optimizer
    mock_create_scheduler.return_value = Mock()
    mock_prepare_training.return_value = (model, optimizer, train_dl)
    mock_train_step.return_value = 0.5

    cfg = TrainConfig(epochs=3, max_steps=5, grad_accum_steps=1, log_every=1)

    train_vqa_generation(
        model=model,
        train_dl=train_dl,
        cfg=cfg,
        writer=None,
    )

    # epochs=3 * len(train_dl)=10 => 30 passos "naturais", mas max_steps=5
    # deve encerrar o treino exatamente em 5, cortando as duas malhas
    # (epoch e batch).
    assert mock_train_step.call_count == 5
