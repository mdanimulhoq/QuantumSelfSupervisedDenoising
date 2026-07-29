"""Unit tests for curriculum learning controller (TDD §4.2.1)."""

import pytest
from src.training.curriculum import CurriculumController, CurriculumConfig, TrainingPhase


def test_phase_boundaries():
    config = CurriculumConfig()
    controller = CurriculumController(config)

    assert controller.get_phase(0) == TrainingPhase.SN_D_ONLY
    assert controller.get_phase(50) == TrainingPhase.SN_D_ONLY
    assert controller.get_phase(99) == TrainingPhase.SN_D_ONLY
    assert controller.get_phase(100) == TrainingPhase.JOINT
    assert controller.get_phase(150) == TrainingPhase.JOINT
    assert controller.get_phase(249) == TrainingPhase.JOINT
    assert controller.get_phase(250) == TrainingPhase.FINE_TUNE
    assert controller.get_phase(299) == TrainingPhase.FINE_TUNE
    assert controller.get_phase(300) == TrainingPhase.FINE_TUNE


def test_model_mode():
    config = CurriculumConfig()
    controller = CurriculumController(config)
    assert controller.get_model_mode(50) == "sn_only"
    assert controller.get_model_mode(150) == "joint"
    assert controller.get_model_mode(260) == "fine_tune"


def test_loss_weights_phase1():
    config = CurriculumConfig()
    controller = CurriculumController(config)
    w1, w2, w3, w4 = controller.get_loss_weights(50)
    assert w1 == 1.0
    assert w2 == 0.0
    assert w3 == 0.0
    assert w4 == 0.1


def test_loss_weights_phase2_ramp():
    config = CurriculumConfig()
    controller = CurriculumController(config)

    # At start of Phase 2 (epoch 100), w2 should be 0 (just starting to ramp)
    w1, w2, w3, w4 = controller.get_loss_weights(100)
    assert w2 == 0.0

    # Mid Phase 2 (epoch 175), w2 should be ~0.5
    w1, w2, w3, w4 = controller.get_loss_weights(175)
    assert 0.4 < w2 < 0.6

    # End of Phase 2 (epoch 249), w2 should be ~1.0 (allow floating point tolerance)
    w1, w2, w3, w4 = controller.get_loss_weights(249)
    assert abs(w2 - 1.0) < 1e-6


def test_loss_weights_phase3_ramp():
    config = CurriculumConfig(max_consistency_weight=0.3)
    controller = CurriculumController(config)

    # At start of Phase 3 (epoch 250), w3 should be 0
    w1, w2, w3, w4 = controller.get_loss_weights(250)
    assert w3 == 0.0

    # Mid Phase 3 (epoch 275), w3 should be ~0.15
    w1, w2, w3, w4 = controller.get_loss_weights(275)
    assert 0.1 < w3 < 0.2

    # End of Phase 3 (epoch 299), w3 should be ~0.3 (allow floating point tolerance)
    w1, w2, w3, w4 = controller.get_loss_weights(299)
    assert abs(w3 - 0.3) < 1e-6


def test_learning_rate():
    config = CurriculumConfig(lr_phase1=3e-4, lr_phase2=3e-4, lr_phase3=1e-4)
    controller = CurriculumController(config)
    assert controller.get_learning_rate(50) == 3e-4
    assert controller.get_learning_rate(150) == 3e-4
    assert controller.get_learning_rate(260) == 1e-4


def test_aug_ratio_ramp():
    config = CurriculumConfig(bootstrap_warmup_epochs=50, max_bootstrap_ratio=0.5)
    controller = CurriculumController(config)
    assert controller.get_aug_ratio(0) == 0.0
    assert controller.get_aug_ratio(49) == 0.0
    assert 0.0 < controller.get_aug_ratio(75) < 0.5
    assert controller.get_aug_ratio(100) == 0.5


def test_phase_transition_detection():
    config = CurriculumConfig()
    controller = CurriculumController(config)
    assert controller.is_phase_transition(0) is True
    assert controller.is_phase_transition(100) is True
    assert controller.is_phase_transition(250) is True
    assert controller.is_phase_transition(150) is False


def test_consistency_loss_applied_only_phase3():
    config = CurriculumConfig()
    controller = CurriculumController(config)
    assert controller.should_apply_consistency_loss(50) is False
    assert controller.should_apply_consistency_loss(150) is False
    assert controller.should_apply_consistency_loss(260) is True


def test_total_epochs():
    config = CurriculumConfig(phase3_end=350)
    controller = CurriculumController(config)
    assert controller.get_total_epochs() == 350
