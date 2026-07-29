"""
Curriculum Learning Controller for N2LN-QEM (TDD §4.2.1).

Manages 3-phase training:
- Phase 1 (0-100): SN-D only
- Phase 2 (100-250): Joint training (SN-D + HN-E)
- Phase 3 (250-300): Fine-tune with consistency loss
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class TrainingPhase(Enum):
    SN_D_ONLY = "sn_d_only"
    JOINT = "joint"
    FINE_TUNE = "fine_tune"


@dataclass
class CurriculumConfig:
    phase1_end: int = 100
    phase2_end: int = 250
    phase3_end: int = 300
    lr_phase1: float = 3e-4
    lr_phase2: float = 3e-4
    lr_phase3: float = 1e-4
    w1_sn_d: float = 1.0
    w2_hn_e: float = 1.0
    w3_consistency: float = 0.3
    w4_physicality: float = 0.1
    bootstrap_warmup_epochs: int = 50
    max_bootstrap_ratio: float = 0.5
    max_consistency_weight: float = 0.3


class CurriculumController:
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self._current_phase: Optional[TrainingPhase] = None

    def get_phase(self, epoch: int) -> TrainingPhase:
        if epoch < self.config.phase1_end:
            return TrainingPhase.SN_D_ONLY
        elif epoch < self.config.phase2_end:
            return TrainingPhase.JOINT
        else:
            return TrainingPhase.FINE_TUNE

    def get_model_mode(self, epoch: int) -> str:
        phase = self.get_phase(epoch)
        if phase == TrainingPhase.SN_D_ONLY:
            return "sn_only"
        elif phase == TrainingPhase.JOINT:
            return "joint"
        else:
            return "fine_tune"

    def get_loss_weights(self, epoch: int) -> Tuple[float, float, float, float]:
        phase = self.get_phase(epoch)
        c = self.config

        if phase == TrainingPhase.SN_D_ONLY:
            return (c.w1_sn_d, 0.0, 0.0, c.w4_physicality)

        elif phase == TrainingPhase.JOINT:
            # Ramp w2 from 0 to w2_hn_e over Phase 2
            phase_start = c.phase1_end
            phase_end = c.phase2_end - 1  # Last epoch of Phase 2
            duration = phase_end - phase_start

            if duration <= 0:
                w2 = c.w2_hn_e
            else:
                clamped_epoch = max(phase_start, min(epoch, phase_end))
                progress = (clamped_epoch - phase_start) / duration
                w2 = progress * c.w2_hn_e
            return (c.w1_sn_d, w2, 0.0, c.w4_physicality)

        else:  # FINE_TUNE
            # Ramp w3 from 0 to max_consistency_weight over Phase 3
            phase_start = c.phase2_end
            phase_end = c.phase3_end - 1  # Last epoch of Phase 3
            duration = phase_end - phase_start

            if duration <= 0:
                w3 = c.max_consistency_weight
            else:
                clamped_epoch = max(phase_start, min(epoch, phase_end))
                progress = (clamped_epoch - phase_start) / duration
                w3 = progress * c.max_consistency_weight
            return (c.w1_sn_d, c.w2_hn_e, w3, c.w4_physicality)

    def get_learning_rate(self, epoch: int) -> float:
        phase = self.get_phase(epoch)
        if phase == TrainingPhase.SN_D_ONLY:
            return self.config.lr_phase1
        elif phase == TrainingPhase.JOINT:
            return self.config.lr_phase2
        else:
            return self.config.lr_phase3

    def get_aug_ratio(self, epoch: int) -> float:
        c = self.config
        if epoch < c.bootstrap_warmup_epochs:
            return 0.0
        progress = min(1.0, (epoch - c.bootstrap_warmup_epochs) / max(c.bootstrap_warmup_epochs, 1))
        return progress * c.max_bootstrap_ratio

    def is_phase_transition(self, epoch: int) -> bool:
        if epoch == 0:
            return True
        return self.get_phase(epoch) != self.get_phase(epoch - 1)

    def should_apply_consistency_loss(self, epoch: int) -> bool:
        return self.get_phase(epoch) == TrainingPhase.FINE_TUNE

    def get_total_epochs(self) -> int:
        return self.config.phase3_end

    def get_phase_description(self, epoch: int) -> str:
        phase = self.get_phase(epoch)
        descriptions = {
            TrainingPhase.SN_D_ONLY: "Phase 1: Training SN-D head only (shot noise denoising)",
            TrainingPhase.JOINT: "Phase 2: Joint training with HN-E head (hardware noise mitigation)",
            TrainingPhase.FINE_TUNE: "Phase 3: Fine-tuning with consistency loss",
        }
        return descriptions[phase]
