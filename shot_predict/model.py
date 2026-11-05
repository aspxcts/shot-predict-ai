from dataclasses import asdict, dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .court import ZONES


@dataclass
class ModelConfig:
    image_size: int = 224
    samples: int = 8
    seconds: float = 0.6
    hidden: int = 128
    freeze_backbone: bool = False
    input_mode: str = "full_frame"
    detector_sha256: str = ""

    def validate(self):
        if self.image_size < 32 or self.samples < 2 or self.seconds <= 0 or self.hidden < 1:
            raise ValueError("Image size >= 32, samples >= 2, and positive window/hidden size required.")
        if self.input_mode not in ("full_frame", "player_crop"):
            raise ValueError("Input mode must be full_frame or player_crop.")
        if self.input_mode == "player_crop" and (
            len(self.detector_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.detector_sha256)
        ):
            raise ValueError("Player-crop models require the detector's SHA-256.")

    def to_dict(self):
        return asdict(self)


class ShotModel(nn.Module):
    def __init__(self, config: ModelConfig, pretrained: bool = False):
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
        self.temporal = nn.GRU(512, config.hidden, batch_first=True)
        self.head = nn.Linear(config.hidden, len(ZONES))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def encode(self, clips: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = clips.shape
        features = self.backbone(clips.reshape(batch * time, channels, height, width))
        return features.reshape(batch, time, -1)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        _, hidden = self.temporal(features)
        return self.head(hidden[-1])

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(clips))
