"""The retained FiLM head, usable with either current frames or causal video."""

from pathlib import Path
import torch
from torch import nn
from .cvae import SpatialCVAE
from .components import SpatialConditioner, FiLMSpatialDecoder
from .encoders import FrozenVJEPA2Encoder, VJEPA2_SHA256


class FiLMCVAE(SpatialCVAE):
    """Non-temporal ablation: repeat only the current frame for JEPA's tubelet."""

    supports_features = True
    image_size = 256
    feature_shape = (2048, 8, 10)

    def __init__(
        self,
        hidden_dim=256,
        context_len=1,
        frame_step=1,
        encoder_repo="external/vjepa2",
        encoder_checkpoint="pretrained/vjepa2/vitl.pt",
        encoder_sha256=VJEPA2_SHA256,
        encoder_batch_size=2,
        **kwargs,
    ):
        kwargs.setdefault("optimizer", "adamw")
        kwargs.setdefault("lr", 0.0001)
        kwargs.setdefault("weight_decay", 0.000001)
        kwargs.setdefault("lr_scheduler_patience", 3)
        super().__init__(**kwargs)
        self.save_hyperparameters(
            dict(
                hidden_dim=hidden_dim,
                context_len=context_len,
                frame_step=frame_step,
                encoder_repo=str(encoder_repo),
                encoder_checkpoint=str(encoder_checkpoint),
                encoder_sha256=encoder_sha256,
                encoder_batch_size=encoder_batch_size,
            )
        )
        if self.grid_hw != (32, 40):
            raise ValueError("FiLM decoder produces a 32x40 gaze grid")
        if (not self.temporal and context_len != 1) or (
            self.temporal and (context_len < 2 or context_len % 2)
        ):
            raise ValueError(
                "Frame FiLM needs one frame; temporal FiLM needs an even context_len>=2"
            )
        if frame_step < 1 or encoder_batch_size < 1 or self.target_index != -1:
            raise ValueError(
                "FiLM predicts the final input frame; strides/batch sizes must be positive"
            )
        self.hidden_dim, self.context_len, self.frame_step = (
            hidden_dim,
            context_len,
            frame_step,
        )
        self.encoder_batch_size = encoder_batch_size
        self.conditioner = SpatialConditioner(2048, hidden_dim)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * self.z_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * self.z_dim),
        )
        self.decoder = FiLMSpatialDecoder(hidden_dim, self.z_dim)
        self.video_encoder = None

    @property
    def data_config(self):
        return dict(
            dataset_type="temporal" if self.temporal else "frame",
            image_size=self.image_size,
            context_len=self.context_len,
            frame_step=self.frame_step,
            target_type="coord",
        )

    @property
    def feature_identity(self):
        return dict(
            context_len=self.context_len,
            frame_step=self.frame_step,
            pretrained_sha256=self.hparams.encoder_sha256,
        )

    def attach_encoder(self, encoder=None):
        if encoder is None:
            root = Path(__file__).resolve().parents[2]
            repo = Path(self.hparams.encoder_repo)
            weights = Path(self.hparams.encoder_checkpoint)
            encoder = FrozenVJEPA2Encoder(
                repo if repo.is_absolute() else root / repo,
                weights if weights.is_absolute() else root / weights,
                self.hparams.encoder_sha256,
            )
        self.video_encoder = encoder.to(self.device).requires_grad_(False).eval()
        return self

    def encode_context(self, images=None, *, features=None):
        if (images is None) == (features is None):
            raise ValueError("Provide exactly one of images or features")
        if features is None:
            if images.ndim == 4:
                images = images[:, None]
            if images.ndim != 5 or images.shape[1] != self.context_len:
                raise ValueError(f"Expected {self.context_len} input frame(s)")
            if self.video_encoder is None:
                self.attach_encoder()
            if not self.temporal:
                images = images.expand(-1, 2, -1, -1, -1)
            features = torch.cat(
                [
                    self.video_encoder(images[first : first + self.encoder_batch_size])
                    for first in range(0, len(images), self.encoder_batch_size)
                ]
            )
        if features.ndim != 4 or tuple(features.shape[1:]) != self.feature_shape:
            raise ValueError(f"Expected features[B,{self.feature_shape}]")
        return self.conditioner(features)

    def decode(self, context, canvas, z):
        return self.decoder(context, canvas, z)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        # Frozen external weights are identified by path/hash, not duplicated.
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("video_encoder.")
        }

    def on_load_checkpoint(self, checkpoint):
        # Lightning can reload the best head into a model whose frozen encoder
        # was already attached during raw-video training. Reuse that encoder.
        if self.video_encoder is not None:
            for key, value in self.video_encoder.state_dict().items():
                checkpoint["state_dict"].setdefault("video_encoder." + key, value)


class TemporalFiLMCVAE(FiLMCVAE):
    """Causal video FiLM CVAE, sharing the complete head and objective."""

    temporal = True

    def __init__(self, context_len=16, frame_step=2, **kwargs):
        # Selected on novice active task A; explicit/checkpoint values take precedence.
        kwargs.setdefault("hidden_dim", 64)
        kwargs.setdefault("lr", 1e-3)
        kwargs.setdefault("weight_decay", 10.0)
        kwargs.setdefault("beta", 0.3)
        kwargs.setdefault("encoder_batch_size", 16)
        super().__init__(context_len=context_len, frame_step=frame_step, **kwargs)
