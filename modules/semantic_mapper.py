"""
Semantic Mapper Module (Z2YMapper)

Transforms VAE latent representations (z) to refined latent space (y) using a bottleneck architecture.
The y representation is then used with InterpolateRegulator for semantic distillation from SeamlessM4Tv2.
"""

import torch
import safetensors
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SemanticMapperConfig:
    """Configuration for the semantic mapper (Z2YMapper)"""

    z_dim: int = 64  # VAE latent dimension (input and output dimension)
    bottleneck_dim: int = 48  # Bottleneck dimension for compression
    dropout: float = 0.1  # Dropout rate
    lambda_cons: float = 0.2  # Weight for consistency loss MSE(y, z)
    pretrained_model_path: Optional[str] = None  # Path to pretrained model

    @property
    def hidden_size(self):
        """Return hidden size for DeepSpeed ZeRO-3 compatibility"""
        return self.z_dim + self.bottleneck_dim

    def to_dict(self):
        """Convert config to dict for DeepSpeed compatibility"""
        return {
            "z_dim": self.z_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "dropout": self.dropout,
            "lambda_cons": self.lambda_cons,
            "hidden_size": self.hidden_size,
            "pretrained_model_path": self.pretrained_model_path,
        }


@dataclass
class SemanticMapperOutput:
    """Output from the semantic mapper"""

    y: torch.Tensor  # Refined latent features [B, T, z_dim]
    cons_loss: torch.Tensor  # Consistency loss with input z


class Z2YMapper(nn.Module):
    """
    Z2Y Mapper: Transforms VAE latents through a bottleneck.

    Architecture:
        z [B, T, z_dim=64]
        -> bottleneck [B, T, bottleneck_dim=48]
        -> y [B, T, z_dim=64]

    The output y is then used with InterpolateRegulator for semantic distillation.
    Loss components:
        - semantic_loss: From InterpolateRegulator(y, semantic_features) - cosine similarity
        - cons_loss: MSE(y, z) to maintain connection to original latent
    """

    def __init__(self, config: SemanticMapperConfig):
        super().__init__()
        self.config = config

        # z_dim -> bottleneck_dim
        self.fc1 = nn.Linear(config.z_dim, config.bottleneck_dim)
        self.ln1 = nn.LayerNorm(config.bottleneck_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.dropout)

        # bottleneck_dim -> z_dim
        self.fc2 = nn.Linear(config.bottleneck_dim, config.z_dim)
        self.ln_out = nn.LayerNorm(config.z_dim)
        if config.pretrained_model_path is not None:
            self.from_pretrained(config.pretrained_model_path)

    def from_pretrained(self, checkpoint_path: str):
        state_dict = safetensors.torch.load_file(checkpoint_path)
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode z to bottleneck"""
        x = self.fc1(z)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        y = self.fc2(x)
        return self.ln_out(y)

    def forward(
        self,
        z: torch.Tensor,
        compute_cons_loss: bool = False,
        mu: Optional[torch.Tensor] = None,
    ) -> SemanticMapperOutput:
        """
        Forward pass through bottleneck.

        Args:
            z: VAE latent [B, T, z_dim]
            compute_cons_loss: Whether to compute consistency loss

        Returns:
            SemanticMapperOutput with y and optional consistency loss
        """
        # z: [B, T, z_dim] -> y: [B, T, z_dim]
        x = self.fc1(z)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        y = self.fc2(x)
        y = self.ln_out(y)

        # Compute consistency loss if requested
        if compute_cons_loss:
            cons_loss = F.mse_loss(y, mu.detach()) * self.config.lambda_cons
        else:
            cons_loss = torch.tensor(0.0, device=y.device, dtype=y.dtype)

        return SemanticMapperOutput(
            y=y,
            cons_loss=cons_loss,
        )


def test_z2y_mapper():
    """Test the Z2Y mapper with dummy data"""
    config = SemanticMapperConfig(
        z_dim=64,
        bottleneck_dim=48,
        dropout=0.1,
    )

    mapper = Z2YMapper(config)

    # Test forward pass
    B, T = 2, 50
    z = torch.randn(B, T, config.z_dim)

    # Without loss
    output = mapper(z, compute_cons_loss=False)
    print(f"Input z shape: {z.shape}")
    print(f"Output y shape: {output.y.shape}")
    assert output.y.shape == (B, T, config.z_dim), f"Expected {(B, T, config.z_dim)}, got {output.y.shape}"
    print(f"Consistency loss (not computed): {output.cons_loss.item():.4f}")

    # With loss
    output = mapper(z, compute_cons_loss=True)
    print(f"Consistency loss: {output.cons_loss.item():.4f}")

    print("✓ All tests passed!")


if __name__ == "__main__":
    test_z2y_mapper()
