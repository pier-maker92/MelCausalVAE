import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
@dataclass
class SemanticMapperConfig:
    z_dim: int = 64
    n_layers: int = 6
    hidden_dim: int = 128
    pretrained_model_path: Optional[str] = None

    @property
    def hidden_size(self) -> int:
        """Alias for hidden_dim, required by DeepSpeed auto-config."""
        return self.hidden_dim

    def to_dict(self) -> dict:
        """Convert config to dict, required by HuggingFace Trainer."""
        from dataclasses import asdict

        return asdict(self)


# ---------------------------------------------------------
# SemanticMapperOutput
# ---------------------------------------------------------
@dataclass
class SemanticMapperOutput:
    """Output from Z2YMapper"""

    y: torch.Tensor  # [B, T, z_dim]
    cons_loss: Optional[torch.Tensor] = None  # Always 0 here, kept for API compatibility


# ---------------------------------------------------------
# Coupling Layer with mask
# ---------------------------------------------------------
class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask  # plain tensor, moved to device in forward/inverse

        self.net_s = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.net_t = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        m = self.mask.to(x.device, x.dtype)
        x_masked = x * m
        s = self.net_s(x_masked)
        s = torch.clamp(s, min=-5.0, max=5.0)  # Prevent exp explosion
        t = self.net_t(x_masked)
        y = x_masked + (1 - m) * (x * torch.exp(s) + t)
        return y

    def inverse(self, y):
        m = self.mask.to(y.device, y.dtype)
        y_masked = y * m
        s = self.net_s(y_masked)
        s = torch.clamp(s, min=-5.0, max=5.0)  # Prevent exp explosion
        t = self.net_t(y_masked)
        x = y_masked + (1 - m) * ((y - t) * torch.exp(-s))
        return x


# ---------------------------------------------------------
# Z2YMapper invertible flow
# ---------------------------------------------------------
class Z2YMapper(nn.Module):
    """
    Z2Y Mapper: invertible flow from z to y with RealNVP-style coupling layers.
    """

    def __init__(self, config: SemanticMapperConfig):
        super().__init__()
        self.config = config
        dim = config.z_dim

        # create alternating masks
        masks = []
        m1 = torch.zeros(dim)
        m1[: dim // 2] = 1
        m2 = 1 - m1
        for i in range(config.n_layers):
            masks.append(m1 if i % 2 == 0 else m2)

        # coupling layers
        self.layers = nn.ModuleList([CouplingLayer(dim, config.hidden_dim, masks[i]) for i in range(config.n_layers)])

        if config.pretrained_model_path:
            self.from_pretrained(config.pretrained_model_path)

    def from_pretrained(self, path):
        import safetensors

        state_dict = safetensors.torch.load_file(path)
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {path}")

    # ------------------------
    # Forward: z -> y
    # ------------------------
    def forward(self, z):
        y = z
        for layer in self.layers:
            y = layer(y)
        return SemanticMapperOutput(y=y)

    # ------------------------
    # Inverse: y -> z
    # ------------------------
    @torch.no_grad()
    def inverse(self, y):
        x = y
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


# ---------------------------------------------------------
# Test invertibility
# ---------------------------------------------------------
def test_z2y_mapper():
    cfg = SemanticMapperConfig()
    model = Z2YMapper(cfg)

    B, T = 16, 3000
    z = torch.randn(B, T, cfg.z_dim, dtype=torch.float32)

    # Forward
    out = model(z)
    y = out.y
    print("Forward output y shape:", y.shape)

    # Inverse
    z_rec = model.inverse(y)
    print("Inverse output z_rec shape:", z_rec.shape)

    # Check reconstruction error
    err = ((z - z_rec).abs()).mean()
    print("Max reconstruction error:", err.item())
    assert err < 1e-3, "Invertibility broken!"
    print("✓ Invertible flow works perfectly.")


if __name__ == "__main__":
    test_z2y_mapper()
