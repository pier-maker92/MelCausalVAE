import torch
import torch.nn as nn


class QformerPooler(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        latent_dim=64,
        num_heads=8,
        num_learnable_queries=3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_learnable_queries = num_learnable_queries

        # 3 learnable queries of dimension embed_dim
        self.learnable_queries = nn.Parameter(
            torch.randn(1, 1, num_learnable_queries, embed_dim)
        )

        # Cross Attention: queries are embed_dim, keys/values are latent_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=latent_dim,
            vdim=latent_dim,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        # Bottleneck projection: concatenating 4 queries (1 embedding + 3 learnable)
        concat_dim = (1 + num_learnable_queries) * embed_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(concat_dim, embed_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(embed_dim, latent_dim)
        self.logvar = nn.Linear(embed_dim, latent_dim)

        # Convolutional Refiner to reconstruct mu and logvar after upsampling
        self.refiner = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(latent_dim * 2, latent_dim * 2, kernel_size=3, padding=1),
        )

    def forward(self, embeddings, alignments, mu, kl_loss_pooler_weight):
        """
        Args:
            embeddings: [B, N, embed_dim] (Phoneme embeddings)
            alignments: [B, T, N] (Monotone matrix of 1s and 0s)
            mu: [B, T, latent_dim] (Continuous representations)

        Returns:
            z: [B, N, latent_dim] (Bottlenecked phoneme-level representations)
            mu_recon: [B, T, latent_dim] (Reconstructed form of mu)
        """
        B, N, D = embeddings.shape
        _, T, _ = mu.shape

        # 1. Prepare Queries
        # queries from embeddings: [B, N, 1, D]
        q_emb = embeddings.unsqueeze(2)
        # learnable queries: [B, N, 3, D]
        q_learn = self.learnable_queries.expand(B, N, -1, -1)

        # Concat to get all 4 queries: [B, N, 4, D]
        queries = torch.cat([q_emb, q_learn], dim=2)

        # Flatten N and 4 for MultiheadAttention: [B, N*4, D]
        queries = queries.reshape(B, N * 4, D)

        # 2. Prepare Attention Mask
        # alignments is [B, T, N]
        # We want a mask of shape [B, N*4, T] where True means NOT allowed to attend
        mask = alignments.transpose(1, 2).unsqueeze(2)  # [B, N, 1, T]
        mask = mask.expand(-1, -1, 1 + self.num_learnable_queries, -1)  # [B, N, 4, T]
        mask = mask.reshape(B, N * 4, T)  # [B, N*4, T]

        bool_mask = mask == 0  # True where alignments are 0 (disallowed)

        # MHA with boolean mask requires that no row is completely True,
        # otherwise it results in NaN. We specify all False for those.
        all_true = bool_mask.all(dim=-1, keepdim=True)
        bool_mask = bool_mask.masked_fill(all_true, False)

        # MHA expects attn_mask to be 2D (L, S) or 3D (B*num_heads, L, S) for batch_first=True
        # Our mask is different per batch item, so we must use 3D.
        bool_mask_mha = bool_mask.unsqueeze(1).repeat(
            1, self.cross_attn.num_heads, 1, 1
        )
        bool_mask_mha = bool_mask_mha.view(B * self.cross_attn.num_heads, N * 4, T)

        # 3. Cross Attention
        # queries: [B, N*4, embed_dim]
        # keys/values (mu): [B, T, latent_dim]
        attn_out, _ = self.cross_attn(
            query=queries, key=mu, value=mu, attn_mask=bool_mask_mha
        )

        # Residual and Norm
        queries = self.norm1(queries + attn_out)  # [B, N*4, embed_dim]

        # 4. Concatenate over the 4 queries
        # Reshape back to [B, N, 4, embed_dim] and then flatten the last two dims
        queries = queries.view(
            B, N, (1 + self.num_learnable_queries) * D
        )  # [B, N, 4*embed_dim]

        # 5. Bottleneck projection
        hidden = self.bottleneck(queries)  # [B, N, embed_dim]
        mu_z = self.mu(hidden)  # [B, N, latent_dim]
        logvar_z = self.logvar(hidden)  # [B, N, latent_dim]

        # Reparameterization trick
        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)
        z = mu_z + eps * std  # [B, N, latent_dim]

        # 6. Upsample and Reconstruct
        # Upsample using alignments: [B, T, N] @ [B, N, latent_dim] -> [B, T, latent_dim]
        align_float = alignments.to(z.dtype)
        z_upsampled = torch.bmm(align_float, z)  # [B, T, latent_dim]

        # Convolutional Refiner
        z_upsampled = z_upsampled.transpose(1, 2)  # [B, latent_dim, T]
        refiner_out = self.refiner(z_upsampled)  # [B, latent_dim * 2, T]
        refiner_out = refiner_out.transpose(1, 2)  # [B, T, latent_dim * 2]

        # Chunk the output into mu and logvar
        mu_recon, logvar_recon = torch.chunk(
            refiner_out, 2, dim=-1
        )  # [B, T, latent_dim] each

        # Reparameterization trick for reconstructed z
        std_recon = torch.exp(0.5 * logvar_recon)
        eps_recon = torch.randn_like(std_recon)
        z_recon = mu_recon + eps_recon * std_recon  # [B, T, latent_dim]

        # kl loss
        kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        kl_loss = kl_loss * kl_loss_pooler_weight

        return z_recon, mu_recon, kl_loss
