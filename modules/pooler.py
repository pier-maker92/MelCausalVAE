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

        # Cross Attention: queries are embed_dim, keys/values are embed_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kdim=embed_dim,
            vdim=embed_dim,
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

        # Convolutional Refiner to reconstruct x after upsampling
        self.refiner = nn.Sequential(
            nn.Conv1d(latent_dim + 1, embed_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

    def forward(self, embeddings, alignments, x, padding_mask_phoneme, padding_mask_frames, kl_loss_pooler_weight):
        """
        Args:
            embeddings: [B, N, embed_dim] (Phoneme embeddings)
            alignments: [B, T, N] (Monotone matrix of 1s and 0s)
            x: [B, T, embed_dim] (Continuous representations)
            padding_mask_phoneme: [B, T] (Padding mask)
            padding_mask_frames: [B, T] (Padding mask)

        Returns:
            x_recon: [B, T, embed_dim] (Reconstructed form of x)
            kl_loss: [1] (KL divergence for pooler bottleneck)
            l1_l2_loss: [1] (Reconstruction loss on x)
        """
        B, N, D = embeddings.shape
        _, T, _ = x.shape

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
        # keys/values (x): [B, T, embed_dim]
        attn_out, _ = self.cross_attn(
            query=queries, key=x, value=x, attn_mask=bool_mask_mha
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
        # 5.1 add mean of the frames
        align_float = alignments.to(hidden.dtype)
        dur = align_float.sum(dim=1).unsqueeze(-1)  # [B, N, 1]
        hidden = hidden + torch.bmm(align_float.transpose(1, 2), x) / (dur + 1e-8)
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

        # Compute phoneme phase clock
        dur = align_float.sum(dim=1).clamp(min=1e-5) # [B, N]
        running = torch.cumsum(align_float, dim=1) - 0.5 # [B, T, N]
        phase_n = running / dur.unsqueeze(1) # [B, T, N]
        phase = (phase_n * align_float).sum(dim=-1, keepdim=True) # [B, T, 1]
        
        z_upsampled = torch.cat([z_upsampled, phase], dim=-1) # [B, T, latent_dim + 1]

        # Convolutional Refiner
        z_upsampled = z_upsampled.transpose(1, 2)  # [B, latent_dim + 1, T]
        x_recon = self.refiner(z_upsampled)  # [B, embed_dim, T]
        x_recon = x_recon.transpose(1, 2)  # [B, T, embed_dim]

        # kl loss
        padding_mask_phoneme = padding_mask_phoneme.squeeze(-1)
        kl_loss = -0.5 * torch.sum(1 + logvar_z[~padding_mask_phoneme] - mu_z[~padding_mask_phoneme].pow(2) - logvar_z[~padding_mask_phoneme].exp())
        kl_loss = kl_loss * kl_loss_pooler_weight

        # l1 + l2 loss
        l1_loss = torch.mean(torch.abs(x[~padding_mask_frames] - x_recon[~padding_mask_frames]))
        l2_loss = torch.mean((x[~padding_mask_frames] - x_recon[~padding_mask_frames])**2)
        l1_l2_loss = l1_loss + l2_loss

        return x_recon, kl_loss, l1_l2_loss
