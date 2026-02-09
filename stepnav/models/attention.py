import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer-based sequence models."""

    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, : x.size(1), :]
        return x


class DIFP(nn.Module):
    """Dynamics-Inspired Feature Refinement (DIFP) module.

    Refines raw temporal features Z = [z_1, ..., z_T] via a goal-conditional
    variational formulation. The refined features are the closed-form solution:

        Z_tilde = (2I + L^T L)^{-1} (Z + z_c 1^T)

    where L is the temporal Laplacian and z_c is the global motion context
    computed as a learned attention-weighted average of the raw features.

    This ensures temporally smooth, goal-aligned features for downstream
    success field estimation.

    Args:
        feature_dim: Dimension D of each temporal feature vector.
        max_seq_len: Maximum number of temporal tokens T.
    """

    def __init__(self, feature_dim, max_seq_len=6):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len

        # Learnable attention weights for computing global motion context z_c
        self.context_attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1),
        )

    def _build_temporal_laplacian(self, T, device):
        """Build the T x T temporal Laplacian matrix L.

        The Laplacian L is a tridiagonal matrix with 2 on the diagonal and
        -1 on the off-diagonals (with appropriate boundary handling).

        Args:
            T: Sequence length.
            device: Torch device.

        Returns:
            L: (T, T) temporal Laplacian matrix.
        """
        L = torch.zeros(T, T, device=device)
        for i in range(T):
            L[i, i] = 2.0
            if i > 0:
                L[i, i - 1] = -1.0
            if i < T - 1:
                L[i, i + 1] = -1.0
        # Boundary adjustment for first and last rows
        L[0, 0] = 1.0
        L[T - 1, T - 1] = 1.0
        return L

    def forward(self, Z):
        """Apply DIFP refinement to temporal feature sequence.

        Args:
            Z: (B, T, D) raw temporal features from the encoder.

        Returns:
            Z_tilde: (B, T, D) refined temporal features.
            z_c: (B, D) global motion context vector.
        """
        B, T, D = Z.shape
        device = Z.device

        # Step 1: Compute attention weights w_t for global motion context
        attn_logits = self.context_attn(Z)  # (B, T, 1)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, T, 1)

        # Step 2: Compute global motion context z_c = sum_t w_t * z_t
        z_c = (attn_weights * Z).sum(dim=1)  # (B, D)

        # Step 3: Build temporal Laplacian L and solve the closed-form system
        # Z_tilde = (2I + L^T L)^{-1} (Z + z_c 1^T)
        L = self._build_temporal_laplacian(T, device)  # (T, T)
        LtL = L.t() @ L  # (T, T)
        A = 2.0 * torch.eye(T, device=device) + LtL  # (T, T)

        # Right-hand side: Z + z_c * 1^T, where 1^T broadcasts z_c across time
        z_c_expanded = z_c.unsqueeze(1).expand_as(Z)  # (B, T, D)
        rhs = Z + z_c_expanded  # (B, T, D)

        # Solve the linear system A @ Z_tilde = rhs for each feature dimension
        # A is shared across batch and feature dims, so we solve per-batch
        # Reshape for batched solve: (B, T, D)
        A_expanded = A.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)
        Z_tilde = torch.linalg.solve(A_expanded, rhs)  # (B, T, D)

        return Z_tilde, z_c
