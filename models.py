"""
Neural network model architecture for survival analysis.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Continuous-time sinusoidal encoding computed only on valid timesteps.
    Padded rows stay zero to avoid injecting signal.
    Uses learnable positional embeddings.
    """
    def __init__(self, d_model, base, dropout, max_seq_len):
        super().__init__()
        self.base = base
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, dynamic_times: torch.Tensor) -> torch.Tensor:
        # Initialize variables
        B, L, D = x.shape
        assert D == self.d_model, "x last dim must match d_model"

        # Use learnable positional embeddings
        if L <= self.max_seq_len:
            pe = self.pos_embedding[:, :L, :]
        else:
            # If sequence is longer than max_seq_len, use interpolation
            pe = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=L,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        # Mask out padded positions
        valid = torch.isfinite(dynamic_times)  # (B, L)
        pe = pe.expand(B, -1, -1)
        pe = pe * valid.unsqueeze(-1).float()

        return self.dropout(x + pe)


class TransformerEncoderBlock(nn.Module):
    """
    A Transformer Encoder block inspired by modern LLM architectures (e.g., LLaMA).

    This block incorporates two key improvements over the standard Transformer block:
    1.  **Pre-Layer Normalization**: LayerNorm is applied *before* the self-attention and
        feed-forward layers. This leads to more stable training, especially for deeper models.
    2.  **SwiGLU Activation**: The standard ReLU-based feed-forward network is replaced
        with a SwiGLU (Gated Linear Unit with Swish activation). This has been shown
        to improve performance by providing a more expressive and dynamically controlled
        information flow.
    3.  **DropPath/StochasticDepth**: Randomly drops entire residual branches during training for regularization.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout, drop_path_rate):
        super().__init__()
        # First layer normalization (before self-attention)
        self.norm1 = nn.LayerNorm(embed_dim)
        # Multi-head self-attention layer
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Second layer normalization (before feed-forward network)
        self.norm2 = nn.LayerNorm(embed_dim)

        # --- SwiGLU feed-forward network ---
        # The ff_dim is the "hidden" dimension inside the FFN.
        # We define three linear layers for the SwiGLU formulation.
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, ff_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.drop_path_rate = drop_path_rate

    def drop_path(self, x, drop_prob, training=False):
        """Drop paths (Stochastic Depth) per sample for regularization."""
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # --- 1. Pre-LN self-attention ---
        # Apply LayerNorm, then self-attention, then add the residual connection.
        norm_x = self.norm1(x)
        # Self-attention: use three same tensors as arugments (norm_x).
        # This allows each element in the sequence to look at and incorporate information from all other elements in
        # the same sequence, creating a context-aware representation for each element.
        attn_output, _ = self.attn(norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask, need_weights=False)
        attn_output = self.dropout(attn_output)
        x = x + self.drop_path(attn_output, self.drop_path_rate, self.training)

        # --- 2. Pre-LN SwiGLU feed-forward network ---
        # Apply LayerNorm, then the SwiGLU FFN, then add the residual connection.
        norm_x = self.norm2(x)
        # SwiGLU logic: F.silu(w1(x)) * w3(x)
        ffn_output = self.w2(nn.functional.silu(self.w1(norm_x)) * self.w3(norm_x))
        ffn_output = self.dropout(ffn_output)
        x = x + self.drop_path(ffn_output, self.drop_path_rate, self.training)

        return x


class TransformerSurv(nn.Module):
    """
    Transformer-based survival model for AFT (Accelerated Failure Time) modelling.
    Processes dynamic features through transformer and static features separately.
    Uses attention-based pooling for aggregating all timesteps.

    This class defines a deep learning network for survival analysis, using a transformer architecture to
    model time-series data alongside static patient features.

    ### Core architecture:
    The model is designed to handle both dynamic (time-varying) and static (time-invariant) features, making it
    suited for clinical data where patient measurements are taken over time.

    1.  **Input projections for features and masks**:
        *   A key feature of this model is its explicit handling of missing data. It uses separate `nn.Linear`
            projection layers for the feature values and their corresponding boolean masks (`dynamic_projection` and
            `dynamic_mask_projection`, `static_projection` and `static_mask_projection`).
        *   This design allows the model to learn a dedicated, meaningful representation (an "embedding") for
            the specific pattern of missingness in the data. Instead of just treating missing values as zero, the model
            can learn, for example, that a missing lab value is a predictive signal in itself.
        *   In the `forward` pass, the feature embedding and the missingness embedding are added together, creating a
            rich input representation that informs the model about both the observed values and the values that were missing.

    2.  **Time-aware positional encoding**:
        *   It employs a custom `PositionalEncoding` layer that uses learnable positional embeddings.
            Padded timesteps (marked by `NaN`) are correctly ignored.

    3.  **Transformer encoder for dynamic features**:
        *   The core of the dynamic feature processing is a stack of `TransformerEncoderBlock` modules. These blocks use
            Pre-Layer Normalization and SwiGLU activations for improved training stability and performance.
        *   An attention mask (`key_padding_mask`) is generated from the `dynamic_times` tensor to ensure the model does not
            attend to padded, non-existent timesteps.

    4.  **Attention-based pooling**:
        *   Uses learnable query vector for attention-based pooling to aggregate information from all timesteps.

    5.  **Combination and output**:
        *   After the Transformer processes the dynamic sequence, attention pooling aggregates all hidden states.
        *   This final dynamic representation is concatenated with the processed static feature embedding.
        *   This combined vector, which summarizes the patient's entire state, is fed into a final prediction head.

    This model outputs log_scale and log_shape parameters for the Weibull AFT model.
    """

    def __init__(self, num_bins: int, num_dynamic_features: int, num_static_features: int,
                 embed_dim: int, num_heads: int, num_transformer_blocks: int, ff_dim: int,
                 dropout: float, drop_path_rate: float, pos_enc_base: int, max_seq_len: int,
                 logger):
        super().__init__()
        # Initialize variables
        self.embed_dim = embed_dim
        self.num_bins = num_bins
        self.logger = logger

        # --- Feature and mask projection layers with residual connections ---
        # Linear layer to project dynamic features into the embedding space.
        self.dynamic_projection = nn.Linear(num_dynamic_features, embed_dim)
        # Linear layer to project the dynamic feature mask, learning an embedding for missingness patterns.
        self.dynamic_mask_projection = nn.Linear(num_dynamic_features, embed_dim)

        # Linear layer to project static features into the embedding space.
        self.static_projection = nn.Linear(num_static_features, embed_dim)
        # Linear layer to project the static feature mask.
        self.static_mask_projection = nn.Linear(num_static_features, embed_dim)

        # --- Optional temporal convolutions for multi-scale feature extraction ---
        # Multi-scale temporal convolutions (different kernel sizes capture different temporal patterns)
        self.temporal_conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.temporal_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.temporal_conv3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim)
        # Pointwise conv to mix multi-scale features
        self.temporal_pointwise = nn.Conv1d(embed_dim * 3, embed_dim, kernel_size=1)
        self.temporal_norm = nn.LayerNorm(embed_dim)

        # --- Feature interaction layer ---
        # Cross-attention between dynamic and static features
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        self.static_to_dynamic_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # --- Transformer core components ---
        # Positional encoding layer that uses learnable embeddings.
        self.positional_encoding = PositionalEncoding(d_model=embed_dim, base=pos_enc_base, dropout=dropout,
                                                      max_seq_len=max_seq_len)
        # A stack of Transformer Encoder blocks to process the time-series data.
        # Use stochastic depth with linearly increasing drop_path_rate across layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_transformer_blocks)]
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
                                    drop_path_rate=dpr[i])
            for i in range(num_transformer_blocks)
        ])

        # --- Attention-based pooling for aggregating all timesteps ---
        # Learnable query vector for attention pooling
        self.attention_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention_pooling = nn.MultiheadAttention(embed_dim, num_heads=1, dropout=dropout, batch_first=True)

        # --- Final prediction head for AFT ---
        # Calculate the input dimension for the final classifier/risk head
        input_dim = embed_dim * 2  # Combined dimension of dynamic hidden state and static embedding

        # For AFT, output the log scale and log shape parameters
        self.params_pred_head = nn.Linear(input_dim, 2)
        # Initialize with smaller weights for more stability
        nn.init.xavier_uniform_(self.params_pred_head.weight, gain=0.01)
        nn.init.zeros_(self.params_pred_head.bias)

    def forward(self, x_dynamic, x_dynamic_mask, dynamic_times, x_static, x_static_mask):
        """
        x_dynamic: (B, L, F_dyn) dynamic features
        x_dynamic_mask: (B, L, F_dyn) True for missing/padded, False for valid
        x_static: (B, F_stat) static features
        x_static_mask: (B, F_stat) True for missing, False for valid
        dynamic_times: (B, L) with NaN on padding
        """
        # Get batch size and sequence length from the input dynamic features tensor.
        batch_size, seq_len, _ = x_dynamic.shape  # B, L, F_dyn

        # --- 1. Process dynamic features ---
        # Project dynamic features into the embedding space. Shape: (B, L, F_dyn) -> (B, L, D)
        x_dyn_proj = self.dynamic_projection(x_dynamic)

        # Project the boolean mask (converted to float) to learn a missingness embedding. Shape: (B, L, F_dyn) -> (B, L, D)
        mask_dyn_proj = self.dynamic_mask_projection(x_dynamic_mask.float())

        # Add the feature and missingness embeddings together. Shape: (B, L, D)
        x_dyn = x_dyn_proj + mask_dyn_proj

        # Scale the combined embedding by the square root of the embedding dimension, a standard practice in transformers.
        x_dyn = x_dyn * math.sqrt(self.embed_dim)

        # Add time-aware positional encoding. Shape remains (B, L, D).
        x_dyn = self.positional_encoding(x_dyn, dynamic_times)

        # --- 2a. Optional temporal convolutions for multi-scale feature extraction ---
        # Apply multi-scale temporal convolutions
        x_conv_input = x_dyn.transpose(1, 2)  # (B, D, L) for Conv1d
        conv1_out = F.gelu(self.temporal_conv1(x_conv_input))
        conv2_out = F.gelu(self.temporal_conv2(x_conv_input))
        conv3_out = F.gelu(self.temporal_conv3(x_conv_input))
        # Concatenate multi-scale features and mix with pointwise conv
        conv_concat = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)  # (B, 3*D, L)
        conv_out = self.temporal_pointwise(conv_concat).transpose(1, 2)  # (B, L, D)
        # Residual connection with layer norm
        x_dyn = x_dyn + self.temporal_norm(conv_out)

        # --- 2. Prepare attention mask for transformer ---
        # Create a key padding mask to ignore padded timesteps (where time is NaN).
        # `True` means the position should be ignored. Shape: (B, L)
        key_padding_mask = ~torch.isfinite(dynamic_times)

        # Safety check: if a sequence is fully padded, unmask the first token to prevent NaN outputs from attention.
        fully_masked = key_padding_mask.all(dim=1)  # Shape: (B,)
        if fully_masked.any():
            key_padding_mask[fully_masked, 0] = False

        # --- 3. Pass through transformer encoder ---
        # Process the sequence through the stack of transformer blocks. Shape remains (B, L, D).
        for block in self.transformer_blocks:
            x_dyn = block(x_dyn, key_padding_mask=key_padding_mask)

        # --- 4. Pool dynamic features using attention-based pooling ---
        # Attention-based pooling using learnable query
        query = self.attention_query.expand(batch_size, -1, -1)  # (B, 1, D)
        x_dynamic_pooled, _ = self.attention_pooling(
            query, x_dyn, x_dyn,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x_dynamic_pooled = x_dynamic_pooled.squeeze(1)  # (B, D)
        
        # --- 5. Process static features ---
        # Project static features into the embedding space. Shape: (B, F_stat) -> (B, D)
        x_stat_proj = self.static_projection(x_static)
        # Project the static feature mask to get a missingness embedding. Shape: (B, F_stat) -> (B, D)
        mask_stat_proj = self.static_mask_projection(x_static_mask.float())
        # Add the static feature and missingness embeddings. Shape: (B, D)
        x_static_hidden = x_stat_proj + mask_stat_proj

        # --- 5a. Cross-attention between static and dynamic features ---
        # Use static features to attend to dynamic sequence
        x_static_query = x_static_hidden.unsqueeze(1)  # (B, 1, D)
        x_static_normed = self.cross_attn_norm(x_static_query)
        cross_attn_out, _ = self.static_to_dynamic_attn(
            x_static_normed, x_dyn, x_dyn,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        # Enhanced static representation with dynamic context
        x_static_hidden = (x_static_hidden + cross_attn_out.squeeze(1)) * 0.5

        # --- 6. Combine embeddings and make prediction ---
        # Create a list of embeddings to be concatenated.
        combined_hidden = [x_dynamic_pooled, x_static_hidden]  # [(B, D), (B, D)]

        # Concatenate all embeddings to form the final representation vector.
        # Shape: (B, D*2)
        combined_hidden = torch.cat(combined_hidden, dim=1)  # torch.Size([B, 192])

        # Pass the final representation through the AFT prediction head.
        log_params = self.params_pred_head(combined_hidden)  # (B, 2)
        # Clamp log_params to prevent numerical instability
        log_params = torch.clamp(log_params, min=-5.0, max=5.0)
        return log_params
