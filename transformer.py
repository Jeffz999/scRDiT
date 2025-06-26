import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from typing import Optional

import math
import numpy as np


def modulate(x, shift, scale):
    """
    Applies adaptive instance normalization.
    """
    # The shift operation is optional
    if shift is None:
        # If no shift is provided, create a zero tensor with the same shape as the scale tensor
        shift = torch.zeros_like(scale)
    # The modulation operation: scale the input x and then add the shift
    # scale is expanded to match the dimensions of x for element-wise multiplication
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(nn.Module):
    """
    Patch Embedding for 1D data.
    """
    def __init__(self, img_size=2000, patch_size=10, in_c=1, embed_dim=768, norm_layer=None, bias=False):
        super().__init__()
        # Ensure img_size and patch_size are treated as 1D
        self.img_size = (img_size,)
        self.patch_size = (patch_size,)
        self.grid_size = (img_size // patch_size,)
        self.num_patches = self.grid_size[0]
        # Use Conv1d for 1D data
        self.proj = nn.Conv1d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.img_size[0], f"Input length ({L}) doesn't match model ({self.img_size[0]})."
        x = self.proj(x)
        x = x.transpose(1, 2)  # [B, C, L] -> [B, L, C]
        x = self.norm(x)
        return x


def attention(q, k, v, heads):
    """
    Convenience wrapper around PyTorch's scaled dot-product attention.
    """
    b, _, dim_head = q.shape
    # Reshape q, k, v for multi-head attention
    q, k, v = map(
        lambda t: t.view(b, -1, heads, dim_head // heads).transpose(1, 2),
        (q, k, v)
    )
    # Apply scaled dot product attention
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    # Reshape output back to original dimensions
    return out.transpose(1, 2).reshape(b, -1, dim_head)


class Attention(nn.Module):
    """
    Attention module with optional Query-Key Normalization.
    This version is designed to be more stable by applying LayerNorm
    to the entire Q and K projections.
    """
    def __init__(self, dim, num_heads, qkv_bias=False, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Linear layer to project input to Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Final projection layer
        self.proj = nn.Linear(dim, dim)

        # Apply LayerNorm to Q and K projections if qk_norm is enabled
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x):
        B, L, C = x.shape

        # 1. Project to Q, K, V and split
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. Apply QK Norm (the corrected approach)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. Perform attention using the helper function
        x = attention(q, k, v, self.num_heads)

        # 4. Final projection
        x = self.proj(x)
        return x


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network from the Stable Diffusion 3 architecture.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
    ):
        super().__init__()
        # Adjust hidden dimension based on SwiGLU architecture
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # Apply the SwiGLU activation: silu(w1(x)) * w3(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    This version is updated to resolve potential conflicts between adaLN and QK-Norm,
    and incorporates SwiGLU for the MLP layer.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qk_norm=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Use the new, corrected Attention class
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # Use SwiGLU which has shown better performance in modern transformers
        self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim)
        
        # Modulation network for adaLN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # Generate modulation parameters from the conditioning signal
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention block with modulation
        modulated_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output = self.attn(modulated_input)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # MLP block with modulation
        modulated_mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(modulated_mlp_input)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=2000,
        patch_size=10,
        in_channels=1,
        hidden_size=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # The core of the model: a sequence of DiTBlocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qk_norm=True) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
           nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # --- FIX: Use a proper initialization for the final layer ---
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # Instead of zero-initializing, use Xavier uniform for the final linear layer
        torch.nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size * C)
        imgs: (N, C, L)
        """
        c = self.out_channels
        p = self.patch_size
        l_p = self.x_embedder.num_patches  # Number of patches
        assert l_p == x.shape[1]

        x = x.reshape(shape=(x.shape[0], l_p, p, c))
        x = torch.einsum('nlpc->nclp', x)
        imgs = x.reshape(shape=(x.shape[0], c, l_p * p))
        return imgs

    def forward(self, x, t):
        """
        Forward pass of DiT.
        x: (N, C, L) tensor of 1D inputs
        t: (N,) tensor of diffusion timesteps
        """
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Create 1D sin/cos positional embeddings.
    """
    grid = np.arange(grid_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


if __name__ == '__main__':
    # Test code.
    t = torch.randint(0, 1000, (16,))
    x = torch.randn((16, 1, 2000))
    dit = DiT(depth=12, hidden_size=768, patch_size=10, num_heads=12)
    out = dit(x, t)
    print(out.shape)
    
    # Check parameter count
    num_params = sum(p.numel() for p in dit.parameters())
    print(f"Total parameters: {num_params:,}")
