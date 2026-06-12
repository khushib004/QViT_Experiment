"""ViT implementation with attention-weight capture for explainability.

vit_tiny default: embed_dim=192, depth=12, heads=3.
Every block stores its attention matrix in `.attn_weights` (B, heads, N, N)
so attention-rollout can later reconstruct a saliency map over patches.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.n_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class MLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class ClassicalMHA(nn.Module):
    """Standard multi-head self-attention that exposes its attention map."""

    def __init__(self, dim, heads, drop=0.0):
        super().__init__()
        assert dim % heads == 0, "embed_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, hd)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.last_attn = attn.detach()
        out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0, attn_module: nn.Module = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_module or ClassicalMHA(dim, heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    @property
    def attn_weights(self) -> Optional[torch.Tensor]:
        return getattr(self.attn, "last_attn", None)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        heads: int = 3,
        num_classes: int = 2,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_factory=None,  # callable(dim) -> nn.Module; None -> classical MHA
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.grid = self.patch_embed.grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim, heads, mlp_ratio, drop,
                    attn_module=attn_factory(embed_dim) if attn_factory else None,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        B = x.shape[0]
        h = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)  # (B, N+1, D)

    def forward(self, x):
        return self.head(self.forward_features(x)[:, 0])

    @torch.no_grad()
    def embed(self, x):
        """CLS-token embedding for t-SNE / clustering visualizations."""
        return self.forward_features(x)[:, 0]

    def attention_maps(self):
        """List of per-block attention tensors (B, heads, N, N) from last forward."""
        return [b.attn_weights for b in self.blocks if b.attn_weights is not None]


def vit_tiny(num_classes: int = 2, attn_factory=None) -> ViT:
    return ViT(embed_dim=192, depth=12, heads=3, num_classes=num_classes, attn_factory=attn_factory)
