"""ViT-Tiny implementation (depth=12, dim=192, heads=3) without external deps."""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0, attn_module: nn.Module = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_module or nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self._is_torch_mha = isinstance(self.attn, nn.MultiheadAttention)

    def forward(self, x):
        h = self.norm1(x)
        if self._is_torch_mha:
            h, _ = self.attn(h, h, h, need_weights=False)
        else:
            h = self.attn(h)
        x = x + h
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
        attn_factory=None,  # callable(dim) -> nn.Module; if None, classical MHA
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    heads,
                    mlp_ratio,
                    drop,
                    attn_module=attn_factory(embed_dim) if attn_factory else None,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        h = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h)[:, 0])


def vit_tiny(num_classes: int = 2, attn_factory=None) -> ViT:
    return ViT(
        embed_dim=192,
        depth=12,
        heads=3,
        num_classes=num_classes,
        attn_factory=attn_factory,
    )
