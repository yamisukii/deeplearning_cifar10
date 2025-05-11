import torch
import torch.nn as nn
from pathlib import Path


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_size))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)                  
        x = x.flatten(2).transpose(1, 2)   
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=128, num_heads=4, depth=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(emb_size),
                nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(emb_size),
                nn.Sequential(
                    nn.Linear(emb_size, int(emb_size * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(emb_size * mlp_ratio), emb_size),
                    nn.Dropout(dropout),
                )
            ]))

    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            x = x + attn(norm1(x), norm1(x), norm1(x))[0]
            x = x + mlp(norm2(x))
        return x


class vitModel(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, emb_size=128, depth=4, heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            in_channels=3,
            patch_size=patch_size,
            emb_size=emb_size,
            img_size=image_size
        )
        self.encoder = TransformerEncoder(
            emb_size=emb_size,
            num_heads=heads,
            depth=depth
        )
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.patch_embed.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)       
        x = self.encoder(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return self.head(cls_token)

    def save(self, save_dir, suffix=""):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.__class__.__name__}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".pt"
        path = save_dir / filename
        torch.save(self.state_dict(), path)
        return path
