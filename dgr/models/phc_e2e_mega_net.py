from typing import Dict, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgr.models.phc_net import (
    ModalityEncoder,
    CrossModalFuse,
    B0SpatialAttention,
)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class AttentionBlock(nn.Module):
    def __init__(self, ch: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_n = self.norm(x)
        q = self.q(x_n).view(b, self.num_heads, c // self.num_heads, h * w)
        k = self.k(x_n).view(b, self.num_heads, c // self.num_heads, h * w)
        v = self.v(x_n).view(b, self.num_heads, c // self.num_heads, h * w)
        attn = torch.softmax(torch.einsum('bhcn,bhcm->bhnm', q, k) / (c // self.num_heads) ** 0.5, dim=-1)
        o = torch.einsum('bhnm,bhcm->bhcn', attn, v).contiguous().view(b, c, h, w)
        return x + self.proj(o)


class DeeperModalityEncoder(nn.Module):
    def __init__(self, in_ch: int, base: int) -> None:
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        # stem 1/2
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            ResidualBlock(c1), ResidualBlock(c1), ResidualBlock(c1),
        )
        # 1/4
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            ResidualBlock(c2), ResidualBlock(c2), ResidualBlock(c2),
        )
        # 1/8
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            ResidualBlock(c3), ResidualBlock(c3), ResidualBlock(c3),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return s1, s2, s3


class DeformableCrossAttention(nn.Module):
    """
    Lightweight deformable cross-attention: query=DWI feat, key=T2 feat (same C/H/W).
    Heads kept small to control memory; offsets are normalized to [-1, 1] grid.
    """
    def __init__(self, dim: int, num_heads: int = 4, num_points: int = 4) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        # Learn offsets and attention weights from concatenated query+key
        self.offset_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, self.num_heads * self.num_points * 2, 3, padding=1),
        )
        self.attn_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, self.num_heads * self.num_points, 3, padding=1),
        )
        self.out_proj = nn.Conv2d(dim, dim, 1)

    @staticmethod
    def _make_base_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=-1)  # [H,W,2]
        return grid

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, C, H, W = q.shape
        concat = torch.cat([q, k], dim=1)
        offsets = self.offset_net(concat)  # [B, heads*pts*2, H, W]
        attn_w = self.attn_net(concat)     # [B, heads*pts,   H, W]
        offsets = offsets.view(B, self.num_heads, self.num_points, 2, H, W)
        attn_w = attn_w.view(B, self.num_heads, self.num_points, H, W)
        attn_w = torch.softmax(attn_w, dim=2)  # softmax over points

        # Base grid for sampling, broadcast to heads/points later
        base_grid = self._make_base_grid(H, W, q.device)  # [H,W,2]
        # Sampling grid per head/point: [B, heads, pts, H, W, 2]
        samp_grid = base_grid.view(1, 1, 1, H, W, 2) + torch.tanh(offsets.permute(0,1,2,4,5,3))
        # Prepare key for sampling per head: split channels across heads
        if C % self.num_heads != 0:
            raise ValueError("Channels must be divisible by num_heads for deformable attention")
        ch = C // self.num_heads
        k_heads = k.view(B, self.num_heads, ch, H, W)
        out = torch.zeros_like(k)
        # Sample and aggregate per head
        for h in range(self.num_heads):
            # Aggregate over points
            agg = 0.0
            for p in range(self.num_points):
                grid_hp = samp_grid[:, h, p]  # [B,H,W,2]
                v = F.grid_sample(k_heads[:, h], grid_hp, mode='bilinear', padding_mode='border', align_corners=False)
                w = attn_w[:, h, p].unsqueeze(1)  # [B,1,H,W]
                agg = agg + v * w
            out[:, h * ch:(h + 1) * ch] = agg
        return self.out_proj(out) + q


class HybridTransformerBlock(nn.Module):
    """
    Local DWConv + window-based MHSA (torch.nn.MultiheadAttention) + MLP.
    """
    def __init__(self, dim: int, num_heads: int = 4, window: int = 8) -> None:
        super().__init__()
        self.window = int(window)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def _pad_to_window(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        ph = (self.window - H % self.window) % self.window
        pw = (self.window - W % self.window) % self.window
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        return x, (ph, pw)

    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window, self.window, W // self.window, self.window)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, nH, nW, win, win, C]
        x = x.view(B * (H // self.window) * (W // self.window), self.window * self.window, C)
        return x

    def _window_reverse(self, xw: torch.Tensor, B: int, C: int, H: int, W: int) -> torch.Tensor:
        nH = H // self.window
        nW = W // self.window
        xw = xw.view(B, nH, nW, self.window, self.window, C)
        xw = xw.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
        return xw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Local branch
        local = self.pw(F.gelu(self.dw(x)))
        # Window attn branch
        x_pad, (ph, pw) = self._pad_to_window(x)
        Bp, Cp, Hp, Wp = x_pad.shape
        tokens = self._window_partition(x_pad)  # [B*nW, win*win, C]
        tokens = self.norm1(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out + tokens
        attn_out = attn_out + self.mlp(self.norm2(attn_out))
        feat = self._window_reverse(attn_out, Bp, Cp, Hp, Wp)
        if ph or pw:
            feat = feat[:, :, :H, :W]
        return local + feat


class PHCE2EMegaNet(nn.Module):
    """
    Mega variant:
      - base_channels=64
      - encoders use 3 residual blocks per scale
      - attention at 1/2 and 1/4 scales
      - multi-scale reconstruction heads at 1/2 and 1/4
      - final output is direct image, no residual add-back
    """

    def __init__(self, dwi_channels: int, t2_channels: int, base_channels: int = 64, latent_dim: int = 0) -> None:
        super().__init__()
        self.base_channels = base_channels
        self.latent_dim = int(latent_dim)
        self.use_contrast = self.latent_dim > 0
        # Encoders with VDM concatenated to DWI stream
        self.enc_dwi = DeeperModalityEncoder(dwi_channels + 1, base=base_channels)
        self.enc_t2 = DeeperModalityEncoder(t2_channels, base=base_channels)

        # Optional contrast FiLM (per-scale) driven by a small latent vector inferred from inputs
        if self.use_contrast:
            def _film(ch: int) -> nn.Module:
                return nn.Sequential(
                    nn.Linear(self.latent_dim, max(16, ch // 2)), nn.ReLU(inplace=True),
                    nn.Linear(max(16, ch // 2), ch * 2)
                )
            # DWI stream FiLM at 3 scales
            self.film_dwi_s1 = _film(base_channels)
            self.film_dwi_s2 = _film(base_channels * 2)
            self.film_dwi_s3 = _film(base_channels * 4)
            # T2 stream FiLM at 3 scales
            self.film_t2_s1 = _film(base_channels)
            self.film_t2_s2 = _film(base_channels * 2)
            self.film_t2_s3 = _film(base_channels * 4)
            # Lightweight contrast encoder from central DWI/T2 slices -> latent
            self.contrast_enc = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.contrast_fc = nn.Linear(16, self.latent_dim)

        # VDM-guided attention on DWI features
        self.b0_att_s1 = B0SpatialAttention(in_ch=base_channels)
        self.b0_att_s2 = B0SpatialAttention(in_ch=base_channels * 2)
        self.b0_att_s3 = B0SpatialAttention(in_ch=base_channels * 4)

        # Cross-modal fusion per scale
        # 1/2: deformable cross-attention (DWI query, T2 key)
        self.cma_s1 = DeformableCrossAttention(base_channels, num_heads=4, num_points=4)
        # 1/4: retain simple fuse then hybrid transformer enhancement
        self.cmf_s2 = CrossModalFuse(base_channels * 2)
        self.hyb_s2 = HybridTransformerBlock(base_channels * 2, num_heads=4, window=8)
        self.cmf_s3 = CrossModalFuse(base_channels * 4)

        # Self-attention at 1/2 and 1/4 (s1, s2)
        self.att_s1 = AttentionBlock(base_channels, num_heads=4)
        self.att_s2 = AttentionBlock(base_channels * 2, num_heads=4)

        # FPN-like lateral projections
        fpn_ch = base_channels
        self.lat_s1 = nn.Conv2d(base_channels, fpn_ch, 1)
        self.lat_s2 = nn.Conv2d(base_channels * 2, fpn_ch, 1)
        self.lat_s3 = nn.Conv2d(base_channels * 4, fpn_ch, 1)
        self.td2 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.td1 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.bu2 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.bu3 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)

        # Full-res fusion
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(fpn_ch * 3, fpn_ch * 3, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(fpn_ch * 3, fpn_ch * 3, 3, padding=1), nn.ReLU(inplace=True),
        )

        fuse_ch = fpn_ch * 3
        # Multi-scale reconstruction heads
        self.head_full = nn.Sequential(
            nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(fuse_ch, 1, 1),
        )
        self.head_s1 = nn.Sequential(  # 1/2 scale
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
        )
        self.head_s2 = nn.Sequential(  # 1/4 scale
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, 1, 1),
        )

    def _encode_fuse(self, dwi_stack: torch.Tensor, t2_stack: torch.Tensor, vdm: torch.Tensor, contrast_vec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_dwi = torch.cat([dwi_stack, vdm], dim=1)
        s1d, s2d, s3d = self.enc_dwi(x_dwi)
        s1t, s2t, s3t = self.enc_t2(t2_stack)

        # Optional FiLM modulation by contrast token
        if self.use_contrast:
            if contrast_vec is None:
                # Build contrast vector from central slices of DWI/T2
                t2_c = t2_stack[:, t2_stack.shape[1] // 2 : t2_stack.shape[1] // 2 + 1]
                dwi_c = dwi_stack[:, dwi_stack.shape[1] // 2 : dwi_stack.shape[1] // 2 + 1]
                contrast_in = torch.cat([dwi_c, t2_c], dim=1)
                h = self.contrast_enc(contrast_in).flatten(1)
                contrast_vec = self.contrast_fc(h)
            def _apply_film(feat: torch.Tensor, head: nn.Module) -> torch.Tensor:
                B, C, H, W = feat.shape
                gb = head(contrast_vec)  # [B, 2C]
                gamma, beta = gb[:, :C], gb[:, C:]
                gamma = gamma.view(B, C, 1, 1)
                beta = beta.view(B, C, 1, 1)
                return feat * (1.0 + torch.tanh(gamma) * 0.5) + beta * 0.1
            s1d = _apply_film(s1d, self.film_dwi_s1)
            s2d = _apply_film(s2d, self.film_dwi_s2)
            s3d = _apply_film(s3d, self.film_dwi_s3)
            s1t = _apply_film(s1t, self.film_t2_s1)
            s2t = _apply_film(s2t, self.film_t2_s2)
            s3t = _apply_film(s3t, self.film_t2_s3)

        s1d = self.b0_att_s1(vdm, s1d)
        s2d = self.b0_att_s2(vdm, s2d)
        s3d = self.b0_att_s3(vdm, s3d)

        z1 = self.cma_s1(s1d, s1t)
        z2 = self.cmf_s2(s2d, s2t)
        z2 = self.hyb_s2(z2)
        z3 = self.cmf_s3(s3d, s3t)

        # (attention handled by deformable/hybrid blocks above)

        # FPN aggregation to full resolution
        p3 = self.lat_s3(z3)
        p2 = self.lat_s2(z2)
        p1 = self.lat_s1(z1)
        td3 = p3
        td2 = self.td2(p2 + F.interpolate(td3, size=p2.shape[-2:], mode="bilinear", align_corners=False))
        td1 = self.td1(p1 + F.interpolate(td2, size=p1.shape[-2:], mode="bilinear", align_corners=False))
        bu2_in = td2 + F.avg_pool2d(td1, kernel_size=2, stride=2)
        bu2 = self.bu2(bu2_in)
        bu3_in = td3 + F.avg_pool2d(bu2, kernel_size=2, stride=2)
        bu3 = self.bu3(bu3_in)

        td1_up = F.interpolate(td1, scale_factor=2, mode="bilinear", align_corners=False)
        bu2_up = F.interpolate(bu2, scale_factor=4, mode="bilinear", align_corners=False)
        bu3_up = F.interpolate(bu3, scale_factor=8, mode="bilinear", align_corners=False)
        f = torch.cat([td1_up, bu2_up, bu3_up], dim=1)
        f = self.fuse_refine(f)
        return f, z1, z2, z3

    def forward(self, dwi_stack: torch.Tensor, t2_stack: torch.Tensor, vdm: torch.Tensor, contrast_vec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        f, z1, z2, _ = self._encode_fuse(dwi_stack, t2_stack, vdm, contrast_vec)
        i_full = self.head_full(f)
        # Multi-scale predictions upsampled to full resolution for loss
        i_s1 = F.interpolate(self.head_s1(z1), scale_factor=2, mode="bilinear", align_corners=False)
        i_s2 = F.interpolate(self.head_s2(z2), scale_factor=4, mode="bilinear", align_corners=False)
        return {
            "I_out": i_full,
            "I_s1": i_s1,
            "I_s2": i_s2,
        }


