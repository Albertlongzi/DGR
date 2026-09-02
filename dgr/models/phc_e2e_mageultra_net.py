from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse proven building blocks from existing MegaNet
from dgr.models.phc_e2e_mega_net import (
    DeeperModalityEncoder,
    AttentionBlock,
    DeformableCrossAttention,
    CrossModalFuse,
    HybridTransformerBlock,
)
from dgr.models.phc_net import ResidualBlock


# Note: Removed signal-quality gating. Use standard B0SpatialAttention.


class DeformableCrossAttentionCA(nn.Module):
    """
    Contrast-aware deformable cross-attention (DCA) with optional contrast token modulation.
    """
    def __init__(self, dim: int, num_heads: int = 4, num_points: int = 4, latent_dim: int = 0) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        self.latent_dim = int(latent_dim)
        self.offset_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, self.num_heads * self.num_points * 2, 3, padding=1),
        )
        self.attn_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, self.num_heads * self.num_points, 3, padding=1),
        )
        self.out_proj = nn.Conv2d(dim, dim, 1)
        if self.latent_dim > 0:
            self.contrast_offset = nn.Sequential(
                nn.Linear(self.latent_dim, dim), nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
            )
            self.contrast_attn = nn.Sequential(
                nn.Linear(self.latent_dim, dim), nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
            )

    @staticmethod
    def _make_base_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing='ij'
        )
        return torch.stack([x, y], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, contrast_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = q.shape
        concat = torch.cat([q, k], dim=1)  # [B,2C,H,W]
        if self.latent_dim > 0 and contrast_vec is not None:
            c_off = self.contrast_offset(contrast_vec).view(B, C, 1, 1)
            c_att = self.contrast_attn(contrast_vec).view(B, C, 1, 1)
            mod_off = torch.cat([c_off, c_off], dim=1)
            mod_att = torch.cat([c_att, c_att], dim=1)
            concat = concat * (1.0 + 0.1 * mod_off) + 0.05 * mod_att
        offsets = self.offset_net(concat)
        attn_w = self.attn_net(concat)
        offsets = offsets.view(B, self.num_heads, self.num_points, 2, H, W)
        attn_w = attn_w.view(B, self.num_heads, self.num_points, H, W)
        attn_w = torch.softmax(attn_w, dim=2)
        base_grid = self._make_base_grid(H, W, q.device)
        samp_grid = base_grid.view(1, 1, 1, H, W, 2) + torch.tanh(offsets.permute(0,1,2,4,5,3))
        if C % self.num_heads != 0:
            raise ValueError("Channels must be divisible by num_heads for deformable attention")
        ch = C // self.num_heads
        k_heads = k.view(B, self.num_heads, ch, H, W)
        out = torch.zeros_like(k)
        for h in range(self.num_heads):
            agg = 0.0
            for p in range(self.num_points):
                grid_hp = samp_grid[:, h, p]
                v = F.grid_sample(k_heads[:, h], grid_hp, mode='bilinear', padding_mode='border', align_corners=False)
                w = attn_w[:, h, p].unsqueeze(1)
                agg = agg + v * w
            out[:, h * ch:(h + 1) * ch] = agg
        return self.out_proj(out) + q


class CrossModalFuseCA(nn.Module):
    """Contrast-aware CrossModalFuse (channel attention)"""
    def __init__(self, ch: int, reduction: int = 4, latent_dim: int = 0) -> None:
        super().__init__()
        hid = max(1, ch // reduction)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * ch, hid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, 2 * ch, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = ResidualBlock(ch, ch)
        self.latent_dim = int(latent_dim)
        if self.latent_dim > 0:
            self.contrast_mlp = nn.Sequential(
                nn.Linear(self.latent_dim, ch), nn.ReLU(inplace=True),
                nn.Linear(ch, 2 * ch)
            )

    def forward(self, fd: torch.Tensor, ft: torch.Tensor, contrast_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.cat([fd, ft], dim=1)
        w = self.mlp(x)
        if self.latent_dim > 0 and contrast_vec is not None:
            B, C, H, W = fd.shape
            c_w = self.contrast_mlp(contrast_vec).view(B, 2 * C, 1, 1)
            w = w * (1.0 + 0.2 * torch.sigmoid(c_w))
        ch = fd.shape[1]
        wd, wt = w[:, :ch], w[:, ch:]
        y = wd * fd + wt * ft
        return self.refine(y)


class PHCE2EMageUltraNet(nn.Module):
    """
    Dual-stream PHC E2E network (MageUltra) without B0/VDM:
      - Separate pipelines for b50 and ADC (no cross-b fusion)
      - Each pipeline mirrors single-stream style without any VDM/B0 attention
      - Pure distortion learning from DWI stacks and T2
    """

    def __init__(self, dwi_channels: int, t2_channels: int, base_channels: int = 64, latent_dim: int = 0, prompt_k: int = 8, prompt_temp: float = 1.0) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.latent_dim = int(latent_dim)
        # For compatibility with training flags; not used in Mega-style latent
        self.prompt_k = int(prompt_k)
        self.prompt_temp = float(prompt_temp)
        self.use_contrast = self.latent_dim > 0

        # DWI encoders (no VDM channel concatenation)
        self.enc_dwi_low = DeeperModalityEncoder(dwi_channels, base=self.base_channels)
        # ADC path encoder
        self.enc_dwi_adc = DeeperModalityEncoder(dwi_channels, base=self.base_channels)
        # T2 encoders (fully independent per stream)
        self.enc_t2_low = DeeperModalityEncoder(t2_channels, base=self.base_channels)
        self.enc_t2_adc = DeeperModalityEncoder(t2_channels, base=self.base_channels)
        # s0 stems (full-res projection)
        self.s0_dwi_low_stem = nn.Conv2d(dwi_channels, self.base_channels, 3, padding=1)
        self.s0_t2_low_stem = nn.Conv2d(t2_channels, self.base_channels, 3, padding=1)
        self.s0_dwi_adc_stem = nn.Conv2d(dwi_channels, self.base_channels, 3, padding=1)
        self.s0_t2_adc_stem = nn.Conv2d(t2_channels, self.base_channels, 3, padding=1)
        # Mega-style optional FiLM driven by contrast latent
        if self.use_contrast:
            def _film(ch: int) -> nn.Module:
                return nn.Sequential(
                    nn.Linear(self.latent_dim, max(16, ch // 2)), nn.ReLU(inplace=True),
                    nn.Linear(max(16, ch // 2), ch * 2)
                )
            # DWI FiLM per b-value and per scale
            self.film_dwi_low_s1 = _film(self.base_channels)
            self.film_dwi_low_s2 = _film(self.base_channels * 2)
            self.film_dwi_low_s3 = _film(self.base_channels * 4)
            self.film_dwi_adc_s1 = _film(self.base_channels)
            self.film_dwi_adc_s2 = _film(self.base_channels * 2)
            self.film_dwi_adc_s3 = _film(self.base_channels * 4)
            # T2 FiLM per stream and per scale (no sharing between low/high)
            self.film_t2_low_s1 = _film(self.base_channels)
            self.film_t2_low_s2 = _film(self.base_channels * 2)
            self.film_t2_low_s3 = _film(self.base_channels * 4)
            self.film_t2_adc_s1 = _film(self.base_channels)
            self.film_t2_adc_s2 = _film(self.base_channels * 2)
            self.film_t2_adc_s3 = _film(self.base_channels * 4)
            # Contrast encoder (Avg DWI central + T2 central) -> latent
            self.contrast_enc = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.contrast_fc = nn.Linear(16, self.latent_dim)

        # Note: No B0/VDM-guided spatial attention in this TSE variant

        # Per-b DWI-T2 interaction (refactored per-scale)
        # b50
        # s0
        self.cma_dt_s0_low = DeformableCrossAttentionCA(self.base_channels, num_heads=4, num_points=4, latent_dim=self.latent_dim if self.use_contrast else 0)
        self.hyb_s0_low = HybridTransformerBlock(self.base_channels, num_heads=4, window=16)
        self.cmf_s0_low = CrossModalFuseCA(self.base_channels, latent_dim=self.latent_dim if self.use_contrast else 0)
        # s1
        self.cma_dt_s1_low = DeformableCrossAttentionCA(self.base_channels, num_heads=4, num_points=4, latent_dim=self.latent_dim if self.use_contrast else 0)
        self.hyb_s1_low = HybridTransformerBlock(self.base_channels, num_heads=4, window=16)
        self.cmf_s1_low = CrossModalFuseCA(self.base_channels, latent_dim=self.latent_dim if self.use_contrast else 0)
        # s2
        self.cma_dt_s2_low = DeformableCrossAttentionCA(self.base_channels * 2, num_heads=4, num_points=4, latent_dim=self.latent_dim if self.use_contrast else 0)
        self.cmf_dt_s2_low = CrossModalFuseCA(self.base_channels * 2, latent_dim=self.latent_dim if self.use_contrast else 0)
        self.hyb_dt_s2_low = HybridTransformerBlock(self.base_channels * 2, num_heads=4, window=16)
        # s3
        self.cmf_dt_s3_low = CrossModalFuseCA(self.base_channels * 4, latent_dim=self.latent_dim if self.use_contrast else 0)
        self.hyb_s3_low = HybridTransformerBlock(self.base_channels * 4, num_heads=4, window=16)
        # ADC path
        # s0
        self.cma_dt_s0_adc = DeformableCrossAttentionCA(self.base_channels, num_heads=4, num_points=4, latent_dim=self.latent_dim if self.use_contrast else 0)
        self.hyb_s0_adc = HybridTransformerBlock(self.base_channels, num_heads=4, window=16)
        self.cmf_s0_adc = CrossModalFuseCA(self.base_channels, latent_dim=self.latent_dim if self.use_contrast else 0)
        # s1
        self.cma_dt_s1_adc = DeformableCrossAttentionCA(self.base_channels, num_heads=4, num_points=4, latent_dim=self.latent_dim if self.use_contrast else 0)
        self.hyb_s1_adc = HybridTransformerBlock(self.base_channels, num_heads=4, window=16)
        self.cmf_s1_adc = CrossModalFuseCA(self.base_channels, latent_dim=self.latent_dim if self.use_contrast else 0)
        # s2
        self.cma_dt_s2_adc = DeformableCrossAttentionCA(self.base_channels * 2, num_heads=4, num_points=4, latent_dim=self.latent_dim if self.use_contrast else 0)
        self.cmf_dt_s2_adc = CrossModalFuseCA(self.base_channels * 2, latent_dim=self.latent_dim if self.use_contrast else 0)
        self.hyb_dt_s2_adc = HybridTransformerBlock(self.base_channels * 2, num_heads=4, window=16)
        # s3
        self.cmf_dt_s3_adc = CrossModalFuseCA(self.base_channels * 4, latent_dim=self.latent_dim if self.use_contrast else 0)
        self.hyb_s3_adc = HybridTransformerBlock(self.base_channels * 4, num_heads=4, window=16)

        # Seeds for propagating aligned features across scales
        self.seed0to1_low = nn.Conv2d(self.base_channels, self.base_channels, 3, stride=2, padding=1)
        self.seed1to2_low = nn.Conv2d(self.base_channels, self.base_channels * 2, 3, stride=2, padding=1)
        self.seed2to3_low = nn.Conv2d(self.base_channels * 2, self.base_channels * 4, 3, stride=2, padding=1)
        self.seed0to1_adc = nn.Conv2d(self.base_channels, self.base_channels, 3, stride=2, padding=1)
        self.seed1to2_adc = nn.Conv2d(self.base_channels, self.base_channels * 2, 3, stride=2, padding=1)
        self.seed2to3_adc = nn.Conv2d(self.base_channels * 2, self.base_channels * 4, 3, stride=2, padding=1)
        # Projections to build ADC-side "k" from [T2, low-corrected] at each scale
        self.kproj_s0 = nn.Conv2d(self.base_channels * 2, self.base_channels, 1)
        self.kproj_s1 = nn.Conv2d(self.base_channels * 2, self.base_channels, 1)
        self.kproj_s2 = nn.Conv2d(self.base_channels * 4, self.base_channels * 2, 1)
        self.kproj_s3 = nn.Conv2d(self.base_channels * 8, self.base_channels * 4, 1)

        # FPN-like projections and refinement (per-b)
        fpn_ch = self.base_channels
        # b50
        self.lat_s1_low = nn.Conv2d(self.base_channels, fpn_ch, 1)
        self.lat_s2_low = nn.Conv2d(self.base_channels * 2, fpn_ch, 1)
        self.lat_s3_low = nn.Conv2d(self.base_channels * 4, fpn_ch, 1)
        self.td2_low = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.td1_low = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.bu2_low = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.bu3_low = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.fuse_refine_low = nn.Sequential(
            nn.Conv2d(fpn_ch * 3, fpn_ch * 3, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(fpn_ch * 3, fpn_ch * 3, 3, padding=1), nn.ReLU(inplace=True),
        )
        # ADC
        self.lat_s1_adc = nn.Conv2d(self.base_channels, fpn_ch, 1)
        self.lat_s2_adc = nn.Conv2d(self.base_channels * 2, fpn_ch, 1)
        self.lat_s3_adc = nn.Conv2d(self.base_channels * 4, fpn_ch, 1)
        self.td2_adc = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.td1_adc = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.bu2_adc = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.bu3_adc = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.fuse_refine_adc = nn.Sequential(
            nn.Conv2d(fpn_ch * 3, fpn_ch * 3, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(fpn_ch * 3, fpn_ch * 3, 3, padding=1), nn.ReLU(inplace=True),
        )

        # Two output heads (low-b and high-b reconstructions)
        fuse_ch = fpn_ch * 3
        self.head_full_low = nn.Sequential(
            nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(fuse_ch, 1, 1),
        )
        self.head_full_adc = nn.Sequential(
            nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(fuse_ch, 1, 1),
        )
        # Multi-scale side heads (s1 at 1/2 scale, s2 at 1/4 scale), per b-value
        self.s1_head_low = nn.Conv2d(self.base_channels, 1, 1)
        self.s2_head_low = nn.Conv2d(self.base_channels * 2, 1, 1)
        self.s1_head_adc = nn.Conv2d(self.base_channels, 1, 1)
        self.s2_head_adc = nn.Conv2d(self.base_channels * 2, 1, 1)

    def _dual_encode(self, dwi_b50_stack: torch.Tensor, adc_stack: torch.Tensor, t2_stack: torch.Tensor
                    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Encode directly (no VDM concatenation)
        s1_low, s2_low, s3_low = self.enc_dwi_low(dwi_b50_stack)
        s1_adc, s2_adc, s3_adc = self.enc_dwi_adc(adc_stack)
        s1_t2_low, s2_t2_low, s3_t2_low = self.enc_t2_low(t2_stack)
        s1_t2_adc, s2_t2_adc, s3_t2_adc = self.enc_t2_adc(t2_stack)

        return (s1_low, s2_low, s3_low), (s1_adc, s2_adc, s3_adc), (s1_t2_low, s2_t2_low, s3_t2_low), (s1_t2_adc, s2_t2_adc, s3_t2_adc)

    def _fpn_aggregate_path(self,
                            z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
                            lat_s1: nn.Module, lat_s2: nn.Module, lat_s3: nn.Module,
                            td2_m: nn.Module, td1_m: nn.Module, bu2_m: nn.Module, bu3_m: nn.Module,
                            fuse_m: nn.Module) -> torch.Tensor:
        p3 = lat_s3(z3)
        p2 = lat_s2(z2)
        p1 = lat_s1(z1)
        td3 = p3
        td2 = td2_m(p2 + F.interpolate(td3, size=p2.shape[-2:], mode="bilinear", align_corners=False))
        td1 = td1_m(p1 + F.interpolate(td2, size=p1.shape[-2:], mode="bilinear", align_corners=False))
        bu2_in = td2 + F.avg_pool2d(td1, kernel_size=2, stride=2)
        bu2 = bu2_m(bu2_in)
        bu3_in = td3 + F.avg_pool2d(bu2, kernel_size=2, stride=2)
        bu3 = bu3_m(bu3_in)
        td1_up = F.interpolate(td1, scale_factor=2, mode="bilinear", align_corners=False)
        bu2_up = F.interpolate(bu2, scale_factor=4, mode="bilinear", align_corners=False)
        bu3_up = F.interpolate(bu3, scale_factor=8, mode="bilinear", align_corners=False)
        f = torch.cat([td1_up, bu2_up, bu3_up], dim=1)
        f = fuse_m(f)
        return f

    def forward(
        self,
        dwi_b50_stack: torch.Tensor,
        adc_stack: torch.Tensor,
        t2_stack: torch.Tensor,
        domain_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Encode streams
        (s1_low, s2_low, s3_low), (s1_adc, s2_adc, s3_adc), (s1_t2_low, s2_t2_low, s3_t2_low), (s1_t2_adc, s2_t2_adc, s3_t2_adc) = self._dual_encode(
            dwi_b50_stack, adc_stack, t2_stack
        )
        # Optional Mega-style FiLM by contrast latent (applied on encoder outputs)
        if self.use_contrast:
            # Build contrast vectors independently per stream (low/high)
            dwi_c_l = dwi_b50_stack[:, dwi_b50_stack.shape[1] // 2 : dwi_b50_stack.shape[1] // 2 + 1]
            dwi_c_a = adc_stack[:, adc_stack.shape[1] // 2 : adc_stack.shape[1] // 2 + 1]
            t2_c_l = t2_stack[:, t2_stack.shape[1] // 2 : t2_stack.shape[1] // 2 + 1]
            t2_c_a = t2_c_l  # same raw t2 central slice as input; encoders are independent downstream
            # Low contrast
            h_low = self.contrast_enc(torch.cat([dwi_c_l, t2_c_l], dim=1)).flatten(1)
            contrast_vec_low = self.contrast_fc(h_low)
            # ADC contrast
            h_adc = self.contrast_enc(torch.cat([dwi_c_a, t2_c_a], dim=1)).flatten(1)
            contrast_vec_adc = self.contrast_fc(h_adc)
            def _apply_film(feat: torch.Tensor, head: nn.Module, cvec: torch.Tensor) -> torch.Tensor:
                B, C, _, _ = feat.shape
                gb = head(cvec)
                gamma, beta = gb[:, :C], gb[:, C:]
                gamma = gamma.view(B, C, 1, 1)
                beta = beta.view(B, C, 1, 1)
                # Enhanced FiLM scaling and bias
                return feat * (1.0 + torch.tanh(gamma) * 1.0) + beta * 0.25
            # Apply to DWI low/high and T2 features at all scales (independently)
            s1_low = _apply_film(s1_low, self.film_dwi_low_s1, contrast_vec_low)
            s2_low = _apply_film(s2_low, self.film_dwi_low_s2, contrast_vec_low)
            s3_low = _apply_film(s3_low, self.film_dwi_low_s3, contrast_vec_low)
            s1_adc = _apply_film(s1_adc, self.film_dwi_adc_s1, contrast_vec_adc)
            s2_adc = _apply_film(s2_adc, self.film_dwi_adc_s2, contrast_vec_adc)
            s3_adc = _apply_film(s3_adc, self.film_dwi_adc_s3, contrast_vec_adc)
            s1_t2_low = _apply_film(s1_t2_low, self.film_t2_low_s1, contrast_vec_low)
            s2_t2_low = _apply_film(s2_t2_low, self.film_t2_low_s2, contrast_vec_low)
            s3_t2_low = _apply_film(s3_t2_low, self.film_t2_low_s3, contrast_vec_low)
            s1_t2_adc = _apply_film(s1_t2_adc, self.film_t2_adc_s1, contrast_vec_adc)
            s2_t2_adc = _apply_film(s2_t2_adc, self.film_t2_adc_s2, contrast_vec_adc)
            s3_t2_adc = _apply_film(s3_t2_adc, self.film_t2_adc_s3, contrast_vec_adc)
        # s0 (full-res): stems → DCA → CMF → HYB → seed to s1
        dwi0_low = self.s0_dwi_low_stem(dwi_b50_stack)
        t20_low = self.s0_t2_low_stem(t2_stack)
        z0l = self.cma_dt_s0_low(dwi0_low, t20_low, contrast_vec_low if self.use_contrast else None)
        z0l = self.cmf_s0_low(z0l, t20_low, contrast_vec_low if self.use_contrast else None)
        z0l = self.hyb_s0_low(z0l)
        seed1_low = self.seed0to1_low(z0l)
        # s1 (1/2): add seed → DCA → CMF → HYB
        s1_low = s1_low + seed1_low
        z1l = self.cma_dt_s1_low(s1_low, s1_t2_low, contrast_vec_low if self.use_contrast else None)
        z1l = self.cmf_s1_low(z1l, s1_t2_low, contrast_vec_low if self.use_contrast else None)
        z1l = self.hyb_s1_low(z1l)
        # s2 (1/4): add seed from s1 → DCA → CMF → HYB
        seed2_low = self.seed1to2_low(z1l)
        s2_low = s2_low + seed2_low
        z2l = self.cma_dt_s2_low(s2_low, s2_t2_low, contrast_vec_low if self.use_contrast else None)
        z2l = self.cmf_dt_s2_low(z2l, s2_t2_low, contrast_vec_low if self.use_contrast else None)
        z2l = self.hyb_dt_s2_low(z2l)
        # s3 (1/8): add seed from s2 → CMF → HYB
        seed3_low = self.seed2to3_low(z2l)
        s3_low = s3_low + seed3_low
        z3l = self.cmf_dt_s3_low(s3_low, s3_t2_low, contrast_vec_low if self.use_contrast else None)
        z3l = self.hyb_s3_low(z3l)
        f_low = self._fpn_aggregate_path(
            z1l, z2l, z3l,
            self.lat_s1_low, self.lat_s2_low, self.lat_s3_low,
            self.td2_low, self.td1_low, self.bu2_low, self.bu3_low,
            self.fuse_refine_low
        )
        out_low = self.head_full_low(f_low)
        # ADC path (use T2 + corrected low-b features as keys for interaction; detach low to block gradients)
        dwi0_adc = self.s0_dwi_adc_stem(adc_stack)
        t20_adc = self.s0_t2_adc_stem(t2_stack)
        k0_adc = self.kproj_s0(torch.cat([t20_adc, z0l.detach()], dim=1))
        z0a = self.cma_dt_s0_adc(dwi0_adc, k0_adc, contrast_vec_adc if self.use_contrast else None)
        z0a = self.cmf_s0_adc(z0a, k0_adc, contrast_vec_adc if self.use_contrast else None)
        z0a = self.hyb_s0_adc(z0a)
        seed1_adc = self.seed0to1_adc(z0a)
        s1_adc = s1_adc + seed1_adc
        k1_adc = self.kproj_s1(torch.cat([s1_t2_adc, z1l.detach()], dim=1))
        z1a = self.cma_dt_s1_adc(s1_adc, k1_adc, contrast_vec_adc if self.use_contrast else None)
        z1a = self.cmf_s1_adc(z1a, k1_adc, contrast_vec_adc if self.use_contrast else None)
        z1a = self.hyb_s1_adc(z1a)
        seed2_adc = self.seed1to2_adc(z1a)
        s2_adc = s2_adc + seed2_adc
        k2_adc = self.kproj_s2(torch.cat([s2_t2_adc, z2l.detach()], dim=1))
        z2a = self.cma_dt_s2_adc(s2_adc, k2_adc, contrast_vec_adc if self.use_contrast else None)
        z2a = self.cmf_dt_s2_adc(z2a, k2_adc, contrast_vec_adc if self.use_contrast else None)
        z2a = self.hyb_dt_s2_adc(z2a)
        seed3_adc = self.seed2to3_adc(z2a)
        s3_adc = s3_adc + seed3_adc
        k3_adc = self.kproj_s3(torch.cat([s3_t2_adc, z3l.detach()], dim=1))
        z3a = self.cmf_dt_s3_adc(s3_adc, k3_adc, contrast_vec_adc if self.use_contrast else None)
        z3a = self.hyb_s3_adc(z3a)
        f_adc = self._fpn_aggregate_path(
            z1a, z2a, z3a,
            self.lat_s1_adc, self.lat_s2_adc, self.lat_s3_adc,
            self.td2_adc, self.td1_adc, self.bu2_adc, self.bu3_adc,
            self.fuse_refine_adc
        )
        out_adc = self.head_full_adc(f_adc)
        # Multi-scale side outputs (upsampled to full resolution)
        H, W = dwi_b50_stack.shape[-2], dwi_b50_stack.shape[-1]
        s1_low = torch.nn.functional.interpolate(self.s1_head_low(z1l), size=(H, W), mode="bilinear", align_corners=False)
        s2_low = torch.nn.functional.interpolate(self.s2_head_low(z2l), size=(H, W), mode="bilinear", align_corners=False)
        s1_adc = torch.nn.functional.interpolate(self.s1_head_adc(z1a), size=(H, W), mode="bilinear", align_corners=False)
        s2_adc = torch.nn.functional.interpolate(self.s2_head_adc(z2a), size=(H, W), mode="bilinear", align_corners=False)
        return {
            "I_out_b50": out_low,
            "I_out_adc": out_adc,
            "I_s1_b50": s1_low,
            "I_s2_b50": s2_low,
            "I_s1_adc": s1_adc,
            "I_s2_adc": s2_adc,
        }



