from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import BasicUNet as MonaiBasicUNet
except Exception:  # pragma: no cover
    MonaiBasicUNet = None  # type: ignore[assignment]

from dgr.utils.warp import grid_warp_x, grid_warp_2d
from dgr.utils.splat import softmax_splat_2d


class ConvHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act: str = "tanh", bias: bool = True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, out_ch, 1, bias=True),
        )
        self.act = act
        # Store pre-activation tensor from the last forward pass for logging/diagnostics
        self.last_pre_act: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        # keep a reference for saturation logging (read-only in training loop)
        self.last_pre_act = y
        if self.act == "sigmoid":
            return torch.sigmoid(y)
        if self.act == "tanh":
            return torch.tanh(y)
        if self.act == "softplus":
            return F.softplus(y)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.proj: Optional[nn.Module]
        if in_ch != out_ch or stride != 1:
            self.proj = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)
        if self.proj is not None:
            identity = self.proj(identity)
        y = y + identity
        y = self.act2(y)
        return y


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.res = ResidualBlock(out_ch, out_ch, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        y = self.act(y)
        y = self.res(y)
        return y


class ModalityEncoder(nn.Module):
    """Residual pyramid encoder returning multi-scale features at 1/2, 1/4, 1/8 resolution."""

    def __init__(self, in_ch: int, base: int) -> None:
        super().__init__()
        c1 = base
        c2 = base * 2
        c3 = base * 4
        # produce 1/2 resolution directly for rich low-level details
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            ResidualBlock(c1, c1),
        )
        self.stage2 = DownsampleBlock(c1, c2)  # 1/4
        self.stage3 = DownsampleBlock(c2, c3)  # 1/8

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        return s1, s2, s3


class CrossModalFuse(nn.Module):
    """Channel-attention fusion of two modality features at the same scale.

    Computes modality-wise channel weights from pooled concatenated features.
    """

    def __init__(self, ch: int, reduction: int = 4) -> None:
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

    def forward(self, fd: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fd, ft], dim=1)
        w = self.mlp(x)
        ch = fd.shape[1]
        wd, wt = w[:, :ch], w[:, ch:]
        y = wd * fd + wt * ft
        y = self.refine(y)
        return y


class OffsetConv2dLite(nn.Module):
    """Lightweight deformable-like conv: predict a per-pixel 2D offset, warp inputs, then mix by 3x3 conv.

    This is a simplified fallback for environments without DCN ops.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.offset = nn.Conv2d(in_ch, 2, 3, padding=1, bias=True)
        self.mix = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ofs = self.offset(x)
        # normalize offsets for grid_sample
        gx = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        gy = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        base = torch.stack([xx, yy], dim=-1)  # [H,W,2]
        # scale pixel offsets to normalized coordinates
        ofs_norm_x = ofs[:, 0] * (2.0 / max(1, W))
        ofs_norm_y = ofs[:, 1] * (2.0 / max(1, H))
        grid = torch.stack([base[..., 0].unsqueeze(0).repeat(B, 1, 1) + ofs_norm_x,
                            base[..., 1].unsqueeze(0).repeat(B, 1, 1) + ofs_norm_y], dim=-1)
        xw = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        y = self.mix(xw)
        return y


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int) -> None:
        super().__init__()
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, 3, padding=1)
        self.cand = nn.Conv2d(in_ch + hidden_ch, hidden_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, h], dim=1)
        gates = self.gates(inp)
        r, z = torch.chunk(gates, 2, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        cand_in = torch.cat([x, r * h], dim=1)
        n = torch.tanh(self.cand(cand_in))
        h_new = (1 - z) * n + z * h
        return h_new


def build_corr_volume(f1: torch.Tensor, f2: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Compute correlation volume between f1 and f2 within a local window.
    Returns [B, (2r+1)^2, H, W].
    """
    B, C, H, W = f1.shape
    f1n = F.normalize(f1, dim=1)
    f2n = F.normalize(f2, dim=1)
    vols = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            pad_l = max(0, dx)
            pad_r = max(0, -dx)
            pad_t = max(0, dy)
            pad_b = max(0, -dy)
            shifted = F.pad(f2n, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
            shifted = shifted[:, :, pad_b:pad_b + H, pad_r:pad_r + W]
            corr = (f1n * shifted).sum(dim=1, keepdim=True)
            vols.append(corr)
    return torch.cat(vols, dim=1)


class B0SpatialAttention(nn.Module):
    """B0-guided spatial attention: Attention = Softmax(Conv(|B0| * scale)).

    Produces a single-channel spatial weight map per scale and applies it onto feature maps.
    """

    def __init__(self, in_ch: int, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.scale = nn.Parameter(torch.tensor(1.0))
        # regularize scale away from collapsing to 0 by using softplus floor at fwd
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=pad, bias=True)

    def forward(self, b0: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        # Resize |B0| to feat spatial size
        scale_pos = F.softplus(self.scale) + 1e-2
        b0_abs = b0.abs() * scale_pos
        b0_rs = F.interpolate(b0_abs, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        att_logits = self.conv(b0_rs)
        # Sigmoid gating in [0,1], residual scale in [1, 1+alpha]
        att = torch.sigmoid(att_logits)
        return feat * (1.0 + att)


class _SimpleEncoder(nn.Module):
    """Fallback encoder producing feature maps of size out_ch.

    A lightweight UNet-like stack without external deps.
    """

    def __init__(self, in_ch: int, out_ch: int, base: int = 32) -> None:
        super().__init__()
        b = base
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(b, b, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.down1 = nn.Conv2d(b, b * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.ReLU(inplace=True), nn.Conv2d(b * 2, b * 2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.down2 = nn.Conv2d(b * 2, b * 4, 3, stride=2, padding=1)
        self.bott = nn.Sequential(
            nn.ReLU(inplace=True), nn.Conv2d(b * 4, b * 4, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up1 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.ReLU(inplace=True), nn.Conv2d(b * 4, b * 2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up2 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.ReLU(inplace=True), nn.Conv2d(b * 2, b, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.bott(self.down2(e2))
        d1 = self.up1(b)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        return self.out(d2)


class PHCNet(nn.Module):
    """
    Physics-aware Hybrid Corrector (PHC) - 2D/2.5D variant.

    Inputs (per-slice stacks):
      - dwi_stack: [B, C_dwi, H, W]
      - t2_stack:  [B, C_t2, H, W]
      - vdm:       [B, 1, H, W] (VDM displacement field in pixels)

    Outputs:
      dict with u_pred (1x), a_pred (1x), sigma_u (1x), I_phys (1x), I_ref (1x),
      gate (1x), I_out (1x), sigma_out (1x)
    """

    def __init__(
        self,
        dwi_channels: int,
        t2_channels: int,
        base_channels: int = 32,
        max_disp_px: float = 16.0,
        vdm_prior_scale: float = 1,
    ) -> None:
        super().__init__()
        # Residual dual encoders with multi-scale outputs (1/2, 1/4, 1/8)
        self.enc_dwi = ModalityEncoder(dwi_channels + 1, base=base_channels)
        self.enc_t2 = ModalityEncoder(t2_channels, base=base_channels)

        # Cross-modal fusion at each scale with channel attention
        self.cmf_s1 = CrossModalFuse(base_channels)
        self.cmf_s2 = CrossModalFuse(base_channels * 2)
        self.cmf_s3 = CrossModalFuse(base_channels * 4)

        # B0-guided spatial attention at each scale (applied to DWI stream features)
        self.b0_att_s1 = B0SpatialAttention(in_ch=base_channels)
        self.b0_att_s2 = B0SpatialAttention(in_ch=base_channels * 2)
        self.b0_att_s3 = B0SpatialAttention(in_ch=base_channels * 4)

        # Bi-FPN style lateral projections to a common channel width
        fpn_ch = base_channels
        self.lat_s1 = nn.Conv2d(base_channels, fpn_ch, 1)
        self.lat_s2 = nn.Conv2d(base_channels * 2, fpn_ch, 1)
        self.lat_s3 = nn.Conv2d(base_channels * 4, fpn_ch, 1)
        # Top-down and bottom-up refinement convs
        self.td2 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.td1 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.bu2 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)
        self.bu3 = nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1)

        # Aggregate multi-scale features to full resolution and refine
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(fpn_ch * 3, fpn_ch * 3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_ch * 3, fpn_ch * 3, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        fuse_ch = fpn_ch * 3
        # Cost volume + aggregation for flow
        self.corr_radius = 2
        corr_ch = (2 * self.corr_radius + 1) ** 2
        self.corr_proj = nn.Conv2d(corr_ch, fpn_ch, 1)
        self.flow_gru = ConvGRUCell(in_ch=fuse_ch + fpn_ch, hidden_ch=fuse_ch)
        self.flow_offset = OffsetConv2dLite(fuse_ch, fuse_ch)
        self.flow_head = nn.Sequential(
            nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_ch, 2, 3, padding=1),
        )
        self.flow_act = nn.Tanh()

        # adaptive temperature for splatting
        self.temp_net = nn.Conv2d(fuse_ch, 1, 1)

        # MDN head for mass with uncertainty (mixture of Gaussians)
        self.mdn_components = 3
        mdn_out = self.mdn_components * 3  # pi, mu, sigma for each component
        self.mass_mdn = nn.Sequential(
            nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_ch, mdn_out, 1),
        )
        # Simple residual intensity refiner (kept)
        self.head_refine = ConvHead(fuse_ch, 1, act=None)

        # Maximum displacement scale (in pixels) applied to tanh-normalized flow
        self.max_disp_px = float(max_disp_px)
        # Optional scale for adding VDM prior in pixel units
        self.vdm_prior_scale = float(vdm_prior_scale)

    def forward(self, dwi_stack: torch.Tensor, t2_stack: torch.Tensor, vdm: torch.Tensor,
                vdm_prior_x: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x_dwi = torch.cat([dwi_stack, vdm], dim=1)
        s1d, s2d, s3d = self.enc_dwi(x_dwi)
        s1t, s2t, s3t = self.enc_t2(t2_stack)

        # Apply VDM-guided spatial attention onto DWI features per scale
        s1d = self.b0_att_s1(vdm, s1d)
        s2d = self.b0_att_s2(vdm, s2d)
        s3d = self.b0_att_s3(vdm, s3d)

        # Cross-modal fusion per scale
        z1 = self.cmf_s1(s1d, s1t)  # 1/2
        z2 = self.cmf_s2(s2d, s2t)  # 1/4
        z3 = self.cmf_s3(s3d, s3t)  # 1/8

        # Lateral projections
        p3 = self.lat_s3(z3)
        p2 = self.lat_s2(z2)
        p1 = self.lat_s1(z1)

        # Top-down pathway
        td3 = p3
        td2 = self.td2(p2 + F.interpolate(td3, size=p2.shape[-2:], mode="bilinear", align_corners=False))
        td1 = self.td1(p1 + F.interpolate(td2, size=p1.shape[-2:], mode="bilinear", align_corners=False))

        # Bottom-up pathway
        bu2_in = td2 + F.avg_pool2d(td1, kernel_size=2, stride=2)
        bu2 = self.bu2(bu2_in)
        bu3_in = td3 + F.avg_pool2d(bu2, kernel_size=2, stride=2)
        bu3 = self.bu3(bu3_in)

        # Aggregate to full resolution
        td1_up = F.interpolate(td1, scale_factor=2, mode="bilinear", align_corners=False)
        bu2_up = F.interpolate(bu2, scale_factor=4, mode="bilinear", align_corners=False)
        bu3_up = F.interpolate(bu3, scale_factor=8, mode="bilinear", align_corners=False)
        f = torch.cat([td1_up, bu2_up, bu3_up], dim=1)
        f = self.fuse_refine(f)

        # Build multi-scale correlations (1/2, 1/4, 1/8) and iteratively update per GRU step
        s1d_up1 = F.interpolate(s1d, size=f.shape[-2:], mode="bilinear", align_corners=False)
        s1t_up1 = F.interpolate(s1t, size=f.shape[-2:], mode="bilinear", align_corners=False)
        s2d_up = F.interpolate(s2d, size=f.shape[-2:], mode="bilinear", align_corners=False)
        s2t_up = F.interpolate(s2t, size=f.shape[-2:], mode="bilinear", align_corners=False)
        s3d_up = F.interpolate(s3d, size=f.shape[-2:], mode="bilinear", align_corners=False)
        s3t_up = F.interpolate(s3t, size=f.shape[-2:], mode="bilinear", align_corners=False)

        h = torch.zeros_like(f)
        flow_logits_acc = None
        for _ in range(3):
            c1 = build_corr_volume(s1d_up1, s1t_up1, radius=self.corr_radius)
            c2 = build_corr_volume(s2d_up, s2t_up, radius=self.corr_radius)
            c3 = build_corr_volume(s3d_up, s3t_up, radius=self.corr_radius)
            corr = c1 + 0.5 * c2 + 0.25 * c3
            corr_f = self.corr_proj(corr)
            gru_in = torch.cat([f, corr_f], dim=1)
            h = self.flow_gru(gru_in, h)
            h = self.flow_offset(h)
            step_flow_logits = self.flow_head(h)
            flow_logits_acc = step_flow_logits if flow_logits_acc is None else (flow_logits_acc + step_flow_logits)
        flow = self.flow_act(flow_logits_acc)  # [-1,1]
        mass_mdn_params = self.mass_mdn(f)
        # derive a mean mass for splatting via soft mixture expectation
        K = self.mdn_components
        pi, mu, sigma = torch.split(mass_mdn_params, [K, K, K], dim=1)
        pi = F.softmax(pi, dim=1)
        mu = torch.sigmoid(mu)  # constrain into [0,1]
        # expectation as mixture mean
        mass = (pi * mu).sum(dim=1, keepdim=True).clamp(0.0, 1.0)

        # scale flow to pixels and add optional VDM prior (correction field estimate)
        if vdm_prior_x is not None:
            # apply prior only on x channel to avoid unintended y drift
            prior_scaled = self.vdm_prior_scale * vdm_prior_x / self.max_disp_px
            flow_logits_acc[:, 0:1] = flow_logits_acc[:, 0:1] + prior_scaled[:, 0:1]
        flow_px = self.flow_act(flow_logits_acc) * self.max_disp_px

        # central distorted input
        dwi_c = dwi_stack[:, dwi_stack.shape[1] // 2 : dwi_stack.shape[1] // 2 + 1]
        # forward splat physical transport with adaptive temperature
        # Clamp temperature to a safe range to prevent splatting instability
        temperature = torch.clamp(F.softplus(self.temp_net(f)) + 0.1, min=0.1, max=5.0)
        T_phys = softmax_splat_2d(mass * dwi_c, flow_px, temperature=temperature)

        # refinement residual
        I_ref = self.head_refine(f)
        I_out = T_phys + I_ref

        return {
            "flow": flow_px,
            "mass": mass,
            "T_phys": T_phys,
            "I_ref": I_ref,
            "I_out": I_out,
            # expose logits for diagnostics
            "flow_logits": flow_logits_acc,
            # expose MDN raw params for loss
            "mass_mdn": mass_mdn_params,
            # expose temperature for diagnostics
            "temperature": temperature,
        }

