from typing import Dict, Optional

import torch
import torch.nn.functional as F

from dgr.utils.warp import grid_warp_x, tv_loss, jacobian_det_penalty_1d_x, grid_warp_2d
from dgr.utils.splat import softmax_splat_2d


class ReconLoss(torch.nn.Module):
    def __init__(self, ssim_weight: float = 0.2):
        super().__init__()
        self.ssim_w = ssim_weight
        try:
            from monai.losses import SSIMLoss  # type: ignore
            self._ssim = SSIMLoss(spatial_dims=2)
        except Exception:
            self._ssim = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = (pred - target).abs().mean()
        if self._ssim is not None:
            ssim_term = self._ssim(pred, target)
        else:
            mu_p = torch.nn.functional.avg_pool2d(pred, 7, stride=1, padding=3)
            mu_t = torch.nn.functional.avg_pool2d(target, 7, stride=1, padding=3)
            var_p = torch.nn.functional.avg_pool2d((pred - mu_p) ** 2, 7, stride=1, padding=3)
            var_t = torch.nn.functional.avg_pool2d((target - mu_t) ** 2, 7, stride=1, padding=3)
            ssim_term = (var_p - var_t).abs().mean()
        return (1 - self.ssim_w) * l1 + self.ssim_w * ssim_term


def _masked_mean(v: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    if w is None:
        return v.mean()
    # ensure shape broadcastable to [B,1,H,W]
    w = w.clamp_min(0.0)
    num = (v * w).sum()
    den = w.sum().clamp_min(1e-8)
    return num / den


def forward_consistency_loss(I_out: torch.Tensor, disp_x: torch.Tensor, I_dist: torch.Tensor,
                             weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Forward consistency: warp I_out using disp_x (pixels) and compare to distorted input.
    disp_x shape: [B,1,H,W]
    """
    I_out_warp = grid_warp_x(I_out, disp_x)
    diff = (I_out_warp - I_dist).abs()
    return _masked_mean(diff, weight)


def gating_regularizer(gate: torch.Tensor, l1_weight: float = 1e-3, tv_weight: float = 1e-3) -> torch.Tensor:
    return l1_weight * gate.mean() + tv_weight * tv_loss(gate)


def heteroscedastic_nll(pred: torch.Tensor, target: torch.Tensor, sigma: torch.Tensor,
                        weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    var = sigma * sigma + 1e-6
    nll_map = 0.5 * ((pred - target) ** 2 / var + torch.log(var))
    return _masked_mean(nll_map, weight)


def mdn_negative_log_likelihood(mdn_params: torch.Tensor, target: torch.Tensor,
                                weight: Optional[torch.Tensor], num_components: int) -> torch.Tensor:
    """MDN NLL for per-pixel scalar target in [0,1]. mdn_params: [B, 3K, H, W]."""
    K = num_components
    pi, mu, sigma = torch.split(mdn_params, [K, K, K], dim=1)
    # mixture weights
    pi = torch.softmax(pi, dim=1)
    # constrain mu to [0,1]
    mu = torch.sigmoid(mu)
    # positive sigma
    sigma = F.softplus(sigma) + 1e-6
    # expand target to [B,K,H,W]
    t = target.expand_as(mu)
    # Gaussian pdf
    log_prob = -0.5 * ((t - mu) ** 2) / (sigma ** 2) - torch.log(sigma) - 0.5 * torch.log(torch.tensor(2 * 3.14159265, device=mdn_params.device, dtype=mdn_params.dtype))
    # log-sum-exp over components with weights pi
    log_mix = torch.log((pi * torch.exp(log_prob)).sum(dim=1, keepdim=True) + 1e-12)
    nll_map = -log_mix
    return _masked_mean(nll_map, weight)


def phc_total_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                   w: Dict[str, float]) -> torch.Tensor:
    # build pixel weights from confidence and optional valid mask
    weight_map = batch.get("loss_mask", None)
    if weight_map is None:
        weight_map = batch.get("valid_mask", None)
    if weight_map is None:
        weight_map = batch.get("conf_b0", None)
    # reconstruction
    l_rec = _masked_mean((outputs["I_out"] - batch["dwi_gt"]).abs(), weight_map) * w.get("rec", 1.0)
    # closed-loop: primary push (splat) with mass, auxiliary pull (warp) for stability
    tau = float(w.get("fwd_temp", 2.0))
    # allow gradients to flow into mass; stop gradients through I_out to avoid trivial shortcut
    mass = outputs["mass"]
    i_in_hat_push = softmax_splat_2d(mass * outputs["I_out"].detach(), -outputs["flow"], temperature=tau)
    l_fwd_push = _masked_mean((i_in_hat_push - batch["dwi_in"]).abs(), weight_map) * w.get("fwd", 0.2)
    i_in_hat_pull = grid_warp_2d(outputs["I_out"], -outputs["flow"])  # small stability term
    l_fwd_pull = _masked_mean((i_in_hat_pull - batch["dwi_in"]).abs(), weight_map) * w.get("fwd_pull", 0.0)
    # regularizers
    # apply TV to both x and y components
    l_tv_u = tv_loss(outputs["flow"]) * w.get("tv_u", 1e-3)  # type: ignore[attr-defined]
    l_tv_m = tv_loss(outputs["mass"]) * w.get("tv_m", 1e-3)
    # 1D jacobian along x for stability（保留原约束）
    l_jac = jacobian_det_penalty_1d_x(outputs["flow"][:, 0:1]) * w.get("jac", 1e-3)  # type: ignore[attr-defined]
    # coverage regularization: encourage splat(mass, flow) ~= 1
    cov_map = softmax_splat_2d(mass, outputs["flow"], temperature=tau)
    l_cov = (cov_map - 1.0).abs().mean() * w.get("cov", 0.0)
    # small L2 on residual refiner to discourage it from absorbing all reconstruction
    l_ref_l2 = (outputs["I_ref"] ** 2).mean() * w.get("ref_l2", 1e-3)
    # optional MDN NLL for mass if provided
    l_mdn = 0.0
    if "mass_mdn" in outputs:
        l_mdn = mdn_negative_log_likelihood(outputs["mass_mdn"], outputs["mass"].detach(), weight_map, num_components=3) * w.get("mdn", 0.0)
    # optional prior alignment on x component during early epochs
    l_prior = 0.0
    if "vdm_prior" in batch and w.get("prior", 0.0) > 0.0:
        # batch['vdm_prior'] is already a correction field (−VDM). Align flow directly to it.
        prior_x = batch["vdm_prior"][:, 0:1]  # [B,1,H,W]
        l_prior = (outputs["flow"][:, 0:1] - prior_x).abs().mean() * w.get("prior", 0.0)
    # small y-component penalty to stabilize (PE axis assumed x)
    l_flow_y0 = outputs["flow"][:, 1:2].abs().mean() * w.get("flow_y0", 1e-4)
    total = l_rec + l_fwd_push + l_fwd_pull + l_tv_u + l_tv_m + l_jac + l_cov + l_mdn + l_prior + l_ref_l2 + l_flow_y0
    return total


def phc_loss_components(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                        w: Dict[str, float]) -> Dict[str, torch.Tensor]:
    weight_map = batch.get("loss_mask", None)
    if weight_map is None:
        weight_map = batch.get("valid_mask", None)
    if weight_map is None:
        weight_map = batch.get("conf_b0", None)
    l_rec = _masked_mean((outputs["I_out"] - batch["dwi_gt"]).abs(), weight_map) * w.get("rec", 1.0)
    tau = float(w.get("fwd_temp", 2.0))
    mass = outputs["mass"]
    i_in_hat_push = softmax_splat_2d(mass * outputs["I_out"].detach(), -outputs["flow"], temperature=tau)
    l_fwd_push = _masked_mean((i_in_hat_push - batch["dwi_in"]).abs(), weight_map) * w.get("fwd", 0.2)
    i_in_hat_pull = grid_warp_2d(outputs["I_out"], -outputs["flow"])  # stability term
    l_fwd_pull = _masked_mean((i_in_hat_pull - batch["dwi_in"]).abs(), weight_map) * w.get("fwd_pull", 0.0)
    l_tv_u = tv_loss(outputs["flow"]) * w.get("tv_u", 1e-3)  # type: ignore[attr-defined]
    l_tv_m = tv_loss(outputs["mass"]) * w.get("tv_m", 1e-3)
    l_jac = jacobian_det_penalty_1d_x(outputs["flow"][:, 0:1]) * w.get("jac", 1e-3)  # type: ignore[attr-defined]
    cov_map = softmax_splat_2d(mass, outputs["flow"], temperature=tau)
    l_cov = (cov_map - 1.0).abs().mean() * w.get("cov", 0.0)
    l_mdn = 0.0
    if "mass_mdn" in outputs:
        l_mdn = mdn_negative_log_likelihood(outputs["mass_mdn"], outputs["mass"].detach(), weight_map, num_components=3) * w.get("mdn", 0.0)
    l_prior = 0.0
    if "vdm_prior" in batch and w.get("prior", 0.0) > 0.0:
        prior_x = batch["vdm_prior"][:, 0:1]
        l_prior = (outputs["flow"][:, 0:1] - prior_x).abs().mean() * w.get("prior", 0.0)
    # small L2 on residual refiner (same as in total loss)
    l_ref_l2 = (outputs["I_ref"] ** 2).mean() * w.get("ref_l2", 1e-4)
    l_flow_y0 = outputs["flow"][:, 1:2].abs().mean() * w.get("flow_y0", 1e-4)
    total = l_rec + l_fwd_push + l_fwd_pull + l_tv_u + l_tv_m + l_jac + l_cov + l_mdn + l_prior + l_ref_l2 + l_flow_y0
    return {
        "total": total,
        "l_rec": l_rec,
        "l_fwd_push": l_fwd_push,
        "l_fwd_pull": l_fwd_pull,
        "l_tv_u": l_tv_u,
        "l_tv_m": l_tv_m,
        "l_jac": l_jac,
        "l_cov": l_cov,
        "l_mdn": torch.as_tensor(l_mdn) if not isinstance(l_mdn, torch.Tensor) else l_mdn,
        "l_prior": torch.as_tensor(l_prior) if not isinstance(l_prior, torch.Tensor) else l_prior,
        "l_ref_l2": torch.as_tensor(l_ref_l2) if not isinstance(l_ref_l2, torch.Tensor) else l_ref_l2,
        "l_flow_y0": torch.as_tensor(l_flow_y0) if not isinstance(l_flow_y0, torch.Tensor) else l_flow_y0,
    }

