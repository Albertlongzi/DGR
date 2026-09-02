import torch
import torch.nn.functional as F


def softmax_splat_2d(src: torch.Tensor, flow: torch.Tensor, temperature: torch.Tensor = 1.0) -> torch.Tensor:
    """
    Softmax splatting (Niklaus & Liu) simplified 2D version.

    Args:
        src: [B,1,H,W] source intensities multiplied by mass m in [0,1]
        flow: [B,2,H,W] forward flow (u_x,u_y) in pixels (x:cols, y:rows)
        temperature: softmax temperature

    Returns:
        tgt: [B,1,H,W] accumulated target image.
    """
    b, c, h, w = src.shape
    assert c == 1, "src must be single-channel"
    fx = flow[:, 0:1]
    fy = flow[:, 1:2]

    # compute normalized target coordinates
    device, dtype = src.device, src.dtype
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    x_t = x + fx.squeeze(1)
    y_t = y + fy.squeeze(1)

    # 4-neighbor bilinear splat: compute integer corners and fractional weights
    x0 = torch.floor(x_t)
    y0 = torch.floor(y_t)
    x1 = x0 + 1
    y1 = y0 + 1

    wx1 = (x_t - x0).clamp(0, 1)
    wy1 = (y_t - y0).clamp(0, 1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    # weights for 4 corners
    w00 = (wx0 * wy0).unsqueeze(1)
    w10 = (wx1 * wy0).unsqueeze(1)
    w01 = (wx0 * wy1).unsqueeze(1)
    w11 = (wx1 * wy1).unsqueeze(1)

    # softmax attention over the 4 corners per pixel
    # stack weights and apply temperature softmax
    W = torch.stack([w00, w10, w01, w11], dim=1)  # [B,4,1,H,W]
    if isinstance(temperature, torch.Tensor):
        # broadcast temperature map [B,1,H,W] to [B,4,1,H,W]
        temp = temperature.to(device=src.device, dtype=src.dtype)
        temp = temp.clamp_min(1e-6).unsqueeze(1)
        A = torch.softmax(W / temp, dim=1)
    else:
        # scalar path
        t = max(1e-6, float(temperature))
        A = torch.softmax(W / t, dim=1)
    w00, w10, w01, w11 = A[:, 0], A[:, 1], A[:, 2], A[:, 3]  # [B,1,H,W]

    # scatter-add into target image
    tgt = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

    def splat_at(xi, yi, wgt):
        xi = xi.clamp(0, w - 1)
        yi = yi.clamp(0, h - 1)
        idx = yi.long() * w + xi.long()  # [B,H,W]
        val = (src * wgt).reshape(b, 1, h * w)
        out = tgt.reshape(b, 1, h * w)
        out.scatter_add_(2, idx.reshape(b, 1, h * w), val)

    splat_at(x0, y0, w00)
    splat_at(x1, y0, w10)
    splat_at(x0, y1, w01)
    splat_at(x1, y1, w11)

    return tgt

@torch.no_grad()
def bilinear_splat_2d(src: torch.Tensor, flow: torch.Tensor):
    """
    纯双线性 forward splat（硬搬运, 不归一化）
    src:  [B,1,H,W]
    flow: [B,2,H,W]  (x=cols, y=rows)
    return: tgt, cov  (都为 [B,1,H,W])
    """
    b, c, h, w = src.shape
    assert c == 1
    fx = flow[:, 0:1]
    fy = flow[:, 1:2]

    device, dtype = src.device, src.dtype
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    x_t = x + fx.squeeze(1)
    y_t = y + fy.squeeze(1)

    # 四邻域坐标
    x0 = torch.floor(x_t); y0 = torch.floor(y_t)
    x1 = x0 + 1;          y1 = y0 + 1
    wx1 = (x_t - x0).clamp(0, 1); wy1 = (y_t - y0).clamp(0, 1)
    wx0 = 1.0 - wx1;            wy0 = 1.0 - wy1

    w00 = (wx0 * wy0).unsqueeze(1)
    w10 = (wx1 * wy0).unsqueeze(1)
    w01 = (wx0 * wy1).unsqueeze(1)
    w11 = (wx1 * wy1).unsqueeze(1)

    # 目标索引（四个角）
    def _clamp_idx(xi, yi):
        xi = xi.clamp(0, w - 1).long()
        yi = yi.clamp(0, h - 1).long()
        return yi * w + xi  # [B,H,W]

    idx00 = _clamp_idx(x0, y0)
    idx10 = _clamp_idx(x1, y0)
    idx01 = _clamp_idx(x0, y1)
    idx11 = _clamp_idx(x1, y1)

    tgt = torch.zeros((b,1,h*w), device=device, dtype=dtype)
    cov = torch.zeros_like(tgt)

    def _splat(idx, wgt):
        val = (src * wgt).reshape(b,1,h*w)
        tgt.scatter_add_(2, idx.reshape(b,1,h*w), val)
        cov.scatter_add_(2, idx.reshape(b,1,h*w), wgt.reshape(b,1,h*w))

    _splat(idx00, w00)
    _splat(idx10, w10)
    _splat(idx01, w01)
    _splat(idx11, w11)

    return tgt.view(b,1,h,w), cov.view(b,1,h,w)
