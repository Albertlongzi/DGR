import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2


def gaussian_blur(x, k=31, sigma=7):
    """Simplified separable Gaussian kernel."""
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k // 2)
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 /= g1.sum()
    g1 = g1.view(1, 1, k, 1); g2 = g1.transpose(2, 3)
    x = F.conv2d(x, g1, padding=(k // 2, 0), groups=x.size(1))
    x = F.conv2d(x, g2, padding=(0, k // 2), groups=x.size(1))
    return x


def t2_lcn(t2):
    """Local Contrast Normalization of T2 contrast."""
    mu = gaussian_blur(t2, k=31, sigma=7)
    var = gaussian_blur((t2 - mu) ** 2, k=31, sigma=7)
    y = (t2 - mu) / (var.sqrt() + 1e-6)
    # Rescale to [0,1]
    y_min = y.amin(dim=(2, 3), keepdim=True)
    y_max = gaussian_blur(y.amax(dim=(2, 3), keepdim=True), k=31, sigma=7)
    y = (y - y_min) / (y_max - y_min + 1e-6)
    return y


def t2_canny(t2_tensor, low_thresh=50, high_thresh=150, gray_mode=False):
    """Extract Canny edges from T2 tensor. Returns binary or grayscale edge map."""
    batch_size = t2_tensor.shape[0]
    processed = []
    
    for i in range(batch_size):
        # Convert to numpy (H,W) for cv2
        img_np = t2_tensor[i, 0].detach().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Canny edge detection
        edges = cv2.Canny(img_np, low_thresh, high_thresh)
        
        if gray_mode:
            # Keep grayscale edges
            edges = edges.astype(np.float32) / 255.0
        else:
            # Binary edges
            edges = (edges > 0).astype(np.float32)
        
        processed.append(edges)
    
    # Stack back to tensor [B,1,H,W]
    result = torch.from_numpy(np.stack(processed)).unsqueeze(1).to(t2_tensor.device)
    return result


def process_t2_conditioning(t2_tensor, method="none", **kwargs):
    """
    Apply T2 contrast modification conditioning.
    
    Args:
        t2_tensor: [B,C,H,W] T2 tensor tensor
        method: "none", "lcn", "canny_binary", "canny_grayscale"
        **kwargs: additional params for canny (low_thresh, high_thresh)
    
    Returns:
        processed_t2_tensor: [B,1,H,W]
    """
    if method == "none":
        return t2_tensor[:, :1]  # Keep only first channel
    
    elif method == "lcn":
        t2_processed = t2_lcn(t2_tensor[:, :1])
        return t2_processed
    
    elif method == "canny_binary":
        return t2_canny(t2_tensor[:, :1], gray_mode=False, **kwargs)
    
    elif method == "canny_grayscale":
        return t2_canny(t2_tensor[:, :1], gray_mode=True, **kwargs)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description="T2 Conditioning Utilities")
    parser.add_argument("--tensor_shape", type=int, nargs=4, default=[1, 1, 256, 256])
    parser.add_argument("--method", type=str, default="none", 
                       choices=["none", "lcn", "canny_binary", "canny_grayscale"])
    parser.add_argument("--low_thresh", type=int, default=50, help="Canny low threshold")
    parser.add_argument("--high_thresh", type=int, default=150, help="Canny high threshold")
    parser.add_argument("--output_path", type=str, help="Save processed tensor to file")
    
    args = parser.parse_args()
    
    # Create dummy tensor for testing
    dummy_t2 = torch.randn(*args.tensor_shape)
    processed = process_t2_conditioning(
        dummy_t2, 
        method=args.method,
        low_thresh=args.low_thresh,
        high_thresh=args.high_thresh
    )
    
    print(f"Input shape: {dummy_t2.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Output range: [{processed.min():.4f}, {processed.max():.4f}]")
    
    if args.output_path:
        torch.save({"processed_t2": processed, "method": args.method}, args.output_path)
        print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()


