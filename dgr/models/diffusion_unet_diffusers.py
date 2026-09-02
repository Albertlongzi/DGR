import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from diffusers import UNet2DModel

from dgr.models.phc_e2e_mega_net import (
    ResidualBlock,
    DeformableCrossAttention,
    HybridTransformerBlock,
)


class MedicalImg2ImgUNet(UNet2DModel):
    """
    SR3-inspired diffusion UNet for joint low-b/ADC reconstruction.
    
    Training: Input [y_gamma (2ch), distorted pair (2ch), optional T2] -> predict 2-channel noise.
    Inference: Input [x_noisy (2ch), condition pair, optional T2] -> predict noise -> clean low-b & ADC.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 2,
        sample_size: int = 256,
        **kwargs,
    ) -> None:
        defaults = {
            "sample_size": sample_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "layers_per_block": 3,
            "block_out_channels": (256, 256, 512, 512, 768),
            "down_block_types": (
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            "up_block_types": (
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
    ):
        # Use all channels directly - no cross-attention complexity
        return super().forward(
            sample=sample,
            timestep=timestep,
            return_dict=return_dict,
        )


def create_medical_img2img_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    sample_size: int = 256,
    **kwargs,
) -> MedicalImg2ImgUNet:
    """Factory to create medical img2img UNet."""
    return MedicalImg2ImgUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        sample_size=sample_size,
        **kwargs,
    )


# ==============================================================================
# Legacy FusionConditioningModule (kept for backward compatibility with old ckpts)
# ==============================================================================
class FusionConditioningModule(nn.Module):
    """
    Lightweight modality fusion that combines distorted low-b/ADC inputs with T2 via
    deformable cross-attention + hybrid transformer refinement.
    (Legacy: kept for loading old checkpoints)
    """

    def __init__(
        self,
        dwi_channels: int = 2,
        t2_channels: int = 1,
        fusion_channels: int = 64,
        num_heads: int = 4,
        window: int = 16,
    ) -> None:
        super().__init__()
        self.out_channels = fusion_channels
        self.dwi_encoder = nn.Sequential(
            nn.Conv2d(dwi_channels, fusion_channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock(fusion_channels),
        )
        self.t2_encoder = nn.Sequential(
            nn.Conv2d(t2_channels, fusion_channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock(fusion_channels),
        )
        self.cross_attn = DeformableCrossAttention(fusion_channels, num_heads=num_heads, num_points=4)
        self.hybrid = HybridTransformerBlock(fusion_channels, num_heads=num_heads, window=window)
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, 1),
            nn.GELU(),
            ResidualBlock(fusion_channels),
        )
        self.refine = nn.Sequential(
            ResidualBlock(fusion_channels),
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
        )

    def forward(self, dwi_pair: torch.Tensor, t2_image: torch.Tensor) -> torch.Tensor:
        fd = self.dwi_encoder(dwi_pair)
        ft = self.t2_encoder(t2_image)
        fused = self.cross_attn(fd, ft)
        fused = self.hybrid(fused)
        fused = self.fuse(torch.cat([fused, ft], dim=1))
        return self.refine(fused)


class DiffusionUNetWithFusion(nn.Module):
    """
    Wrapper around UNet2DModel that consumes noisy latents together with fused
    distorted DWI + T2 conditioning features.
    (Legacy: kept for loading old checkpoints)
    """

    def __init__(
        self,
        fusion_channels: int = 64,
        sample_size: int = 320,
        noisy_channels: int = 2,
        t2_channels: int = 1,
        **unet_kwargs,
    ) -> None:
        super().__init__()
        self.noisy_channels = noisy_channels
        self.fusion = FusionConditioningModule(
            dwi_channels=2,
            t2_channels=t2_channels,
            fusion_channels=fusion_channels,
        )
        self.unet = MedicalImg2ImgUNet(
            in_channels=self.noisy_channels + fusion_channels,
            out_channels=2,
            sample_size=sample_size,
            **unet_kwargs,
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        dwi_pair: torch.Tensor,
        t2_image: Optional[torch.Tensor],
        timestep: Union[torch.Tensor, float, int],
    ):
        if t2_image is None:
            t2_image = torch.zeros(
                dwi_pair.shape[0],
                1,
                dwi_pair.shape[2],
                dwi_pair.shape[3],
                device=dwi_pair.device,
                dtype=dwi_pair.dtype,
            )
        h_cond = self.fusion(dwi_pair, t2_image)
        model_input = torch.cat([noisy_latent, h_cond], dim=1)
        return self.unet(model_input, timestep)


# ==============================================================================
# NEW: T2-Only Conditioning Module (simplified, no DWI/ADC conditioning)
# ==============================================================================
class T2OnlyConditioningModule(nn.Module):
    """
    Simplified T2-only conditioning: encode T2 with HybridTransformer only.
    No cross-modality attention or fusion layers needed since we only have T2.
    """

    def __init__(
        self,
        t2_channels: int = 1,
        fusion_channels: int = 64,
        num_heads: int = 4,
        window: int = 16,
    ) -> None:
        super().__init__()
        self.out_channels = fusion_channels
        # T2 encoder: conv -> residual -> hybrid transformer
        self.t2_encoder = nn.Sequential(
            nn.Conv2d(t2_channels, fusion_channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock(fusion_channels),
        )
        # Only HybridTransformer for spatial feature extraction
        self.hybrid = HybridTransformerBlock(fusion_channels, num_heads=num_heads, window=window)
        # Refine output
        self.refine = nn.Sequential(
            ResidualBlock(fusion_channels),
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
        )

    def forward(self, t2_image: torch.Tensor) -> torch.Tensor:
        ft = self.t2_encoder(t2_image)
        ft = self.hybrid(ft)
        return self.refine(ft)


class DiffusionUNetT2Only(nn.Module):
    """
    Diffusion UNet with T2-only conditioning.
    
    Training workflow:
      - Add noise to (dwi_gt, adc_gt) pair
      - Condition only on T2 (no distorted DWI/ADC)
      - Learn to denoise -> recover clean DWI/ADC prior given T2 anatomy
    
    Validation workflow:
      - Use external CNN (MageUltra) to get initial correction from (dwi_in, adc_in)
      - Add noise to CNN output
      - Denoise with T2 conditioning to refine the result
    """

    def __init__(
        self,
        fusion_channels: int = 64,
        sample_size: int = 320,
        noisy_channels: int = 2,  # (low-b, ADC)
        t2_channels: int = 1,
        num_heads: int = 4,
        window: int = 16,
        **unet_kwargs,
    ) -> None:
        super().__init__()
        self.noisy_channels = noisy_channels
        self.fusion_channels = fusion_channels
        
        # T2-only conditioning module
        self.t2_cond = T2OnlyConditioningModule(
            t2_channels=t2_channels,
            fusion_channels=fusion_channels,
            num_heads=num_heads,
            window=window,
        )
        
        # UNet: input = [noisy (2ch), T2 features (fusion_channels)]
        self.unet = MedicalImg2ImgUNet(
            in_channels=self.noisy_channels + fusion_channels,
            out_channels=2,  # predict 2-channel noise
            sample_size=sample_size,
            **unet_kwargs,
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,  # [B, 2, H, W] - noisy (low-b, ADC)
        t2_image: torch.Tensor,       # [B, 1, H, W] - T2 slice
        timestep: Union[torch.Tensor, float, int],
    ):
        """
        Args:
            noisy_latent: Noisy (low-b, ADC) pair to denoise
            t2_image: T2 conditioning image
            timestep: Diffusion timestep
        
        Returns:
            UNet output with predicted noise
        """
        # Handle None T2 (fallback to zeros)
        if t2_image is None:
            t2_image = torch.zeros(
                noisy_latent.shape[0],
                1,
                noisy_latent.shape[2],
                noisy_latent.shape[3],
                device=noisy_latent.device,
                dtype=noisy_latent.dtype,
            )
        
        # Get T2 conditioning features
        h_cond = self.t2_cond(t2_image)  # [B, fusion_channels, H, W]
        
        # Concatenate noisy input with T2 features
        model_input = torch.cat([noisy_latent, h_cond], dim=1)
        
        return self.unet(model_input, timestep)


# ==============================================================================
# NEW: T2 + CNN Output Conditioning Module (for refining CNN results)
# ==============================================================================
class T2AndCNNConditioningModule(nn.Module):
    """
    Conditioning module that combines T2 anatomical prior with CNN output.
    
    The CNN output tells the diffusion model "what the CNN already did",
    so it can focus on refining/correcting rather than reconstructing from scratch.
    
    Architecture:
      - T2 encoder -> T2 features
      - CNN output encoder -> CNN features  
      - Cross-attention: CNN features attend to T2 features (learn what to fix based on anatomy)
      - HybridTransformer for spatial refinement
      - Fuse and refine
    """

    def __init__(
        self,
        t2_channels: int = 1,
        cnn_channels: int = 2,  # CNN outputs (low-b, ADC)
        fusion_channels: int = 64,
        num_heads: int = 4,
        window: int = 16,
    ) -> None:
        super().__init__()
        self.out_channels = fusion_channels
        
        # T2 encoder
        self.t2_encoder = nn.Sequential(
            nn.Conv2d(t2_channels, fusion_channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock(fusion_channels),
        )
        
        # CNN output encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(cnn_channels, fusion_channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock(fusion_channels),
        )
        
        # Cross-attention: CNN features query T2 features
        # This helps the model understand "what anatomical context should guide the refinement"
        self.cross_attn = DeformableCrossAttention(fusion_channels, num_heads=num_heads, num_points=4)
        
        # HybridTransformer for spatial feature extraction
        self.hybrid = HybridTransformerBlock(fusion_channels, num_heads=num_heads, window=window)
        
        # Fuse T2 and CNN features
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, 1),
            nn.GELU(),
            ResidualBlock(fusion_channels),
        )
        
        # Final refinement
        self.refine = nn.Sequential(
            ResidualBlock(fusion_channels),
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
        )

    def forward(self, t2_image: torch.Tensor, cnn_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t2_image: [B, 1, H, W] T2 anatomical image
            cnn_output: [B, 2, H, W] CNN output (low-b, ADC)
        
        Returns:
            [B, fusion_channels, H, W] conditioning features
        """
        ft = self.t2_encoder(t2_image)      # [B, C, H, W]
        fc = self.cnn_encoder(cnn_output)   # [B, C, H, W]
        
        # CNN features attend to T2 (learn what to fix based on anatomy)
        fc_attn = self.cross_attn(fc, ft)   # [B, C, H, W]
        fc_attn = self.hybrid(fc_attn)      # [B, C, H, W]
        
        # Fuse with T2 features
        fused = self.fuse(torch.cat([fc_attn, ft], dim=1))  # [B, C, H, W]
        
        return self.refine(fused)


class DiffusionUNetT2AndCNN(nn.Module):
    """
    Diffusion UNet with T2 + CNN output conditioning.
    
    This model learns to REFINE CNN outputs rather than reconstruct from scratch.
    The CNN output provides "what has been done", T2 provides "anatomical guidance".
    
    Training workflow:
      1. Run frozen CNN on (dwi_in, adc_in) -> cnn_output
      2. Add noise to (dwi_gt, adc_gt) 
      3. Condition on (T2, cnn_output)
      4. Learn to denoise -> the model learns the residual between CNN output and GT
    
    Inference workflow:
      1. Run CNN on (dwi_in, adc_in) -> cnn_output
      2. Add noise to cnn_output (strength controls how much to refine)
      3. Denoise with (T2, cnn_output) conditioning
    """

    def __init__(
        self,
        fusion_channels: int = 64,
        sample_size: int = 320,
        noisy_channels: int = 2,  # (low-b, ADC)
        t2_channels: int = 1,
        cnn_channels: int = 2,    # CNN output channels
        num_heads: int = 4,
        window: int = 16,
        **unet_kwargs,
    ) -> None:
        super().__init__()
        self.noisy_channels = noisy_channels
        self.fusion_channels = fusion_channels
        
        # T2 + CNN conditioning module
        self.cond_module = T2AndCNNConditioningModule(
            t2_channels=t2_channels,
            cnn_channels=cnn_channels,
            fusion_channels=fusion_channels,
            num_heads=num_heads,
            window=window,
        )
        
        # UNet: input = [noisy (2ch), conditioning features (fusion_channels)]
        self.unet = MedicalImg2ImgUNet(
            in_channels=self.noisy_channels + fusion_channels,
            out_channels=2,  # predict 2-channel noise
            sample_size=sample_size,
            **unet_kwargs,
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,  # [B, 2, H, W] - noisy (low-b, ADC)
        t2_image: torch.Tensor,       # [B, 1, H, W] - T2 slice
        cnn_output: torch.Tensor,     # [B, 2, H, W] - CNN output
        timestep: Union[torch.Tensor, float, int],
    ):
        """
        Args:
            noisy_latent: Noisy (low-b, ADC) pair to denoise
            t2_image: T2 conditioning image
            cnn_output: CNN output (what the CNN already predicted)
            timestep: Diffusion timestep
        
        Returns:
            UNet output with predicted noise
        """
        # Handle None T2 (fallback to zeros)
        if t2_image is None:
            t2_image = torch.zeros(
                noisy_latent.shape[0],
                1,
                noisy_latent.shape[2],
                noisy_latent.shape[3],
                device=noisy_latent.device,
                dtype=noisy_latent.dtype,
            )
        
        # Handle None CNN output (fallback to zeros - pure T2 conditioning)
        if cnn_output is None:
            cnn_output = torch.zeros(
                noisy_latent.shape[0],
                2,
                noisy_latent.shape[2],
                noisy_latent.shape[3],
                device=noisy_latent.device,
                dtype=noisy_latent.dtype,
            )
        
        # Get conditioning features from T2 + CNN output
        h_cond = self.cond_module(t2_image, cnn_output)  # [B, fusion_channels, H, W]
        
        # Concatenate noisy input with conditioning features
        model_input = torch.cat([noisy_latent, h_cond], dim=1)
        
        return self.unet(model_input, timestep)
