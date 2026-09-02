"""DGR — distortion-guided restoration for prostate diffusion MRI.

Two halves:
  * ``dgr.physics``  — forward susceptibility-distortion simulator (B0 -> EPI warp)
  * ``dgr.models`` / ``dgr.inference`` — the reverse (restoration) network stack
"""

__version__ = "0.1.0"
