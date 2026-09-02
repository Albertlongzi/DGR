# SLURM templates

Portable job scripts — no absolute paths are baked in. Set `DGR_ROOT` (this repository) and
`DGR_DATA` (where the simulated pairs live), then submit:

```bash
export DGR_ROOT=/path/to/DGR
export DGR_DATA=/path/to/dgr_data
sbatch slurm/simulate_dwi_pairs.sbatch
```

Every tunable is an environment variable with a default, so a variation needs no edit:

```bash
EPOCHS=50 BATCH=4 sbatch slurm/train_stage2_diffusion.sbatch
```

Partition names, GRES strings and `--time` are site-specific; adjust the `#SBATCH` header
for your cluster.
