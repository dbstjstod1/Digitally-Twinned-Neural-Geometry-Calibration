# Digitally Twinned Neural Geometry Calibration

**Digitally Twinned Neural Geometry Calibration for CBCT via Learnable Projection Matrix Updates**

Sungho Yun and Seungryong Cho · KAIST MIR Lab

The proposed method estimates nine bounded, view-dependent geometry corrections
from a fixed reference volume and observed CBCT projections:

```text
view index → hash-grid MLP → 9 geometry corrections → projection matrix
                                                   ↓
                                  differentiable Joseph projection → image loss
```

All nine parameters are predicted per view: three intrinsic corrections, three
translations, and three rotations. The reference volume stays fixed. This
repository contains the method, the fixed reproduction recipes, and numerical
checks. Generated results, checkpoints, parameter searches, and writing assets
are excluded.

## Environment

Linux, an NVIDIA CUDA GPU, Python 3.11, and a CUDA toolkit compatible with PyTorch
are required. The recorded Joseph environment uses **PyTorch 2.8.0 / CUDA 12.8,
Triton 3.4.0, and tinycudann 2.0**. Python package versions are pinned in
[requirements.txt](requirements.txt).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation \
  "git+https://github.com/NVlabs/tiny-cuda-nn.git@749dd70c5afc5a9dadb85e5652ed65d55e0ba187#subdirectory=bindings/torch"
```

The tiny-cuda-nn revision matches the recorded installation. See the upstream
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v280)
and [tiny-cuda-nn build instructions](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)
for compiler and CUDA setup.

## Reproduction

Run commands from the repository root. The two acquisitions have different
coordinate conventions and recorded training settings; use their respective
entry points.

| Acquisition | Entry points | Configuration |
| --- | --- | --- |
| Measured Denseball | `geocal.train_real`, `geocal.export_real` | [Denseball preset](geocal/presets/denseball.py) |
| Circular / sineSpin simulation | `geocal.prepare`, `geocal.train`, `geocal.evaluate` | [Acquisition](examples/trajectories.json), [calibration](examples/calibration.json) |

### Measured Denseball

Supply little-endian float32, C-order raw data. The reference shape is
`(801, 929, 929)` in `(z, y, x)`; projections are `(480, 1264, 776)` in
`(view, row, column)`. Voxel pitch is 0.2 mm, detector pitch 0.228 mm,
SOD 443 mm, and SDD 650 mm. The preset records the remaining conventions.

```bash
CUDA_VISIBLE_DEVICES=0 python -m geocal.train_real 1 \
  --volume /data/reference.raw --projections /data/projections.raw \
  --out-dir result/measured
CUDA_VISIBLE_DEVICES=0 python -m geocal.export_real 1 \
  --volume /data/reference.raw --train-dir result/measured \
  --out-dir result/measured/export
```

The positional argument is the training view stride; export evaluates every
acquired view. The established measured-data recipe uses MONAI LNCC31,
100 epochs, Adam at 0.001, batch 4, seed 0, a 16-level hash grid, and the fixed
panel crop `u=26:776, v=0:1182`. Export restores geometry and correction bounds
from its checkpoint. Outputs include `P_updated_9DoF.npy`, the recovered
corrections, gantry `.dat` files, and simulated projections.

### Circular and noncircular simulation

Follow [the simulation instructions](examples/README.md) to prepare the same
35-ball phantom acquisition, run the fixed two-stage calibration, and evaluate
RPE, source positions, and all nine corrections. The sineSpin orbit is the
noncircular example; circular data provide the corresponding reference case.
Only data generation requires the independent LEAP build. Training uses the
included Triton implementation.

Raw acquisition data and the reference phantom are **not bundled**. Exact
paper reproduction requires those original inputs; the code does not silently
replace them with a different phantom. CUDA kernels may produce small numerical
differences across devices and builds.

## Code layout

```text
geocal/              method, geometry, projection, and command-line entry points
  models/            hash-grid encoder and nine-output MLP
  presets/           recorded measured-data acquisition
examples/            fixed simulation configurations and execution guide
patches/             independent LEAP Joseph projector build patch
tests/public/        coordinate, geometry, noise, and GPU workflow checks
```

The Joseph operator requires an NVIDIA GPU, cubic voxels, and equal x/y volume
dimensions. Simulation volumes use centered physical xyz; projection matrices
exported as `*_pixel.npy` map centered physical xyz to zero-based detector pixel
centers. `*_world_mm.npy` uses the projector's internal swapped-axis convention.
Use [coordinates.py](geocal/coordinates.py) for conversion.

## Verification

```bash
python -m unittest discover -s tests/public -v
python -m geocal.smoke --out-dir result/smoke --gpu 0
```

The small generated-volume smoke test exercises CUDA projection, gradients,
two-stage training, and export without external data. It is a software check,
not the paper's phantom experiment.

## Citation and license

```bibtex
@misc{yun_digitally_twinned_neural_geocal,
  author = {Yun, Sungho and Cho, Seungryong},
  title = {Digitally Twinned Neural Geometry Calibration for CBCT via Learnable Projection Matrix Updates}
}
```

Copyright (c) KAIST, MIR Lab (Medical Imaging and Radiotherapy Lab). All rights
reserved. Please contact the lab regarding usage, redistribution, and licensing
terms before external use.

The Joseph kernel follows [LLNL LEAP](https://github.com/LLNL/LEAP) (MIT).
The hash-grid encoder uses [NVlabs tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
Upstream components retain their respective licenses and credits.
