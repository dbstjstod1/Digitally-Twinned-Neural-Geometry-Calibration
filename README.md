# AI-Geocal

**Digitally Twinned Neural Geometry Calibration for CBCT via Learnable Projection Matrix Updates**

Sungho Yun and Seungryong Cho · KAIST MIR Lab

AI-Geocal estimates a bounded, per-view **9-DoF geometric correction** from a
known reference volume and measured cone-beam CT projections. A hash-encoded MLP
updates an analytic nominal orbit and learns by matching projections with LNCC.
The nominal orbit requires scalar system geometry, but no initial gantry or
projection-matrix file.

```text
view index → hash MLP → 9-DoF correction → projection matrix → Joseph projection → LNCC
```

Training, export, and the direct per-view baseline use the same differentiable
**Joseph operator implemented in Triton**. Gradients are computed with respect to
geometry; the reference volume is fixed. Cubic voxels and `imsx == imsy` are
required. Calibration does not require a LEAP installation.

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
for compiler and CUDA setup. The former PyTorch 1.13 environment cannot run this operator.

## Data and geometry

Place these **headerless float32** files in the repository root, or pass their
locations with `--volume` and `--projections`:

| Input | Filename | Array shape |
| --- | --- | --- |
| Reference volume | `open_top_cylinder_ball_OD180_H160_wall3.0_bottom3.0_balldiam1.50_Ntheta24_zpitch20.00_929x929x801.float32.raw` | `(801, 929, 929)` = `(z, y, x)` |
| Measured projections | `Denseball_proj_480.raw` | `(480, 1264, 776)` = `(view, v, u)` |

Raw data are not distributed in this repository; reproducing the measured-data
experiment requires these inputs. Data, checkpoints, and generated results are
excluded from Git.

The shared [Denseball configuration](configs/denseball.py) specifies:

| Parameter | Value |
| --- | --- |
| Views / scan / start angle | 480 / 360° / 180° |
| Voxel / detector pixel pitch | 0.2 mm / 0.228 mm |
| SOD / SDD | 443 mm / 650 mm |
| Nominal `(k, un, vn)` | `(650, 34, 15)` mm |
| Orbit axis / direction / endpoint | world `y` / `-1` / excluded |
| Detector reversals / reconstruction type | `+1, +1` / `1` |
| Motion bounds | intrinsic translation ±10 mm, extrinsic translation ±10 mm, rotation ±10° |
| Training | 100 epochs, batch 4, Adam, learning rate `1e-3`, seed 0, AMP off |
| Loss | `1 + LNCC`, rectangular 31-pixel window, crop `v=0:1182, u=26:776` |

Coordinates use `INTERNAL(x,y,z) = WORLD(x,z,y)`; the reference grid starts at
`X0 = -imsx*dx/2`, `Y0 = -imsy*dy/2`, `Z0 = 0`. Intrinsic skew is fixed to zero.
The supplied loss crop is specific to this Denseball panel.

## Reproduce calibration

Run from the repository root and select the GPU assigned to your run. The examples
use GPU 1; change `CUDA_VISIBLE_DEVICES` to match your machine.

```bash
# Train on all 480 views.
CUDA_VISIBLE_DEVICES=1 python run_single_viewstep.py 1

# Export using the latest checkpoint and its saved geometry and motion bounds.
CUDA_VISIBLE_DEVICES=1 python run_single_sample.py 1

# Plot the recovered nine parameters (CPU).
python print_numpy.py result_denseball/joseph_mlp_vs1
```

Training writes `result_denseball/joseph_mlp_vs1/`; export writes its
`export/` subdirectory. Use a view step of `2`, `4`, or `8` to reproduce
the view-subsampling study. Export still evaluates all 480 views.

For custom locations:

```bash
CUDA_VISIBLE_DEVICES=1 python run_single_viewstep.py 8 \
  --volume /data/reference.raw --projections /data/projections.raw \
  --out-dir result_denseball/my_run
CUDA_VISIBLE_DEVICES=1 python run_single_sample.py \
  --volume /data/reference.raw --train-dir result_denseball/my_run \
  --out-dir result_denseball/my_export
```

`--checkpoint` selects an explicit checkpoint. Each runner supports `--help`.
`AI_Geocal.py` and `Sample.py` also delegate to these training and export CLIs.
Reduce `--batch-size` or export `--proj-batch` if GPU memory is limited.

| Output | Contents |
| --- | --- |
| `motion_model_ep*.pth` | Model, optimizer, geometry, bounds, ROI, and training settings |
| `loss_history.csv`, `training_time.txt` | Training loss and elapsed time |
| `motion_{ts_mm,tp_mm,rot_deg}.npy` | All-view learned motion, `(480, 3)` each |
| `Projections_{before,aligned}_9DoF.raw` | Nominal and corrected **synthetic** projections |
| `Gantry_{nominal,updated}_9DoF.dat` | Exported gantry geometry |

Export also saves projection matrices and motion arrays. It renders the known
volume at nominal/corrected geometry; it does not resample measured detector images.

## Direct per-view comparison

The research baseline optimizes nine parameters independently for each view,
using the same Joseph operator, geometry, bounds, and LNCC crop.

```bash
CUDA_VISIBLE_DEVICES=1 python run_direct_param.py run --niters 150 --lr 0.05
python run_direct_param.py merge
CUDA_VISIBLE_DEVICES=1 python compare_full.py
```

The comparison requires completed neural and direct runs. It scores both final
motion estimates with Joseph and saves a summary and comparison plots. Use
`--mlp-dir` and `--direct-dir` for custom runs. Historical results from other
projection discretizations should be rerun for this comparison.

## Code

| File | Role |
| --- | --- |
| `AI_Geocal.py`, `Sample.py` | Neural calibration and export |
| `geometry.py`, `configs/denseball.py` | Shared orbit, geometry, and reproduction settings |
| `DoF_transform.py` | Bounded 9-DoF projection-matrix updates |
| `fast_projectors.py` | Joseph forward operator and geometry gradients |
| `models/` | Hash-encoded motion network |
| `helpers.py` | Raw I/O and gantry geometry utilities |
| `AI_Geocal_direct.py`, `compare_full.py` | Direct per-view ablation and evaluation |
| `run_*.py`, `print_numpy.py` | Reproduction CLIs and motion plots |

## Noncircular geometry calibration

The [ball-phantom sineSpin experiment](docs/sinespin_calibration.md) tests whether
the existing 9-DoF hash MLP can recover a noncircular P-matrix trajectory from a
**circular initialization**. It fits independent Joseph projection images;
true poses and fixed ball IDs are used only for evaluation. Both trajectories
use the same 220° arc and 546 view angles. The current 0.2-mm phantom fits the
detector in every view. Targets include Poisson transmission noise at
**I₀ = 44,000**; the guide records input inspection, 9-DoF bounds and the
superseded Denseball diagnostic.

The current experiment uses **vanilla LNCC31 without pooling**, with the
existing 9-DoF MLP unchanged. In the historical zero-head initialization run,
with **10 mm / 10 mm / 15°** bounds and 100 epochs,
ball reprojection RMS falls from 11.46 to **5.30 px**; source-position RMS is
**42.40 mm**. The negative-tilt region remains inaccurate, so the complete
trajectory has not been recovered. The guide shows the
[nine final parameter estimates and bounds](docs/sinespin_ball_parameters.png).

The [supplied vanilla-code cross-check](docs/sinespin_vanilla_crosscheck.md)
restores the original network initialization and compares geometry, gradients
and convergence. Three original-initialization seeds still miss the same angle
interval. A separate image-initialization control recovers individual views,
but is not presented as a recovered neural trajectory or added to the default
training algorithm.

```bash
python run_sinespin_calibration.py prepare --gpu 1 \
  --volume phantom_density_v1_643x643x651.float32.raw \
  --shape-zyx 651 643 643 --voxel-mm 0.2 --i0 44000
python show_sinespin_calibration.py \
  --input-dir result_sinespin/ball_calibration/input \
  --out-dir result_sinespin/ball_calibration/input_preview
```

## Sine Spin reconstruction

The [analytical reconstruction](docs/grangeat.md) implements the Grangeat method
cited as reference 19 in the Sine Spin paper. Circular scans use Parker-weighted
FDK; Sine Spin uses weighted detector line derivatives, 3D Radon rebinning and
Radon inversion. **There are no reconstruction iterations.** Joseph generates
the synthetic cone-beam data.

```bash
# First build the Joseph-pinned LEAP library linked in the guide.
python prepare_cq500_head.py --data-root /data/CQ500
python run_grangeat_head.py --gpu 1 --n-polar 128 --n-azimuth 512
CUDA_VISIBLE_DEVICES=1 python validate_grangeat.py
```

The CQ500 head is centred at isocentre. Geometry follows the nominal
200°/496-view and 220°/546-view protocols with a ±10° sine tilt, assuming
SOD/SDD 750/1200 mm. The original detector size is preserved. The guide records
numerical convergence, detector truncation and missing-plane support separately;
the proprietary Siemens implementation is not reproduced.

Results stay in `result_sinespin/cq500_grangeat/`. The full-head
`head_grangeat_estimate.png` shows the finite-detector estimate and labels its
truncation assumption. Separate outputs mark where the complete-data conditions
fail. [Earlier Shepp–Logan IR](docs/sinespin.md) and
[CQ500 IR](docs/cq500_sinespin.md) results are retained as explicitly different
baselines, not as reproductions of the paper's reconstruction method.

## Citation and terms

If you use this code in research, please cite:

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
