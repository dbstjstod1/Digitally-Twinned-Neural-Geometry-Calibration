# Digitally Twinned Neural Geometry Calibration

**Digitally Twinned Neural Geometry Calibration for CBCT via Learnable Projection Matrix Updates**

Sungho Yun and Seungryong Cho · KAIST MIR Lab

Reference-based neural geometry calibration for cone-beam CT. Given a known
reference volume, observed projections, and nominal acquisition geometry, the
proposed method learns view-dependent corrections to the projection matrices
through differentiable forward projection.

## Method

A hash-grid MLP maps each view index to nine bounded geometry corrections:
three intrinsic parameters (a shared focal length and two principal-point
coordinates), three translations, and three rotations. These corrections update
the nominal projection matrix for each view. A differentiable Joseph projector
renders the fixed reference volume under the corrected geometry, and an image
similarity objective compares the rendered and observed projections.

```text
view index → hash-grid MLP → geometry corrections → corrected projection matrix
                                                             ↓
reference volume ───────────────────────────→ differentiable projection
                                                             ↓
observed projections ───────────────────────────────→ image similarity
```

All nine corrections are estimated per view. The reference volume remains fixed;
training updates the geometry model. The nominal geometry provides the starting
trajectory, allowing the same correction formulation to be used with circular
and noncircular acquisitions.

## Inputs and outputs

| Input | Description |
| --- | --- |
| Reference volume | Known attenuation volume with specified voxel spacing and coordinate frame |
| Observed projections | Projection images indexed by acquisition view |
| Nominal geometry | Initial projection matrices or system geometry used to construct them |
| Acquisition metadata | Detector dimensions and pixel spacing, view ordering, and coordinate conventions |

The method produces corrected projection matrices and nine parameter corrections
for every requested view, together with model checkpoints and training logs.
Simulation evaluation additionally reports reprojection and source-position
errors when ground-truth geometry is available. Ground truth is excluded from
the calibration loss.

Detector dimensions, projection count, image crop, and parameter bounds belong
to the acquisition configuration. They do not define the calibration method.
Image regions and training settings should be chosen for the available reference
and observed data.

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

## Getting started

Run commands from the repository root. Start with the generated-volume software
check below, or follow a complete paper reproduction example:

| Example | Instructions | Entry points |
| --- | --- | --- |
| Measured acquisition | [Denseball reproduction](examples/measured.md) | `geocal.train_real`, `geocal.export_real` |
| Circular and noncircular acquisition | [Simulation reproduction](examples/README.md) | `geocal.prepare`, `geocal.train`, `geocal.evaluate` |

These examples record the data sizes, crops, scanner geometry, and training
settings used for their respective experiments. The command-line entry points
currently implement those reproduction workflows. Supplying a new data path
alone does not configure a different scanner or acquisition.

For another acquisition, provide its reference-volume and detector metadata,
construct nominal matrices in the matching coordinate convention, and adapt the
input preparation and loss region. The reusable geometry update and projection
modules are listed below. Parameter bounds and any intrinsic prior should reflect
the uncertainty in that acquisition.

Raw acquisition data and the paper's reference phantom are not bundled. The
reproduction guides specify the required inputs. CUDA kernels may produce small
numerical differences across devices and builds.

## Implementation

| Component | Code |
| --- | --- |
| View-dependent geometry model | [models/](geocal/models/) |
| Nine-parameter geometry update | [transforms.py](geocal/transforms.py) |
| Differentiable Joseph projector | [projector.py](geocal/projector.py) |
| Projection-matrix coordinate conversion | [coordinates.py](geocal/coordinates.py) |
| Simulation training and evaluation | [train.py](geocal/train.py), [evaluate.py](geocal/evaluate.py) |
| Acquisition-specific presets | [presets/](geocal/presets/), [examples/](examples/) |

The current Joseph implementation requires an NVIDIA GPU, isotropic voxels, and
equal x/y volume dimensions. Coordinate origins, axis conventions, and detector
pixel-center definitions must agree across the reference and projection data.
In the simulation workflow, `*_pixel.npy` maps centered physical xyz to zero-based
detector pixel centers; `*_world_mm.npy` uses the projector's swapped-axis
coordinate convention. Calibration uses the included Triton projector; LEAP is
needed only for independent data generation in the simulation example.

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
