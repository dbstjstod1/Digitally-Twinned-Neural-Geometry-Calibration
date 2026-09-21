# CQ500 head: Sine Spin sampling and display support

This numerical head experiment tests the skull-base artifact question in
[Jones et al., Fig. 9](https://doi.org/10.1117/1.JMI.11.4.043503), using a CQ500
CT as the reference object. It compares three nominal scan protocols; it is not
an exact reproduction of the manufacturer's geometry, algorithm, or phantom.

## Input and physical coordinates

**CQ500CT283** is the first held-out case in the existing local
`_cq500_index.json`, selected before image inspection. The split selects thin
series, sorts patients numerically, and reserves 150/50 patients for training/
validation before the test set. Selection details and the index hash are saved;
source data and training caches are unchanged.

The 240-slice series has approximately 16° gantry tilt. DICOM orientation, pixel
spacing, and actual `ImagePositionPatient` increments define its sheared affine;
GDCM only decodes HU pixels. Coordinates are LPS millimetres: x left, y posterior,
z superior; arrays are z/y/x, following the
[DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html).
Sagittal fixes x; coronal fixes y. Nearest central slices are x/y=−0.5 mm.

Reference-only inspection identified the skull base near −50 mm relative to the
series centre. A **+100 mm z translation**, fixed before reconstruction, places
it near +50 mm. Both grids share an isocentre and 256 × 256 × 384 mm box:

| Sampling | Voxel | Array shape, z/y/x |
| --- | --- | --- |
| Forward object | 0.5 mm | 768 × 512 × 512 |
| Reconstruction and reference | 1 mm | 384 × 256 × 256 |

Both linearly sample the same native attenuation field,
`mu = 0.02 * (max(HU, -1000) + 1000) / 1000` in mm⁻¹ without an upper HU clip.
Unknown anatomy is air padded and excluded from scoring. No acquired voxel
centre with HU >−500 falls outside the target box.

## Protocols and measurements

| Arm | Arc / views | Tilt |
| --- | --- | --- |
| Paper circular protocol | 200° / 496 | 0° |
| Matched circular control | 220° / 546 | 0° |
| Sine Spin | 220° / 546 | ±10°, one sinusoidal cycle |

All arms use the [detector model and Joseph-pinned LEAP build](sinespin.md):
assumed SOD/SDD 750/1200 mm, a centred 397.936 × 292.908 mm detector, and
646 × 476 numerical samples. Zero sine phase and the start angles are assumptions.
Nonnegative LEAP LS with SQS preconditioning starts from zero for **160 iterations
in 40-iteration blocks**. Both kernels use Joseph; backprojection is an approximate
adjoint. No intensity fitting or reconstruction mask is applied.

The primary ROI was fixed before reconstruction: **30≤z<70 mm, radius ≤80 mm,
reference HU −100 to 150**, intersected with acquired CT coverage and visibility
in every view of all three protocols. Scores include HU RMSE, MAE, signed bias,
and the fraction of errors below −50 HU, with additional common head/soft/bone
scores. The matched circular arm controls arc and view count. Check convergence
alongside final scores.

Figures show raw reconstructions with detector outlines and a separate
**display-only crop** using each arm's every-view detector mask. This is not
Tuy completeness or manufacturer Grangeat support, and is not fitted to 120 mm.
The common HU window is [−110, 210], with ±100 HU error maps; Fig. 9's 48/315 AU
window cannot be directly converted to CQ500 HU.

## Run

From the repository root, provide the CQ500 DICOM tree and its existing project
index. Preparation additionally requires SciPy, pydicom, and SimpleITK:

```bash
python -m pip install scipy pydicom SimpleITK
python prepare_cq500_head.py --data-root /path/to/CQ500 \
  --test-index 0 --head-shift-z-mm 100 \
  --out-dir result_sinespin/cq500_fig9/input
python -m unittest discover -s tests -p 'test_cq500_head.py'
```

The recorded CPU preparation used Python 3.8, SciPy 1.5.2, pydicom 2.4.4, and
SimpleITK 2.3.1. GPU reconstruction used Python 3.11 and PyTorch 2.8.0/CUDA 12.8.
GPU 1 was shared with another process during the later arms, so the recorded
elapsed times are not a controlled speed comparison.
In the GPU environment, first follow the [patched LEAP build instructions](sinespin.md#run),
including its `PYTHONPATH`, then run:

```bash
python sim_cq500_sinespin.py --input-dir result_sinespin/cq500_fig9/input \
  --out-dir result_sinespin/cq500_fig9 --gpu 1 --iterations 160 --check-every 40
```

Add `--resume` after an interruption. Atomic checkpoints store the volume and
history together after each block; geometry, input metadata, solver settings,
and the LEAP library must match. Completed arms are retained. `--arms` can select
individual protocols, and `--plots-only` regenerates figures from saved volumes.

Arrays, poses, metadata, and convergence histories stay under the Git-ignored
`result_sinespin/cq500_fig9/`. `head_metadata.json` records selection/transforms;
`metrics.json` records scores and the LEAP SHA256. Figures are `head_raw.png`,
`head_masked.png`, `head_skullbase.png`, and `head_skullbase_error.png`.

## Detector scale, image crop, and reconstruction support

The paper specifies a **398 × 293 mm detector**, 0.154 mm native pitch, and
2×2 acquisition binning. Our approximate 1292 × 951 acquisition grid at 0.308 mm
preserves that physical scale; additional numerical binning changes sampling,
not detector width or height. The long detector axis is transverse and the short
axis longitudinal, consistent with the paper's reconstructed dimensions:

| Volume | Physical box, x/y/z | Isotropic voxel |
| --- | --- | --- |
| Paper circular, Table 1 | 248 × 248 × 182 mm | 0.485 mm |
| Paper Sine Spin, Table 1 | 249 × 249 × 181 mm | 0.485 mm |
| Current numerical reconstruction | 256 × 256 × 384 mm | 1 mm |

At the assumed 750/1200 mm distances, the demagnified detector footprint at
isocentre is 248.71 × 183.07 mm; the sine orbit's all-view central z extent
is 181.72 mm. These agree in scale with Table 1, but do not establish identical
calibrated geometry. The paper does not publish these distances or exact integer
image matrices. A [Siemens ARTIS icono floor brochure](https://cdn0.scrvt.com/39b415fb07de4d9656c7b516d8e2d907/54121253d399c9cd/2bf0bbd113de/siemens-healthineers-at-hybrid-or-brochure.pdf)
shows a 120 cm SID, while [Siemens detector operation instructions](https://academy.siemens-healthineers.com/_/en-us/artis-icono-move-the-flat-detector-fd-usa/)
describe adjustable SID and portrait/landscape rotation. Neither confirms the
paper's specific biplane protocol or our assumed 750 mm source-to-isocentre
distance; matching the actual system requires its acquisition geometry.

**`head_skullbase.png` is an enlargement, not the full FOV:** its axes cover
x/y=−90…90 mm and z=10…90 mm, a 180 × 80 mm sagittal/coronal window. Moreover,
the +100 mm head translation places known reference voxels above −500 HU at
z=−1.5…183.5 mm. Only **47.44%** of those voxels are visible in every sine view.
Translating the same reference back by 100 mm raises this geometric coverage
to **98.08%** without changing the detector. The chosen placement tests the
off-plane skull base; it does not test whole-head coverage. `fov_audit.json`,
`head_fov_overview.png`, and `head_placement_fov.png` document the full scale and
placement comparison. The centred comparison is a reference-only visibility
calculation, not a new projection or reconstruction experiment.

Omitting a display mask leaves partially observed reconstruction values visible
outside the detector intersection. Applying it makes a sharp boundary, but that
boundary is not the manufacturer's reconstruction support.

The separate CPU diagnostic examines the convex hull of the sine source path,
whose interior meets the continuous-orbit plane-intersection condition, together
with the chosen all-view detector visibility condition:

```bash
python analyze_sinespin_support.py --out-dir result_sinespin/cq500_fig9
```

| Transverse position | Diagnostic longitudinal extent |
| --- | ---: |
| Centre | 119.32 mm |
| x=±65 mm, y=0 (coronal) | 118.86 mm |
| x=0, y=+65 mm (sagittal) | 97.68 mm |
| x=0, y=−65 mm (sagittal) | 134.05 mm |

At radius 65 mm the azimuthal range is 97.68–134.05 mm. Increasing the source
samples from 546 to 5,461 changes these boundaries by less than 0.001 mm.
This is a geometric reason a reconstruction-support rule can give a much smaller
region than detector visibility alone. The coronal 118.86 mm value must not be
identified with the paper's sagittal 120 mm measurement.

The source hull and point visibility do not guarantee exact reconstruction from
truncated projections; Grangeat also requires sufficient detector-plane data.
See [Clackdoyle and Noo's completeness analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7837589/).
Circular FDK is approximate and can reconstruct away from its source plane.
The paper does not state that the manufacturer applies a source-hull mask.
`support_geometry.png` and `.json` report this diagnostic separately; it changes
neither the head images nor the reconstruction scores.

## Interpretation and results

The CT lacks much of the paper phantom's jaw and neck. Excluding unknown coverage
from scoring does not remove artifacts caused by its cut-off boundary. Both
sampling grids share the same reference CT. Single-energy projections add no
noise or scatter and omit beam hardening: dental beam-hardening improvement,
clinical NPS, and exact equipment/Fig. 9 reproduction are not validated.

All three arms completed 160 iterations. The
[compact result record](cq500_summary.json) includes every checkpoint's scores,
geometry, library hash, and exported-volume hashes. The common primary ROI
contains 476,869 voxels.

| Measurement at 160 iterations | Circular 200° | Circular 220° | Sine 220° |
| --- | ---: | ---: | ---: |
| Skull-base soft-tissue MAE, HU | 14.83 | 14.46 | 10.58 |
| Skull-base soft-tissue RMSE, HU | 29.88 | 29.58 | 27.50 |
| Skull-base error below −50 HU, % | 1.816 | 1.755 | 1.712 |
| Common head RMSE, HU | 47.19 | 47.16 | 42.45 |
| Common soft-tissue RMSE, HU | 37.88 | 37.81 | 33.61 |
| Common bone RMSE, HU | 70.98 | 70.95 | 64.83 |
| Relative projection residual, % | 0.510 | 0.501 | 0.430 |

At matched arc and view count, Sine reduces primary-ROI MAE by **26.8%** and
RMSE by **7.0%**. The sagittal/coronal comparisons show reduced horizontal bands,
while bone-boundary errors remain. The below−50-HU fraction changes only slightly;
these results do not establish the near elimination of artifacts reported for
the paper's physical phantom and proprietary algorithms.

| Iterations | Circular 220° skull-base RMSE, HU | Sine 220° skull-base RMSE, HU |
| --- | ---: | ---: |
| 40 | 59.26 | 46.09 |
| 80 | 38.37 | 31.18 |
| 120 | 31.88 | 28.13 |
| 160 | 29.58 | 27.50 |

The error gap narrows with iteration. This validates an improvement for this
case at the stated iteration budget; full convergence and asymptotic superiority
are not established. Thirteen CPU tests passed. Independent NumPy scoring of
the saved volumes agrees with the GPU primary-ROI RMSE/MAE within 0.001 HU.
