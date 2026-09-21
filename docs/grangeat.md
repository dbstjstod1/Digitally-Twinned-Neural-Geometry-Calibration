# Grangeat reconstruction: Sine Spin reference 19

`grangeat_recon.py` implements the analytical framework of
[Grangeat, *Mathematical framework of cone beam 3D reconstruction via the first
derivative of the Radon transform*](https://doi.org/10.1007/BFb0084509).
It replaces the previous iterative least-squares experiment as the reconstruction
path for the head comparison. `circular_fdk.py` supplies the circular FDK control.
Neither method iterates an image estimate, fits image intensities to the
reference, or clips negative reconstructed values.

This is an independent numerical implementation of the published framework.
[Jones et al.](https://doi.org/10.1117/1.JMI.11.4.043503) used proprietary Siemens
FDK and Grangeat-based implementations, whose additional processing is not
specified. Their calibrated poses and exact FOV rule are also unavailable.

## Implemented equations

Coordinates and distances are physical millimetres. For source `S`, unit plane
normal `n`, and unit source-to-detector direction `k`, define

```
g(S, omega) = integral_0^infinity f(S + t*omega) dt
R(n, p) = integral f(x) delta(p - n.x) dx
w(u,v) = D / sqrt(D^2 + u^2 + v^2), D = SDD
Q(alpha,s) = integral (w*g)(s*cos(alpha)-t*sin(alpha),
                           s*sin(alpha)+t*cos(alpha)) dt
h = sqrt((n.col)^2 + (n.row)^2)
alpha = atan2(n.row, n.col)
s0 = -D*(n.k)/h
R_p(n, n.S) = Q_s(alpha,s0)/h^2
f(x) = -1/(4*pi^2) * integral_hemisphere R_pp(n,n.x) dOmega
```

These correspond to original printed pp72–74, equations (2.7), (3.1), (3.3),
(3.4), with the detector derivative inside the line integral as in (4.1).
The coefficient above uses one hemisphere; the full-sphere coefficient is
`-1/(8*pi^2)`. Actual detector coordinates require SDD, not SOD, in the cosine
weight. The [public original scan](https://people.csail.mit.edu/bkph/courses/papers/Exact_Conebeam/Grangeat_Mathematical_Framework_2000.pdf)
contains the derivation and Joseph line-integration discussion on pp79–81.

For each target plane, the implementation first finds its source-path
intersections, as in pp88–89, equations (6.7)–(6.8). It interpolates adjacent
projection gradients at that target detector line, retaining the detector's
fine spatial sampling. It averages redundant intersections with weights summing
to one, differentiates in `p`, and performs Gauss–Legendre/periodic-azimuth
spherical integration on CUDA. Merely interpolating the derived samples
`R_p(n,n.S_view)` would introduce much coarser plane-offset sampling and visible
stripes; that ordering is not used in the final head reconstruction.
The 220-degree path is open: endpoints are never joined. Intermediate poses use
source chords and orthonormal detector axes; angular interpolation remains a
finite-sampling approximation rather than an additional acquired view.

FDK uses cosine preweighting, Parker short-scan redundancy, a zero-padded
Ram-Lak filter in detector millimetres and cone-weighted voxel backprojection.
The circular reconstruction is approximate away from the source plane.

## What “exact” requires

Exactness describes the continuous complete-data formula, not zero error on a
finite voxel grid. All required Radon planes and the data needed by the derivative
stencil must be available. Point visibility in every projection is insufficient
when a plane's detector line is truncated. See
[Clackdoyle and Noo, Section III.C](https://pmc.ncbi.nlm.nih.gov/articles/PMC7837589/).

The strict output requires every sampled angular contribution, an untruncated
detector-line diagnostic, and membership in the sampled-source hull interior.
The line diagnostic flags nonzero edge signal above `1e-5` in line-integral
units; it cannot prove the absence of unknown anatomy beyond a real detector.
The source hull is a necessary geometry condition, not the manufacturer's mask.
An integer count checks all directions, rather than a rounded coverage fraction.

Outputs distinguish the following cases:

- `recon_sinespin_220.npy`: values passing those checks; `NaN` elsewhere.
- `recon_sinespin_220_partial.npy`: sum of available complete-line angular
  contributions, without renormalizing for missing angles. This is diagnostic.
- The finite-detector estimate uses all measured lines and explicit zero
  continuation at detector edges, including the continuation's boundary
  derivatives. It is an approximation when projection data are truncated.
- Coverage arrays report missing angular data separately from image intensity.

No unsupported region is filled with reference CT or an iterative reconstruction.

## Reproduce

Use the Python dependencies in `requirements.txt` and the
[Joseph-pinned LEAP build](sinespin.md#run). DICOM preparation additionally needs
`pydicom` and `SimpleITK`; it preserves the native sheared DICOM lattice.

```bash
python prepare_cq500_head.py --data-root /path/to/CQ500 --head-shift-z-mm 0
python run_grangeat_head.py --gpu 1 --voxel 2 \
  --n-polar 128 --n-azimuth 512 --radon-step-mm 1
CUDA_VISIBLE_DEVICES=1 python validate_grangeat.py
python -m unittest discover -s tests -p 'test_*.py'
```

The head simulation uses a 0.5-mm Joseph forward grid and a 2-mm output grid.
The nominal panel is 397.936 × 292.908 mm, sampled at 646 × 476 pixels.
`--resume` verifies saved projection content, input, geometry and LEAP hashes
before reusing projections; the analytical reconstruction is recalculated.
Arrays and figures are Git-ignored; small numerical records remain in `docs/`.

An enlarged-detector control tests truncation independently of the source path:

```bash
python run_grangeat_head.py --gpu 1 --arms sinespin_220 \
  --n-polar 128 --n-azimuth 512 --detector-padding-factor 2 \
  --out-dir result_sinespin/cq500_grangeat_untruncated
```

This control doubles row/column counts while retaining pixel pitch and every
source/detector pose. Its panel is 795.872 × 585.816 mm. It is explicitly a
diagnostic virtual panel, **not the paper's equipment**. Source-plane gaps outside
the trajectory hull remain even with this larger panel.

Recompute the comparison table and file hashes from saved arrays on CPU:

```bash
python summarize_grangeat_head.py \
  --nominal-dir result_sinespin/cq500_grangeat \
  --control-dir result_sinespin/cq500_grangeat_untruncated \
  --summary-json docs/cq500_grangeat_summary.json
```

## Validation records

[Gaussian validation](grangeat_validation.json) uses an analytic translated,
anisotropic Gaussian to check cone data, first and second Radon derivatives,
absolute reconstruction scale and sampling convergence independently of Joseph.
The same fixed physical ROI is evaluated at every resolution; no gain is fitted.
The Gaussian's negligible noncompact tails are subject to the edge diagnostic.

| Views | Detector binning | Hemisphere quadrature | Radon spacing | Relative L2 error |
| --- | --- | --- | --- | --- |
| 546 | 8 | 12 × 48 | 2 mm | 2.833% |
| 1092 | 4 | 24 × 96 | 1 mm | 0.722% |
| 2184 | 2 | 32 × 128 | 0.5 mm | 0.181% |

All three levels cover the same 15,500-voxel ROI. At the finest level the
reconstructed peak is 99.724% of the analytic peak, without intensity fitting.
The independent CPU/GPU detector-derivative comparison has relative L2 error
below `1.2e-6`. The 34 CPU unit tests also pass.

## Centred CQ500 head result

The [numerical record](cq500_grangeat_summary.json) identifies the inputs,
projection library, geometry, reconstruction code and output-array hashes.
The case is CQ500CT283, selected before image inspection. DICOM geometry is
preserved, including the native gantry tilt; the head is placed at isocentre with
zero additional vertical shift. The nominal detector results use 546 acquired
views for both the circular 220° control and the Sine Spin protocol.

All entries below use **the same 42,097 voxels**: acquired reference CT,
every-view detector visibility in all three scans, membership in the Sine Spin
source hull, `z` in `[-70,-30)` mm, radius at most 80 mm, and reference HU in
`[-100,150]`. The hull only defines a common scoring region here; it does not
certify exactness.

| Nominal detector reconstruction | Skull-base soft-tissue RMSE |
| --- | --- |
| Circular 200° / 496 views, Parker FDK | 59.10 HU |
| Circular 220° / 546 views, Parker FDK | 53.78 HU |
| Sine Spin 220° / 546 views, Grangeat finite-detector estimate | 23.84 HU |

The Sine Spin result uses 128 × 512 quadrature directions and 1-mm Radon spacing.
With the **same measured projections**, the coarser 64 × 256 quadrature produced
27.22 HU RMSE on that identical ROI. These are integration resolutions, not
iterations. The full-head sagittal/coronal figure is
`result_sinespin/cq500_grangeat/head_grangeat_estimate.png`; it includes the vertex
and shows the reduced skull-base bands without hiding unsupported anatomy.
Its cyan outline is the source hull, not a claimed reproduction of Fig. 3's FOV.

The nominal panel has nonzero signal at its boundaries. **No voxel passes all
of this implementation's strict complete-line checks** for this head dataset;
the primary figure therefore explicitly shows the finite-detector estimate.
This does not establish that no other truncated-data reconstruction is possible.

The enlarged-panel diagnostic preserves all source/detector poses exactly. Its
central physical rays coincide with the nominal panel (projection-matrix offset
residual below `3e-10`); the separately computed Joseph data differ by relative
L2 `1.68e-6`, so they are not bit-identical. The enlarged-panel estimate has
23.76 HU RMSE on the same 42,097-voxel ROI. Its strict output retains 40,602 of
those voxels and has 23.83 HU RMSE on that smaller subset. Enlarging the detector
restores the detector-line completeness checks inside much of the source hull,
but does not fill missing source-plane support outside it. This control is not
the paper's equipment or the primary comparison.

These simulations support a reduction in cone-beam artefacts in this case and
validate the implemented analytic framework separately. They do not establish
that the proprietary clinical Fig. 9 processing or calibrated scanner geometry
has been reproduced.
