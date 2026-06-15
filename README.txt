================================================================================
AI-Geocal : Self-Supervised Geometric Calibration / Motion Estimation
            for Cone-Beam CT via a Differentiable Forward Projector
================================================================================

Overview
--------------------------------------------------------------------------------
This package estimates per-view geometric calibration / motion parameters of a
cone-beam CT (CBCT) system directly from measured projection data, without
requiring an initial gantry/calibration file.

A small neural network ("motion field") predicts a bounded correction
(translation, rotation, and optional intrinsic skew) for every view index. The
correction is applied to an analytically generated nominal scan orbit to obtain
updated projection matrices. A fully differentiable ray-marching forward
projector then renders synthetic projections from a known reference volume, and
the network is trained self-supervised by matching the synthetic projections to
the measured ones with a Local Normalized Cross-Correlation (LNCC) loss.

Because every stage (nominal orbit build -> DoF transform -> forward
projection -> loss) is differentiable, gradients flow back to the motion network
and the calibration is recovered by gradient descent.

Pipeline (high level):

    view index  ->  MotionNetHash (MLP + hash encoding)
                ->  raw 10-DoF parameters
                ->  bounded (ts, tp, rot, skew)        [motion10_to_ts_tp_rot_skew]
                ->  apply correction to nominal P      [apply_9DoF_transform_effective]
                ->  differentiable forward projection  [sinoproj_rdsh_pinv_raycast_dominant]
                ->  LNCC( rendered , measured )
                ->  backprop -> update MotionNet

Degrees of freedom (10-DoF default):
    ts   (3) : detector / intrinsic-side translation (cu, cv, f), in mm
    tp   (3) : extrinsic translation (tx, ty, tz), in mm
    rot  (3) : extrinsic rotation (rx, ry, rz), in degrees
    skew (1) : intrinsic skew (gamma), in degrees

6-DoF and 9-DoF motion models and a 10-DoF variant with a single global
intrinsic correction shared across views are also provided.

For a pure 9-DoF formulation (ts, tp, rot only, no intrinsic skew), set
skew_max = 0 so the skew DoF is fixed at zero and contributes no correction.


Requirements
--------------------------------------------------------------------------------
- Python 3.9+
- PyTorch (CUDA build strongly recommended; the projector and training are
  GPU-heavy)
- NumPy
- matplotlib                (plotting of recovered motion curves)
- MONAI                     (LocalNormalizedCrossCorrelationLoss)
- tinycudann (tcnn)         (CUDA Instant-NGP hash encoder used by MLP_hash)

Notes:
- tinycudann must be built against your CUDA / PyTorch version. If you cannot
  build it, hash_encoder.py also contains a pure-PyTorch HashEmbedder that can
  be substituted (slower).
- A CUDA-capable GPU with enough memory for the 3D volume + projection batch is
  expected. Reduce batch_size, chunk_size, n_samples, or view_step if you run
  out of memory.

Example install:
    pip install torch numpy matplotlib monai
    # tinycudann: follow https://github.com/NVlabs/tiny-cuda-nn


Directory layout
--------------------------------------------------------------------------------
The code expects the model files to live inside a "models" package
(imports use "from models.MotionNetHash import ..." and "from models.hash_encoder
import ..."). A working layout is:

    AI_Geocal.py                        Training entry point
    Sample.py                           Inference / export entry point
    differentiable_forward_projector.py Differentiable ray-marching projector
    DoF_transform.py                    DoF parameter -> projection-matrix transforms
    helpers.py                          I/O, geometry recompute, gantry file utils
    print_numpy.py                      Plots exported motion curves
    models/
        __init__.py
        MotionNetHash.py                Motion networks (6/9/10-DoF, Fourier MLP)
        hash_encoder.py                 Hash-grid encoders (tcnn + PyTorch), SIREN
        UNet_openai.py                  U-Net (OpenAI guided-diffusion style)*
        nn.py                           NN utilities used by UNet_openai*

    * UNet_openai.py and nn.py are auxiliary building blocks and are not on the
      core calibration path (AI_Geocal / Sample). Keep them only if you use them.

Create models/__init__.py (can be empty) so the package imports resolve.


Input data
--------------------------------------------------------------------------------
1) Reference volume
   Raw float32 binary, shape (imsz, imsy, imsx), voxel size (dz, dy, dx) mm.
   This is the known object used to render synthetic projections.

2) Measured projections
   Raw float32 binary, shape (NLAM, nv, nu), pixel size (dv, du) mm,
   where NLAM is the number of views.

Both are loaded as memory-mapped arrays via load_raw_f32_memmap(). All geometry
parameters (detector size, pixel/voxel pitch, scan angle, reverse flags, ROI,
recon_type, etc.) are set in the ReconConfig dataclass and the call site in
__main__.

No initial projection matrix or gantry file is required: a nominal circular
orbit is built analytically from a few scalar parameters
(k_nominal, un_nominal, vn_nominal, SOD, SDD, orbit axis, scan/start angle).


Usage
--------------------------------------------------------------------------------
1) Training (recover the motion / calibration)

   Edit the ReconConfig and the train_motion_hash_model(...) call at the bottom
   of AI_Geocal.py to point at your volume and projections and to set the
   nominal geometry, then run:

       python AI_Geocal.py

   Key arguments (see __main__ for a full example):
       cfg               ReconConfig with detector/volume geometry
       volume_path       path to reference volume .raw (float32)
       proj_meas_path    path to measured projections .raw (float32)
       roi               RT_PARAM region of interest
       out_dir           output directory
       epochs, batch_size, lr, view_step
       ts_max_mm, tp_max_mm, rot_max_deg, skew_max   bounds for each DoF
       k_nominal, un_nominal, vn_nominal, SOD, SDD   nominal geometry
       nominal_orbit_axis, nominal_clockwise_sign, nominal_include_endpoint

   Training outputs written to out_dir:
       motion_model_epXXXX.pth     checkpoints (model + optimizer + config)
       loss_history.csv / .npy     per-epoch loss and timing
       training_time.txt           total wall-clock timing summary
       P_nominal_analytic.npy      nominal projection matrices (V,12)
       geo_nominal_analytic.npy    nominal geometry parameters (V,7)
       motion_p10_raw.npy          raw network outputs (V,10)
       motion_ts_mm.npy            recovered intrinsic translation (V,3)
       motion_tp_mm.npy            recovered extrinsic translation (V,3)
       motion_rot_deg.npy          recovered rotation (V,3)
       motion_skew.npy             recovered skew (V,1)

2) Export aligned projections and gantry file (inference)

   Sample.py loads a trained checkpoint and exports the corrected projection
   matrices, before/after projections, and gantry (.dat) files. Configure the
   call to export_aligned_projections_and_gantry(...) in Sample.py __main__,
   then run:

       python Sample.py

   Typical outputs:
       Projections_before_10DoF.raw    rendered from nominal baseline
       Projections_aligned_10DoF.raw   rendered after learned correction
       Gantry_nominal_10DoF.dat        nominal gantry
       Gantry_updated_10DoF.dat        corrected gantry

3) Plot the recovered motion curves

   print_numpy.py reads the exported motion .npy files and saves
   intrinsic/extrinsic parameter plots. Update base_dir at the top of the file
   to your export folder, then run:

       python print_numpy.py


Coordinate / convention notes
--------------------------------------------------------------------------------
- World vs internal axes: the pipeline uses INTERNAL(x,y,z) = WORLD(x,z,y).
  The y/z swap is handled explicitly (see _swap_yz in the projector and the
  permutation S in DoF_transform.py). Keep this in mind when comparing against
  external conventions.
- Volume origin is set as X0 = -0.5*imsx*dx (+offset), similarly for Y0; Z0 = 0
  by the existing gantry convention.
- ureverse / vreverse flags handle detector axis flips.
- recon_type selects the ROI-to-detector mapping (0 or 1).
- grid_sample uses align_corners=False by default; the normalized-coordinate
  math in _world_to_grid_norm must match the projector's setting.


Tips / troubleshooting
--------------------------------------------------------------------------------
- Out of GPU memory: lower batch_size, chunk_size, or n_samples, or increase
  view_step to train on fewer views.
- Loss not decreasing: check that the nominal geometry (SOD/SDD, k, un, vn,
  orbit axis, clockwise sign, start angle) and reverse flags roughly match the
  real system; the optimizer only corrects within +/- the *_max bounds.
- Make sure the reference volume and measured projections share a consistent
  intensity scale; LNCC is fairly robust to scale but gross mismatches hurt.
- The example geometry values in __main__ are placeholders for specific scans;
  replace them with your own system's parameters.
- For a 9-DoF run (no skew), set skew_max = 0; the 10-DoF model still runs but
  its skew output is scaled to zero, effectively reducing it to 9-DoF.


License / citation
--------------------------------------------------------------------------------
Copyright (c) KAIST, MIR Lab (Medical Imaging and Radiotherapy Lab).
All rights reserved.

This software is developed and owned by the MIR Lab at KAIST. Please contact the
lab regarding usage, redistribution, and licensing terms before any external use.

Portions of models/hash_encoder.py adapt the Instant-NGP hash encoding
(NVlabs/tiny-cuda-nn) and HashNeRF-pytorch (yashbhalgat/HashNeRF-pytorch).
models/UNet_openai.py and models/nn.py follow the OpenAI guided-diffusion
codebase. Please retain the relevant upstream licenses and credits.
================================================================================
