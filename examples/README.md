# Fixed circular and sineSpin reproduction

The reference is the original `phantom_density_v1_643x643x651.float32.raw`:
35 balls, little-endian float32 in C-order `(651, 643, 643)`, attenuation in
1/mm, with isotropic 0.4 mm voxels. Supply the original file separately.
The data are not included in this repository.

## Independent projection generator

Use LEAP 1.26 at the revision below and the supplied Joseph-selection patch.
The patch changes kernel selection, not the calibration model. Build with
an NVIDIA CUDA toolkit supported by the installed PyTorch environment.

```bash
export AIGEOCAL_LEAP_DIR=/path/to/LEAP
git clone https://github.com/LLNL/LEAP.git "$AIGEOCAL_LEAP_DIR"
git -C "$AIGEOCAL_LEAP_DIR" checkout 0c8846f42b2e59340d5559fc1271d590a292f9a0
git -C "$AIGEOCAL_LEAP_DIR" apply "$PWD/patches/leap_force_joseph.patch"
cmake -S "$AIGEOCAL_LEAP_DIR" -B "$AIGEOCAL_LEAP_DIR/build"
cmake --build "$AIGEOCAL_LEAP_DIR/build" --parallel 8
export PYTHONPATH="$AIGEOCAL_LEAP_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Generate, calibrate, evaluate

Choose unused output directories; the commands refuse to overwrite prior runs.

```bash
for family in circular sinespin; do
  python -m geocal.prepare --family "$family" \
    --volume /data/phantom_density_v1_643x643x651.float32.raw \
    --out-dir "result/$family/input" --gpu 0
  python -m geocal.train --input-dir "result/$family/input" \
    --volume /data/phantom_density_v1_643x643x651.float32.raw \
    --out-dir "result/$family/proposed" --gpu 0
  python -m geocal.evaluate --input-dir "result/$family/input" \
    --run-dir "result/$family/proposed"
done
```

## Recorded settings

- Both scans: 546 views over 220°, SOD 750 mm, SDD 1200 mm.
- sineSpin: one nominal tilt cycle with ±10° amplitude. This is an angle,
  not a ±10 mm source displacement.
- The local perturbation rotates the rigid source/detector assembly about a
  tangential axis halfway from isocenter to source. It gives a one-cycle
  source-z wobble of 5 mm peak-to-peak over 10–45% of the scan, followed by
  a triangular 2.5 mm rise/fall over 55–90%. This is a declared simulated
  mechanical stress case, not a measurement of scanner motion.
- LEAP Joseph projections, Beer–Lambert Poisson noise with I0=44,000 and seed 0.
  Both implementations use the same sampled reference and Joseph discretization;
  independent implementations do not remove that modeling limitation.
- ROI: reference bead bounding boxes plus a two-voxel halo, projected with
  nominal geometry, 20-pixel margin, and common crop size; frozen before fitting.
- All nine outputs remain per-view. Bounds: intrinsic ±1 mm, translation ±3 mm,
  rotation ±0.6°. Eight hash levels, base resolution 16, finest resolution 128.
- Signed LNCC31 plus `0.01 * mean(abs(intrinsic_correction / 1 mm))`.
  No translation or rotation penalty; no shared intrinsics or trajectory template.
- Seed 1; batch 4; training uses every second view. First stage: 120 epochs at
  initial LR 0.003. Second stage: model-only warm start, fresh Adam, 60 epochs
  at initial LR 0.0003. Each stage uses cosine decay to 5% of its initial LR.
- Use the final raw weights, without EMA or checkpoint search. The fixed recipe
  was selected during development on these simulated cases. Replaying it is
  not a new independent validation study; GT is excluded from the fitting loss.

`geocal.train` reads only the reference, nominal matrices, observed projections,
and fixed ROI. `geocal.evaluate` separately reads GT and fixed bead centers.
The evaluation outputs RPE mean, population standard deviation, RMS and maximum,
source-position error, nominal-relative parameter errors, and per-view CSV/NumPy
arrays. RPE is the Euclidean detector error over view/ball pairs; source error
is in physical millimeters. No per-view alignment is performed.

The original measured-data recipe is retained separately in
`geocal.presets.denseball`; it is not replaced by the simulation settings.
