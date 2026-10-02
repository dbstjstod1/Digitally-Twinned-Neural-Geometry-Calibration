# Measured Denseball reproduction example

This guide records the acquisition and training settings used for the paper's
measured-data experiment. Its array sizes, panel crop, distances, and training
hyperparameters are specific to that dataset. The CLI loads the
[Denseball preset](../geocal/presets/denseball.py); changing only input filenames
does not adapt the workflow to another acquisition.

## Input data

Supply little-endian float32, C-order raw data. The reference shape is
`(801, 929, 929)` in `(z, y, x)`; projections are `(480, 1264, 776)` in
`(view, row, column)`. Voxel pitch is 0.2 mm, detector pitch 0.228 mm,
SOD 443 mm, and SDD 650 mm. The preset records the remaining conventions.
The original files are required and are not distributed in this repository.

## Train and export

```bash
CUDA_VISIBLE_DEVICES=0 python -m geocal.train_real 1 \
  --volume /data/reference.raw --projections /data/projections.raw \
  --out-dir result/measured
CUDA_VISIBLE_DEVICES=0 python -m geocal.export_real 1 \
  --volume /data/reference.raw --train-dir result/measured \
  --out-dir result/measured/export
```

The positional argument is the training view stride; export evaluates every
acquired view. The recorded recipe uses MONAI LNCC31, 100 epochs, Adam at 0.001,
batch 4, seed 0, and a 16-level hash grid. The panel-specific loss crop is
`u=26:776, v=0:1182` and is currently implemented in
[real_train.py](../geocal/real_train.py). This crop must be adapted for other data;
it is not a general detector requirement.

Export restores geometry and correction bounds from its checkpoint. Outputs
include `P_updated_9DoF.npy`, the recovered corrections, gantry `.dat` files,
and simulated projections.

## Adapting the acquisition

Update the volume dimensions and spacing, detector dimensions and pitch,
nominal geometry, view ordering, coordinate origin, and correction bounds in the
configuration used by your caller. Adjust the loss crop in the measured training
path to the valid shared image region. Preserve the coordinate conventions
expected by the projector and verify nominal projections before fitting.

The example-specific settings above are retained to make the recorded experiment
reproducible. They do not prescribe the projection size or ROI of the method.
