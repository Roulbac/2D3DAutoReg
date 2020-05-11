# GPU Digitally Reconstructed Radiograph (DRR) Registration GUI in PySide2

This repository contains code for CPU-based and GPU-based DRR to X-Ray registration software.

The registration is set up with 4 views, needing 4 camera intrinsic and extrinsic matrices.

## Camera file format 

> K = [3510.918213, 0.000000, 368.718994; 0.000000, 3511.775635, 398.527802; 0.000000, 0.000000, 1.000000]

> M = [-0.785341, -0.068020, -0.615313, -5.901115; 0.559239, 0.348323, -0.752279, -4.000824; 0.265498, -0.934903, -0.235514, -663.099792]

> H = 768

> W = 768

# Usage

The registration GUI is started with the following command
> python main_window.py

Initialization steps are as following

1. Open camera files
2. Open CT volume
3. Open target X-Ray images

To run DRR generation through the GPU, tick the 'Enable GPU Mode' option.

To auto-refresh DRR generation at parameter change, which can be useful for quick initial positioning, tick the 'Enable auto-refresh' option.

# Coordinate system

The coordinate system is defined in physical space (milimeters).
Its basis is along the CT's ijk axes, scaled with thir respective voxel dimensions.
This means that the CT is kept fixed, and that it is the cameras that are moving.


# Requirements

PySide2
Numba
Numpy
PyCuda
SimpleITK
Scikit-Image

