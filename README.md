# Improved 3D segmentation for Cellpose

## Summary

When performing 3D segmentation on images without a very high signal-to-noise
ratio, Cellpose can sometimes create striped or hash patterns, oversegmenting
cells and making the segmentations unusable.  The following patch fixes this
problem.

This also includes a workaround for the ["torch.OutOfMemoryError: CUDA out of
memory" bug](https://github.com/MouseLand/cellpose/issues/1182) in Cellpose
([also here](https://github.com/MouseLand/cellpose/issues/918)) during the mask
finding step.


## Usage

Simply download this Python file and import it after after importing Cellpose:

``` python
import cellpose
import cellpose3dplus
```

To confirm it is running, you will see "Running Cellpose3DPlus" printed to the
command line when running 3D segmentation.

If you run cellpose from the command line or use the GUI, you can alternatively
use this in place of Cellpose.  Instead of:

> python -m cellpose --Zstack

Or:

> cellpose --Zstack

Use:

> python -m cellpose3dplus --Zstack

## Requirements

This ONLY supports Cellpose version 3.1.1.x.  (Recommended: version 3.1.1.3)

## Technical details

In brief, Cellpose's 3D algorithm makes three passes through the image: once on
the z axis (running the segmentation network on each xy plane), once on the y
axis (running it on each xz plane), and once on the x axis (running it on each
yz plane).  The output of each run of the segmentation network is a map of the
probability that a given location is a cell, and a gradient pointing towards the
centre of the cell.  By default, Cellpose averages the cell probabilities for
each run of the network with equal weighting.  However, usually, the z dimension
(with xy planes) has a much stronger signal than the others.  This patch
modifies cellpose's algorithm to use the cell probabilities and gradients
derived from the most reliable run for all gradients except the z gradient, for
which it averages this components of the y and x axis runs (discarding the y and
x components and cell probabilities from these runs).

<img align="right" src="https://raw.githubusercontent.com/mwshinn/cellpose3dplus/master/img/cellpose3dplus.jpg" width="100%">


