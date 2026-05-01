# Patch by Max shinn 2025: https://github.com/mwshinn/cellpose3dplus
# Available under the MIT license
import numpy as np
from cellpose.core import core_logger, run_net
import cellpose.models
from packaging.version import Version
import torch

v = Version(cellpose.version)
assert v.major == 3 and v.minor == 1, "This package requires cellpose version 3.1.x (ideally 3.1.1.2)"
if v.micro != 1:
    print("Warning, this patch was only tested on Cellpose 3.1.1.2, use at your own risk!")

def run_3D_xy_zsplit(net, imgs, batch_size=8, augment=False,
                     tile_overlap=0.1, bsize=224, net_ortho=None,
                     progress=None):
    """
    Monkey patch for cellpose.core.run_3D

    Get X and Y flows and cellprob from XY planes, and Z flows from YZ and XZ planes.
    """
    core_logger.info("Running Cellpose3DPlus")
    print("Running Cellpose3DPlus")
    sstr = ["YX", "ZY", "ZX"]
    pm   = [(0, 1, 2, 3), # XY
            (1, 0, 2, 3), # YZ
            (2, 0, 1, 3)] # XZ
    ipm  = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]
    
    shape = imgs.shape[:-1] # Z,Y,X
    yf    = np.zeros((*shape, 4), np.float32) # Z, Y, X, cell-prob
    yf[..., 0] = 0.0
    
    # First run in the xy dimension
    core_logger.info("running %s: %d planes of size (%d, %d)",
                     sstr[0], shape[0], shape[1], shape[2])
    y_xy, style = run_net(net, imgs,
                          batch_size=batch_size, augment=augment,
                          bsize=bsize, tile_overlap=tile_overlap, rsz=None)
    # channel-map: 0 = Y flow, 1 = X flow, 2 = cellprob
    yf[..., 1] = y_xy[..., 0]
    yf[..., 2] = y_xy[..., 1]
    yf[..., 3] = y_xy[..., 2]
    del y_xy
    if progress is not None:
        progress.setValue(25)
        
    # Now run in xy and yz dimensions
    for p in (1, 2):
        xsl = imgs.transpose(pm[p])
        core_logger.info("running %s: %d planes of size (%d, %d)",
                         sstr[p], shape[pm[p][0]],
                         shape[pm[p][1]], shape[pm[p][2]])
        y_ortho, _ = run_net(net if net_ortho is None else net_ortho,
                             xsl, batch_size=batch_size, augment=augment,
                             bsize=bsize, tile_overlap=tile_overlap, rsz=None)
        # channel 0 of orthogonal output is the Z flow for that view
        yf[...,0] += y_ortho[..., 0].transpose(ipm[p])
        del y_ortho
        if progress is not None:
            progress.setValue(25 + 35 * p)
    # average the two orthogonal estimates
    yf[..., 0] *= 0.5
    if progress is not None:
        progress.setValue(95)
    return yf, style


cellpose.core.run_3D = run_3D_xy_zsplit
cellpose.run_3D = run_3D_xy_zsplit
cellpose.models.run_3D = run_3D_xy_zsplit
core_logger.info("Patched cellpose.core.run_3D with Cellpose3DPlus")

# Also fix the bug where Cellpose crashes due to out of memory errors.  This
# always comes not from running the network, but from computing the masks, so we
# just use the CPU for that portion.

old_compute_masks = cellpose.models.CellposeModel._compute_masks
TorchOutOfMemoryError = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError


def new_compute_masks(self, *args, **kwargs):
    print("Overriding compute masks")
    device = self.device
    try:
        val = old_compute_masks(self, *args, **kwargs)
    except TorchOutOfMemoryError:
        print("Falling back to CPU mask computations to work around a cellpose bug")
        self.device = torch.device("cpu")
        val = old_compute_masks(self, *args, **kwargs)
    finally:
        self.device = device
    return val

cellpose.models.CellposeModel._compute_masks = new_compute_masks

# Allow running from the command line
if __name__ == "__main__":
    import cellpose.__main__
    cellpose.__main__.main()
