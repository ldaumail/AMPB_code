import os
import os.path as op
import itertools
import nibabel as nib
import numpy as np
from collections import Counter
from scipy.ndimage import binary_erosion
from nilearn import plotting, image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from dipy.io.streamline import load_tractogram, load_trk
from dipy.tracking.streamline import transform_streamlines

from nibabel.freesurfer import io as fsio

participant = 'sub-EBxGxEYx1965'
base_path = op.join('/Volumes','cos-lab-wpark78','LoicDaumail','ampb','derivatives','pyafq', 'gpu-afq_MT-STS1_nseeds20_0mm_nowm_dist3',participant)
bundle_path = op.join(base_path,'bundles')
qsiprep_path = op.join('/Volumes','cos-lab-wpark78','LoicDaumail','ampb','derivatives', 'qsiprep', participant)
surf_roi_path = op.join('/Users','ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb', 'analysis', 'functional_surf_roi', participant)


#Upload tracts in

t1w_img = nib.load(op.join(qsiprep_path, 'anat',
                           participant+'_space-ACPC_desc-preproc_T1w.nii.gz'))
t1w = t1w_img.get_fdata()

sts1_mtL = load_tractogram(op.join(bundle_path,
                          participant+'_ses-04_acq-HCPdir99_desc-STS1xMTL_tractography.trx'), t1w_img)
sts1_mtR = load_tractogram(op.join(bundle_path,
                           participant+'_ses-04_acq-HCPdir99_desc-STS1xMTR_tractography.trx'), t1w_img)

# read in the bundles, transform into anatomical coordinates (might be useful if in anatomical space)
# sts1_mtL_t1w = transform_streamlines(sts1_mtL.streamlines,
#                                 np.linalg.inv(t1w_img.affine))
# sts1_mtR_t1w = transform_streamlines(sts1_mtR.streamlines,
#                                 np.linalg.inv(t1w_img.affine))
# streamlinesL = sts1_mtL_t1w
# streamlinesR = sts1_mtR_t1w
streamlinesL = sts1_mtL.streamlines
streamlinesR = sts1_mtR.streamlines

# print(t1w_img.affine)
# print(t1w_img.header)
#### Calculate endpoint densities at MT ROI boundary

### --- Step 1: get MRI slice coordinates:
# All 8 corners of the 3D volume
corners = list(itertools.product(
    [0, t1w_img.shape[0]-1],
    [0, t1w_img.shape[1]-1],
    [0, t1w_img.shape[2]-1]
))

# Transform all corners to world space
world_corners = nib.affines.apply_affine(t1w_img.affine, corners)

# Get actual min/max along each physical axis
world_min = world_corners.min(axis=0)
world_max = world_corners.max(axis=0)

# --- Create new grid with 0.55 mm resolution ---
# res = 0.55  # mm
# x = np.arange(world_min[0], world_max[0] + res, res)
# y = np.arange(world_min[1], world_max[1] + res, res)
# z = np.arange(world_min[2], world_max[2] + res, res)

# # Generate meshgrid in world coordinates
# X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

### --- Step 2: calculate streamlines density across the brain -- ###
# Assume your streamline coordinates are in world (RASMM) space
def streamline_density_grid(streamlines, world_min, world_max, res=1):
    """
    Compute streamline density across a 3D grid in world (RASMM) coordinates.
    
    Parameters
    ----------
    streamlines : list of ndarray
        Each element is an (N_i, 3) array of streamline coordinates in RASMM space.
    world_min : array-like, shape (3,)
        Minimum XYZ coordinates (mm) of the bounding box.
    world_max : array-like, shape (3,)
        Maximum XYZ coordinates (mm) of the bounding box.
    res : float
        Grid resolution in mm.
    
    Returns
    -------
    density : ndarray, shape (nx, ny, nz)
        Density values (number of streamline points in each voxel).
    X, Y, Z : ndarray
        Meshgrid arrays of world coordinates.
    """
    # streamlines = streamlinesL
    # Generate grid edges
    x = np.arange(world_min[0], world_max[0] + res, res)
    y = np.arange(world_min[1], world_max[1] + res, res)
    z = np.arange(world_min[2], world_max[2] + res, res)
    
    # Meshgrid for reference
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    
    # Initialize density grid
    density = np.zeros((len(x), len(y), len(z)), dtype=int)
    # density.shape
    # Count streamline points
    for sl in streamlines:
        # Convert continuous coords to grid indices
        idx_x = ((sl[:, 0] - world_min[0]) / res).astype(int)
        idx_y = ((sl[:, 1] - world_min[1]) / res).astype(int)
        idx_z = ((sl[:, 2] - world_min[2]) / res).astype(int)
        # Increment density
        for ix, iy, iz in zip(idx_x, idx_y, idx_z):
            if (0 <= ix < density.shape[0] and
                0 <= iy < density.shape[1] and
                0 <= iz < density.shape[2]):
                density[ix, iy, iz] += 1
    
    return density, X, Y, Z


#Left STS1-MT
res = 1
voxel_hit_countsL, XL, YL, ZL = streamline_density_grid(streamlinesL, world_min, world_max, res)

#  
#Right STS1-MT
voxel_hit_countsR, XR, YR, ZR = streamline_density_grid(streamlinesR, world_min, world_max, res)

maskL = voxel_hit_countsL > 0
maskR = voxel_hit_countsR > 0

# Get coordinates of nonzero voxels for left and right separately
coordsL = np.column_stack(np.nonzero(maskL))
coordsR = np.column_stack(np.nonzero(maskR))

# Convert voxel indices to world coordinates (voxel center positions)
world_coordsL = np.column_stack([
    world_min[0] + coordsL[:, 0] * res,
    world_min[1] + coordsL[:, 1] * res,
    world_min[2] + coordsL[:, 2] * res
])
world_coordsR = np.column_stack([
    world_min[0] + coordsR[:, 0] * res,
    world_min[1] + coordsR[:, 1] * res,
    world_min[2] + coordsR[:, 2] * res
])

# Get hit counts for color mapping
countsL = voxel_hit_countsL[maskL]
countsR = voxel_hit_countsR[maskR]

# Normalize for color mapping
normL = Normalize(vmin=countsL.min(), vmax=countsL.max())
normR = Normalize(vmin=countsR.min(), vmax=countsR.max())


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Left MT points in red colormap
scL = ax.scatter(world_coordsL[:, 0], world_coordsL[:, 1], world_coordsL[:, 2],
                 c=countsL, cmap='Reds', norm=normL, s=5, alpha=0.8, label='Left STS1-MT')

# Right MT points in blue colormap
scR = ax.scatter(world_coordsR[:, 0], world_coordsR[:, 1], world_coordsR[:, 2],
                 c=countsR, cmap='Blues', norm=normR, s=5, alpha=0.8, label='Right STS1-MT')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Streamline Density (Scatter with Gradient)')

ax.legend()

# Add colorbars
fig.colorbar(scL, ax=ax, shrink=0.5, pad=0.1, label='Left STS1-MT Hit Count')
fig.colorbar(scR, ax=ax, shrink=0.5, pad=0.05, label='Right STS1-MT Hit Count')

plt.show()

## -- Project Streamline densities to surface MT ROI vertices -----
## Load Surface ROI data 
vertex_indices_mt_lh = fsio.read_label(op.join(surf_roi_path, participant+'_hemi-L_space-fsnative_label-MT_mask.label'), read_scalars=False)
vertex_indices_mt_rh = fsio.read_label(op.join(surf_roi_path, participant+'_hemi-R_space-fsnative_label-MT_mask.label'), read_scalars=False)

print(vertex_indices_mt_rh)



#-----------------------------------------------------#
#Check coordinates and space for each file
def center_world(img):
    center_voxel = np.array(img.shape) // 2
    center_world = img.affine @ np.append(center_voxel, 1)
    return center_world[:3]
mtR_roi = nib.load(op.join(roi_path, participant+'_hemi-R_space-ACPC_label-MT_mask_dilated.nii.gz'))
mtR_roi.affine
print("Center world coords image 1:", center_world(mtR_roi))
mtR_roi =  nib.load(op.join(base_path, 'ROIs', participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-STS1xMTREnd_mask.nii.gz'))
mtR_roi.affine
print("Center world coords image 2:", center_world(mtR_roi))
julich_path = op.join('/Users','ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb', 'analysis', 'julich_space-ACPC_rois')
sts1_roi = nib.load(op.join(julich_path, participant, 'ses-04', 'anat', participant+'_ses-04_desc-lhSTS103SyN_mask.nii.gz'))
sts1_roi.affine
fa_path = op.join(base_path,'models')
fa_img = nib.load(op.join(fa_path, participant+'_ses-04_acq-HCPdir99_model-dki_param-fa_dwimap.nii.gz'))
fa = fa_img.get_fdata()
print("Center world coords image 3:", center_world(fa_img))
print(fa_img.header)
t1w_img.affine
print(t1w_img.header)
print(sts1_mtR.space)


########### Visualize Bundles #############
from fury import actor, window
from fury.colormap import create_colormap

import AFQ.data.fetch as afd
from AFQ.viz.utils import PanelFigure


qsiprep_path = op.join('/Volumes','cos-lab-wpark78','LoicDaumail','ampb','derivatives', 'qsiprep', participant, 'anat')
# read in the bundles, trasnform into RASMM coordinates and then into anatomical coordinates
t1w_img = nib.load(op.join(qsiprep_path,
                           participant+'_space-ACPC_desc-preproc_T1w.nii.gz'))
t1w = t1w_img.get_fdata()
sts1_mtL.to_rasmm()
sts1_mtR.to_rasmm()
sts1_mtL_t1w = transform_streamlines(sts1_mtL.streamlines,
                                np.linalg.inv(t1w_img.affine))
sts1_mtR_t1w = transform_streamlines(sts1_mtR.streamlines,
                                np.linalg.inv(t1w_img.affine))

#Visualizing bundles with principal direction coloring

def lines_as_tubes(sl, line_width, **kwargs):
    line_actor = actor.line(sl, **kwargs)
    line_actor.GetProperty().SetRenderLinesAsTubes(1)
    line_actor.GetProperty().SetLineWidth(line_width)
    return line_actor


sts1_mtL_actor = lines_as_tubes(sts1_mtL_t1w, 8)
sts1_mtR_actor = lines_as_tubes(sts1_mtR_t1w, 8)

#Slice anatomical image using slicer actors

def slice_volume(data, x=None, y=None, z=None):
    slicer_actors = []
    slicer_actor_z = actor.slicer(data)
    if z is not None:
        slicer_actor_z.display_extent(
            0, data.shape[0] - 1,
            0, data.shape[1] - 1,
            z, z)
        slicer_actors.append(slicer_actor_z)
    if y is not None:
        slicer_actor_y = slicer_actor_z.copy()
        slicer_actor_y.display_extent(
            0, data.shape[0] - 1,
            y, y,
            0, data.shape[2] - 1)
        slicer_actors.append(slicer_actor_y)
    if x is not None:
        slicer_actor_x = slicer_actor_z.copy()
        slicer_actor_x.display_extent(
            x, x,
            0, data.shape[1] - 1,
            0, data.shape[2] - 1)
        slicer_actors.append(slicer_actor_x)

    return slicer_actors


slicers = slice_volume(t1w, x=t1w.shape[0] // 2, z=t1w.shape[-1] // 3)

#Add actors to 3D window scen object
scene = window.Scene()

#scene.add(sts1_mtL_actor)
scene.add(sts1_mtR_actor)

#mtL_actor = actor.contour_from_roi(mtL_dat, color=(1, 0, 0), opacity=0.5)
mtR_actor = actor.contour_from_roi(mtR_dat, color=(1, 0, 0), opacity=0.5)

# Show in interactive window
#scene.add(mtL_actor)
scene.add(mtR_actor)


for slicer in slicers:
    scene.add(slicer)

#Now visualize:
window.show(scene, size=(1200, 1200), reset_camera=False)
