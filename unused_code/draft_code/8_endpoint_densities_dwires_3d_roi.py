import os
import os.path as op
import nibabel as nib
import numpy as np
from collections import Counter
from scipy.ndimage import binary_erosion
from nilearn import plotting, image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dipy.io.streamline import load_tractogram, load_trk
from dipy.tracking.streamline import transform_streamlines

participant = 'sub-EBxGxEYx1965'
base_path = op.join('/Volumes','cos-lab-wpark78','LoicDaumail','ampb','derivatives','pyafq', 'gpu-afq_MT-STS1_nseeds20_0mm_nowm_dist3',participant)
bundle_path = op.join(base_path,'bundles')
qsiprep_path = op.join('/Volumes','cos-lab-wpark78','LoicDaumail','ampb','derivatives', 'qsiprep', participant, 'anat')
roi_path = op.join('/Users','ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb', 'analysis', 'functional_vol_roi', participant)

#Upload tracts in

t1w_img = nib.load(op.join(qsiprep_path,
                           participant+'_space-ACPC_desc-preproc_T1w.nii.gz'))
t1w = t1w_img.get_fdata()

sts1_mtL = load_tractogram(op.join(bundle_path,
                          participant+'_ses-04_acq-HCPdir99_desc-STS1xMTL_tractography.trx'), t1w_img)
sts1_mtR = load_tractogram(op.join(bundle_path,
                           participant+'_ses-04_acq-HCPdir99_desc-STS1xMTR_tractography.trx'), t1w_img)

# read in the bundles, transform into RASMM coordinates and then into anatomical coordinates
# sts1_mtL.to_rasmm()
# sts1_mtR.to_rasmm()
sts1_mtL_t1w = transform_streamlines(sts1_mtL.streamlines,
                                np.linalg.inv(t1w_img.affine))
sts1_mtR_t1w = transform_streamlines(sts1_mtR.streamlines,
                                np.linalg.inv(t1w_img.affine))
streamlinesL = sts1_mtL_t1w
streamlinesR = sts1_mtR_t1w

#### Load MT ROI ###
# mtL_roi =  nib.load(op.join(base_path, 'ROIs', participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-STS1xMTLEnd_mask.nii.gz'))
# mtR_roi =  nib.load(op.join(base_path, 'ROIs', participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-STS1xMTREnd_mask.nii.gz'))

# mtL_dat = mtL_roi.get_fdata()
# mtR_dat = mtR_roi.get_fdata()
# print("Affine 1:\n", mtR_roi.affine)
# print(mtR_roi.header)
# #binarize
# mtL_dat[mtL_dat > 0] = 1
# mtR_dat[mtR_dat > 0] = 1


mtL_roi = nib.load(op.join(roi_path, participant+'_hemi-L_space-ACPC_label-MT_mask_dilated.nii.gz'))
mtR_roi = nib.load(op.join(roi_path, participant+'_hemi-R_space-ACPC_label-MT_mask_dilated.nii.gz'))
mtL_dat = mtL_roi.get_fdata()
mtR_dat = mtR_roi.get_fdata()


#### Calculate endpoint densities at MT ROI boundary
### --- Step 1: obtain surface ROI coordinates -- ###

# --- Compute surface voxels ---
mtL_eroded = binary_erosion(mtL_dat) #If no structuring element is provided, an element is generated with a square connectivity equal to one. Connectivity = 1 keeps the erosion conservative, shrinking only along face-adjacent voxels
surface_mtL = mtL_dat - mtL_eroded  # Surface voxels are 1; others are 0

mtR_eroded = binary_erosion(mtR_dat)
surface_mtR = mtR_dat - mtR_eroded 

# --- Check the ROIs created ---
# plot 
# plotting.plot_roi(mtL_roi, bg_img=fa_img, alpha=0.5, cmap="autumn", title="MT")
# plt.show()

#mtL_roi_eroded = nib.Nifti1Image(mtL_eroded.astype(np.float32), affine=mtL_roi.affine)

# plotting.plot_roi(mtL_roi_eroded, bg_img=fa_img, alpha=0.5, cmap="autumn", title="MT")
# plt.show()

#mtL_roi_surface = nib.Nifti1Image(surface_mtL.astype(np.float32), affine=mtL_roi.affine)
# plotting.plot_roi(mtL_roi_surface, bg_img=fa_img, alpha=0.5, cmap="autumn", title="MT")
# plt.show()

# --- Get surface voxel coordinates ---
surface_voxel_mtLcoords = np.argwhere(surface_mtL)
surface_voxel_mtLset = set(map(tuple, surface_voxel_mtLcoords))  # Faster lookup

surface_voxel_mtRcoords = np.argwhere(surface_mtR)
surface_voxel_mtRset = set(map(tuple, surface_voxel_mtRcoords))  # Faster lookup

### -- Step 2: Count streamlines that cross ROI surface voxels
# Assume your streamline coordinates are in world (RASMM) space
def streamlinePerVoxel(streamlines, surfaceRoi):
    #Inputs: streamlines coordinates, surface ROI coordinates
    valid_surface_hits = [[] for _ in range(len(streamlines))]
    for s in range(len(streamlines)):
        streamline = np.round(streamlines[s]).astype(int)
        for coord in streamline:
            voxel = tuple(coord)
            if voxel in surfaceRoi:
                valid_surface_hits[s].append(voxel)

    # Flatten the list of lists
    all_voxel_hits = [voxel for streamline_hits in valid_surface_hits for voxel in streamline_hits]
    # Count occurrences of each voxel triplet
    voxel_hit_counts = Counter(all_voxel_hits)
    return voxel_hit_counts

#Left MT
voxel_hit_countsL = streamlinePerVoxel(streamlinesL, surface_voxel_mtLset)

#Right MT
voxel_hit_countsR = streamlinePerVoxel(streamlinesR, surface_voxel_mtRset)

# Plot the data

# Merge surface voxel coordinates into a set with intensity = 0 if not hit, else the hit count
surface_voxel_mtLset = set(map(tuple, surface_voxel_mtLcoords))
all_voxels = list(surface_voxel_mtLset.union(voxel_hit_countsL.keys()))
intensities = [voxel_hit_countsL.get(v, 0) for v in all_voxels]

# Convert to arrays for plotting
coords = np.array(all_voxels)
x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
c = np.array(intensities)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(x, y, z, c=c, cmap='Spectral', s=60)
ax.view_init(elev=20, azim=180)  # Try azim=180 for a "back" view
ax.set_title("STS1 Endpoint Density on Left MT ROI Surface")
fig.colorbar(img, ax=ax, label='Endpoint Count')

savedir = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis','ampb','analysis','plots')
os.makedirs(savedir, exist_ok=True)
plt.savefig(op.join(savedir, participant+'STS1_MT_L_endpoint_density_plot.png'), dpi=300, bbox_inches='tight')
plt.show()

#Right MT
surface_voxel_mtRset = set(map(tuple, surface_voxel_mtRcoords))
all_voxels = list(surface_voxel_mtRset.union(voxel_hit_countsR.keys()))
intensities = [voxel_hit_countsR.get(v, 0) for v in all_voxels]

# Convert to arrays for plotting
coords = np.array(all_voxels)
x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
c = np.array(intensities)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(x, y, z, c=c, cmap='Spectral', s=60)
# ax.view_init(elev=20, azim=180)  # Try azim=180 for a "back" view
ax.set_title("STS1 Endpoint Density on Right MT ROI Surface")
fig.colorbar(img, ax=ax, label='Endpoint Count')

savedir = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis','ampb','analysis','plots')
os.makedirs(savedir, exist_ok=True)
plt.savefig(op.join(savedir, participant+'STS1_MT_R_endpoint_density_plot.png'), dpi=300, bbox_inches='tight')
plt.show()

# print("Affine 2:\n", mtR_roi.affine)
# print(mtR_roi.header)


# ------- Check coordinates and space for each file ----
def center_world(img):
    center_voxel = np.array(img.shape) // 2
    center_world = img.affine @ np.append(center_voxel, 1)
    return center_world[:3]
mtR_roi = nib.load(op.join(roi_path, participant+'_hemi-R_space-ACPC_label-MT_mask_dilated.nii.gz'))
mtR_roi.affine
print("Center world coords image 2:", center_world(mtR_roi))
mtR_roi =  nib.load(op.join(base_path, 'ROIs', participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-STS1xMTREnd_mask.nii.gz'))
mtR_roi.affine
print("Center world coords image 1:", center_world(mtR_roi))
fa_path = op.join(base_path,'models')
fa_img = nib.load(op.join(fa_path, participant+'_ses-04_acq-HCPdir99_model-dki_param-fa_dwimap.nii.gz'))
fa = fa_img.get_fdata()
print("Center world coords image 3:", center_world(fa_img))
print(fa_img.header)
t1w_img.affine
print(t1w_img.header)
print(sts1_mtR.space)
# #Plot 
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting


# # Set up the 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot a few streamlines (for performance)
# max_streamlines = len(sts1_mtL.streamlines)  # adjust as needed
# for i, sl in enumerate(streamlinesL):
#     if i >= max_streamlines:
#         break
#     ax.plot(sl[:, 0], sl[:, 1], sl[:, 2], linewidth=0.5)

# max_streamlines = len(sts1_mtR.streamlines)  # adjust as needed
# for i, sl in enumerate(streamlinesR):
#     if i >= max_streamlines:
#         break
#     ax.plot(sl[:, 0], sl[:, 1], sl[:, 2], linewidth=0.5)
# # Customize axes
# ax.set_title("3D Streamlines (Preview)")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# plt.tight_layout()
# plt.show()



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

scene.add(sts1_mtL_actor)
#scene.add(sts1_mtR_actor)

mtL_actor = actor.contour_from_roi(mtL_dat, color=(1, 0, 0), opacity=0.5)
#mtR_actor = actor.contour_from_roi(mtR_dat, color=(1, 0, 0), opacity=0.5)

# Show in interactive window
scene.add(mtL_actor)
#scene.add(mtR_actor)


for slicer in slicers:
    scene.add(slicer)

#Now visualize:
window.show(scene, size=(1200, 1200), reset_camera=False)
