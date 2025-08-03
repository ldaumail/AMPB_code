import os
import os.path as op
import nibabel as nib
import numpy as np
import ants

from dipy.io.streamline import load_tractogram, load_trk
from dipy.tracking.streamline import transform_streamlines

participant = 'sub-EBxGxCCx1986'
base_path = op.join('/Volumes','cos-lab-wpark78','LoicDaumail','ampb','derivatives','pyafq', 'gpu-afq_MT-STS1_nseeds20_0mm_nowm_dist3',participant)
bundle_path = op.join(base_path,'bundles')
fa_path = op.join(base_path,'models')

#Upload tracts in
fa_img = nib.load(op.join(fa_path, participant+'_ses-04_acq-HCPdir99_model-dki_param-fa_dwimap.nii.gz'))
fa = fa_img.get_fdata()


sts1_mtL = load_tractogram(op.join(bundle_path,
                          participant+'_ses-04_acq-HCPdir99_desc-STS1xMTL_tractography.trx'), fa_img)
sts1_mtR = load_tractogram(op.join(bundle_path,
                           participant+'_ses-04_acq-HCPdir99_desc-STS1xMTR_tractography.trx'), fa_img)


streamlinesL = sts1_mtL.streamlines
streamlinesR = sts1_mtR.streamlines

# streamlinesL[0].shape
# streamlinesL[1].shape
# streamlinesL[2].shape

# Startpoints: first point in each streamline
startpointsL = np.array([s[0] for s in streamlinesL]) #Start points are negative, so more posterior, thus they are from MT
startpointsR = np.array([s[0] for s in streamlinesR])

# Endpoints: last point in each streamline
endpointsL = np.array([s[-1] for s in streamlinesL]) #End points are positive, so more anterior, thus they are near PT
endpointsR = np.array([s[-1] for s in streamlinesR])

# Calculate endpoint densities at MT ROI boundary

#First get MT ROI boundary coordinates
#Load MT ROI
# mtL_roi =  nib.load(op.join(base_path, 'ROIs', participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-STS1xMTLEnd_mask.nii.gz'))
# mtR_roi =  nib.load(op.join(base_path, 'ROIs', participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-STS1xMTREnd_mask.nii.gz'))

# mtL_dat = mtL_roi.get_fdata()
# mtR_dat = mtR_roi.get_fdata()
# print("Affine 1:\n", mtR_roi.affine)
# print(mtR_roi.header)
# #binarize
# mtL_dat[mtL_dat > 0] = 1
# mtR_dat[mtR_dat > 0] = 1

roi_path = op.join('/Users','ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb', 'analysis', 'functional_vol_roi', participant)
mtL_roi = nib.load(op.join(roi_path, participant+'_hemi-L_space-ACPC_label-MT_mask_dilated.nii.gz'))
mtR_roi = nib.load(op.join(roi_path, participant+'_hemi-R_space-ACPC_label-MT_mask_dilated.nii.gz'))
mtL_dat = mtL_roi.get_fdata()
mtR_dat = mtR_roi.get_fdata()


# print("Affine 2:\n", mtR_roi.affine)
# print(mtR_roi.header)

def center_world(img):
    center_voxel = np.array(img.shape) // 2
    center_world = img.affine @ np.append(center_voxel, 1)
    return center_world[:3]
mtR_roi =  nib.load(op.join(base_path, 'ROIs', participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-STS1xMTREnd_mask.nii.gz'))
print("Center world coords image 1:", center_world(mtR_roi))
mtR_roi = nib.load(op.join(roi_path, participant+'_hemi-R_space-ACPC_label-MT_mask_dilated.nii.gz'))
print("Center world coords image 2:", center_world(mtR_roi))

print("Center world coords image 3:", center_world(fa_img))
print(fa_img.header)

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
scene.add(sts1_mtR_actor)

mtL_actor = actor.contour_from_roi(mtL_dat, color=(1, 0, 0), opacity=0.5)
mtR_actor = actor.contour_from_roi(mtR_dat, color=(1, 0, 0), opacity=0.5)

# Show in interactive window
scene.add(mtL_actor)
scene.add(mtR_actor)


for slicer in slicers:
    scene.add(slicer)

#Now visualize:
window.show(scene, size=(1200, 1200), reset_camera=False)
