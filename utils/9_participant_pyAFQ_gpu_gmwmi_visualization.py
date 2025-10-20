
##----
import ants
import nibabel as nib
import numpy as np
import os.path as op
from fury import window, actor
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import transform_streamlines

# ------------------------------------------------------------
# 1. Define paths
# ------------------------------------------------------------
participant = 'sub-NSxGxIFx1991'
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 
                    'ampb_mt_tractometry_analysis', 'ampb')
fs_path = op.join(bids_path, 'derivatives', 'freesurfer', participant, 'mri')
afq_path = op.join(bids_path, 'derivatives', 'pyAFQ', 'wmgmi', 'LeftMTxLGN', participant)
qsiprep_path = op.join(bids_path,'derivatives', 'qsiprep', participant)

# Files
gmwmi_mask_file = op.join(afq_path, f"{participant}_ses-concat_acq-HCPdir99_desc-wmgmi_mask.nii.gz")
t1w_acpc_file = op.join(qsiprep_path, 'anat', participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
tract_file = op.join(afq_path, 'bundles', f"{participant}_ses-concat_acq-HCPdir99_desc-LeftMTmaskxLGN_tractography.trx")
mt_path = op.join(bids_path, 'analysis','functional_vol_roi', participant, f"{participant}_hemi-L_space-ACPC_label-MT_mask_dilated.nii.gz")

# ------------------------------------------------------------
# 2. Load images
# ------------------------------------------------------------
gmwmi_img = ants.image_read(gmwmi_mask_file)
t1w_img = nib.load(t1w_acpc_file)
t1w_ants_img = ants.image_read(t1w_acpc_file)
mt_lgn_trct = load_tractogram(tract_file, t1w_img)
mt_roi_img = ants.image_read(mt_path) # Load MT ROI file

# 3. Prepare tract for plotting
# streamlinesL = mt_lgn_trct.streamlines
mt_lgn_trct.to_rasmm()

lgn_mtL_t1w = transform_streamlines(mt_lgn_trct.streamlines,
                                np.linalg.inv(t1w_img.affine))

#Visualizing bundles with principal direction coloring

def lines_as_tubes(sl, line_width, **kwargs):
    line_actor = actor.line(sl, **kwargs)
    line_actor.GetProperty().SetRenderLinesAsTubes(1)
    line_actor.GetProperty().SetLineWidth(line_width)
    return line_actor


lgn_mtL_actor = lines_as_tubes(lgn_mtL_t1w, 8)

# Resample the GMWMI mask to the T1w space (reference)
# This ensures it has the same dimensions, voxel size, and orientation
# The `interp_type='nearest_neighbor'` is used for binary masks.
gmwmi_resampled = ants.resample_image_to_target(
    image=gmwmi_img, 
    target=t1w_ants_img, # Use the T1w as the spatial target
    interp_type='nearestNeighbor' # Crucial for binary masks
)

# # Resample the MT ROI mask to the T1w space
mt_roi_resampled = ants.resample_image_to_target(
    image=mt_roi_img,
    target=t1w_ants_img, # Use the same T1w as the spatial target
    interp_type='nearestNeighbor'
)
# ------------------------------------------------------------
# 4. Create FURY actors
# ------------------------------------------------------------
# Background anatomical T1 in diffusion space
## #Slice anatomical image using slicer actors
t1w = t1w_img.get_fdata()
t1_actor = actor.slicer(t1w)

# Smooth contour for GMWMI (binary mask)
gmwmi_actor = actor.contour_from_roi(
    gmwmi_resampled.numpy(),
    color=(1, 0, 0),   # red surface
    opacity=0.5
)

# Smooth contour for MT ROI (binary mask)
mt_roi_actor = actor.contour_from_roi(
    mt_roi_resampled.numpy(),
    color=(0, 0, 1),   # blue surface (change color as desired)
    opacity=0.5
)
# ------------------------------------------------------------
# 5. Visualize
# ------------------------------------------------------------
scene = window.Scene()
scene.add(t1_actor)
scene.add(gmwmi_actor) 
scene.add(lgn_mtL_actor)
scene.add(mt_roi_actor)

scene.reset_camera_tight()
window.show(scene)
