#This script aims to resample MT ROI and other ROIs into the diffusion space based on 
# registrations from FS brain to QSIPREP T1 (and other registrations to QSIPREP T1 depending
# on ROI original space).
# Loic 01/29/2025

import ants
import numpy as np
import os
WORK_DIR = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/work'

# Step 1: Load the Atlas Image
atlas_path = "/Users/ldaumail3/Documents/research/brain_atlases/MNI2009a_GM_Glasser_2016/HCP-MMP1_on_MNI152_ICBM2009a_nlin_hd.nii.gz"  # Replace with your atlas file
atlas_img = ants.image_read(atlas_path)

# Step 2: Define the ROI label (Check the atlas' label map for correct IDs)
roi_label = 23 | 2 | 157  # Replace with the label corresponding to your ROI

# Step 3: Create a Binary Mask for the ROI
roi_mask = atlas_img.numpy() == roi_label  # Boolean mask
roi_mask_img = ants.from_numpy(roi_mask.astype(np.uint8), origin=atlas_img.origin, spacing=atlas_img.spacing, direction=atlas_img.direction)

# Step 4: Extract ROI Voxel Values
roi_voxel_values = atlas_img.numpy()[roi_mask]


# Step 5: Resample ROI to ACPC space
FIXED_QSIPREPT1 = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-NSxGxHNx1952/anat/sub-NSxGxHNx1952_space-ACPC_desc-preproc_T1w.nii.gz'
qsiprep_t1 = ants.image_read(FIXED_QSIPREPT1)

mytx = "/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-NSxGxHNx1952/anat/sub-NSxGxHNx1952_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5"

roi_mask_img_warped = ants.apply_transforms( fixed = qsiprep_t1, 
                                       moving = roi_mask_img, 
                                       transformlist = mytx,                                       
                                       interpolator  = 'genericLabel', 
                                       ) #whichtoinvert = [True, False]


# Save the ROI mask if needed
roi_mask_path = os.path.join(WORK_DIR, "sub-NSxGxHNx1952/MT_roi_glasser_mask_space-ACPC.nii.gz")
ants.image_write(roi_mask_img_warped, roi_mask_path)


#Step 5 bis:
WORK_DIR = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/work'

FIXED_QSIPREPT1 = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-EBxGxCCx1986/anat/sub-EBxGxCCx1986_space-ACPC_desc-preproc_T1w.nii.gz'
qsiprep_t1 = ants.image_read(FIXED_QSIPREPT1)

mytx = "/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-EBxGxCCx1986/anat/sub-EBxGxCCx1986_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5"

roi_mask_img_warped = ants.apply_transforms( fixed = qsiprep_t1, 
                                       moving = roi_mask_img, 
                                       transformlist = mytx,                                       
                                       interpolator  = 'genericLabel', 
                                       ) #whichtoinvert = [True, False]


# Save the ROI mask if needed
roi_mask_path = os.path.join(WORK_DIR, "sub-EBxGxCCx1986/MT_roi_glasser_mask_space-ACPC.nii.gz")
ants.image_write(roi_mask_img_warped, roi_mask_path)
