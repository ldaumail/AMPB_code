#This script dilates ROIs
#Loic Daumail 07/28/2025

from scipy.ndimage import binary_dilation
import nibabel as nib
import numpy as np
import os.path as op
import argparse

def main(participant_file):

    for participant in participant_file:
        # participant = 'sub-EBxLxQPx1957'
        # directories
        paths_local = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis','ampb')
        paths_roi = op.join(paths_local, 'analysis', 'julich_space-ACPC_rois', participant, 'ses-04', 'anat') # roi
        mask_list = [participant+'_ses-04_desc-lhHeschl03SyN_mask.nii.gz', participant+'_ses-04_desc-rhHeschl03SyN_mask.nii.gz']
        
        dilated_list = [participant+'_ses-04_desc-lhHeschl03SyN_mask_dilated.nii.gz', participant+'_ses-04_desc-rhHeschl03SyN_mask_dilated.nii.gz']
        for mask, dilated_name in zip(mask_list, dilated_list):
            # define ROIs 
            roi_mask = op.join(paths_roi, mask)

            # load the ROI mask
            mask_img = nib.load(roi_mask)
            mask_data = mask_img.get_fdata()

            # dilate the mask slightly
            mask_dilated = binary_dilation(mask_data)

            # save the new mask
            new_mask_img = nib.Nifti1Image(mask_dilated.astype(np.uint8), mask_img.affine)
            nib.save(new_mask_img, op.join(paths_roi, dilated_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dilate ACPC masks for a list of participants.")
    parser.add_argument(
        "--participants_file",
        type=str,
        required=True,
        help="Path to a text file containing participant IDs (one per line)."
    )
    args = parser.parse_args()

    # Read participants from file
    with open(args.participants_file, "r") as f:
        participants = [line.strip() for line in f if line.strip()]

    main(participants)