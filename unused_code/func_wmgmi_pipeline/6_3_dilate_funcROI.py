#This script dilates MT masks

from scipy.ndimage import binary_dilation
import nibabel as nib
import numpy as np
import os.path as op
import argparse
import sys
current_dir = op.dirname(op.abspath(__file__))
project_dir = current_dir  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.dilate_mask import dilate_mask

def main(participants_file, roi_name):
    #Provide participants list text file as input
    for participant in participants_file:
        # directories
        paths_local = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis','ampb')
        paths_roi = op.join(paths_local, 'analysis', 'functional_vol_roi', participant) # roi

        mask_list = [participant+'_hemi-L_space-ACPC_label-'+roi_name+'_mask.nii.gz', participant+'_hemi-R_space-ACPC_label-'+roi_name+'_mask.nii.gz']
        dilated_list = [participant+'_hemi-L_space-ACPC_label-'+roi_name+'_mask_dilated.nii.gz', participant+'_hemi-R_space-ACPC_label-'+roi_name+'_mask_dilated.nii.gz']
        for mask, dilated_name in zip(mask_list, dilated_list):
            # define ROIs 
            input_mask = op.join(paths_roi, mask)
            output_mask = op.join(paths_roi, dilated_name)

            # # load the ROI mask
            # mask_img = nib.load(roi_mask)
            # mask_data = mask_img.get_fdata()

            # # dilate the mask over 2 rounds
            # mask_dilated = binary_dilation(mask_data, iterations = dilate)
            # mask_dilated2 = binary_dilation(mask_dilated, iterations = dilate)

            # # save the new mask
            # new_mask_img = nib.Nifti1Image(mask_dilated2.astype(np.uint8), mask_img.affine)
            # nib.save(new_mask_img, op.join(paths_roi, dilated_name))

            dilate_mask(input_mask, output_mask, dilate = 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dilate ACPC masks for a list of participants.")
    parser.add_argument(
        "--participants_file",
        type=str,
        required=True,
        help="Path to a text file containing participant IDs (one per line)."
    )
    parser.add_argument(
        "--roi_name",
        type=str,
        required=True,
        help="Name of the ROI as written in label file name"
    )
    args = parser.parse_args()

    # Read participants from file
    with open(args.participants_file, "r") as f:
        participants = [line.strip() for line in f if line.strip()]

    main(participants, args.roi_name)