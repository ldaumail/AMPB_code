#In this script the goals are to create a Wang 2015 volumetric MT ROI for pyAFQ:
#1. combine MT and MST from Wang atlas
#2. Resample the new ROI into ACPC space
#3. Create an inflated version of the combined ROI.
import os
import os.path as op
import ants
import argparse
import sys
current_dir = op.dirname(op.abspath(__file__))
project_dir = op.abspath(op.join(current_dir, '..'))  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.dilate_mask import dilate_mask

# -------------------------------
## Step 1: Load MT and MST from each hemisphere and combine them
# -------------------------------

def main(participants_file, bids_path):

    for participant in participants_file:
        # participant = 'sub-NSxGxHNx1952'
        # bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
        save_dir = op.join(bids_path, 'analysis','wang_space-ACPC_rois', participant)
        os.makedirs(save_dir, exist_ok=True)
        qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'anat')
        acpc_t1_path       = op.join(qsiprep_path, participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
        acpc_t1_img       = ants.image_read(acpc_t1_path)
        acpc_brain_mask_img = ants.image_read(op.join(qsiprep_path, participant+'_space-ACPC_desc-brain_mask.nii.gz'))
        mni_wang_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases','Wang_2015')
        mni_t1_img = ants.image_read(op.join(mni_wang_path, 'MNI152_T1_1mm.nii.gz'))
        
        # MNI to ACPC T1 registration
        reg = ants.registration(
            fixed = acpc_t1_img,
            moving = mni_t1_img,
            type_of_transform = 'SyN',#'SyN', #SyN here, as qsiprep T1 and MNI152NLin2009cAsym are different brains. For same brains, use 'Rigid'
            mask = acpc_brain_mask_img,  
            reg_iterations = (1000, 500, 250, 100),  
            verbose = True
        )

        for hemi_fs in ['lh', 'rh']:
                #Register and Transform mask from MNI to fs native space
                mni_mt_img = ants.image_read(op.join(mni_wang_path, 'subj_vol_all', f"perc_VTPM_vol_roi13_{hemi_fs}.nii.gz"))
                mni_mst_img = ants.image_read(op.join(mni_wang_path, 'subj_vol_all', f"perc_VTPM_vol_roi12_{hemi_fs}.nii.gz"))
                hemi  = "L" if hemi_fs == "lh" else "R"
                transformed_MT_path = op.join(save_dir, participant+'_hemi-'+hemi+'_space-ACPC_desc-MT_mask.nii.gz')
                if os.path.exists(transformed_MT_path):
                    print("File exists!")
                else:
                    print("File does not exist. Creating it now")
                

                # import numpy as np
                # import matplotlib.pyplot as plt

                # # Example array
                # data = mni_mst_img.numpy()
                # # data[data < 0] = 0  # make some zeros

                # # Keep only non-zero values
                # nonzero_vals = data[data != 0]

                # # Plot histogram
                # plt.hist(nonzero_vals, bins=50, color='steelblue', edgecolor='black')
                # plt.xlabel("Value")
                # plt.ylabel("Count")
                # plt.title("Histogram of Non-Zero Values")
                # plt.show()

                # binarize
                mni_mt_img[mni_mt_img >= 1] = 1 #Peak probability for MT is about 50%
                mni_mt_img[mni_mt_img < 1] = 0

                mni_mst_img[mni_mst_img >= 1] = 1 #Peak probability for MST is about 40%
                mni_mst_img[mni_mst_img < 1] = 0


                img1_data = mni_mt_img.numpy()
                img2_data = mni_mst_img.numpy()

                union_img = (img1_data > 0) | (img2_data > 0) 
                union_img = union_img.astype(mni_mt_img.dtype)  # Keep same data type
                
                # Convert back to ANTs image
                wang_mask_img = ants.from_numpy(union_img, origin=mni_mt_img.origin, spacing=mni_mt_img.spacing, direction=mni_mt_img.direction)


                #Resample binary mask into ACPC
                # Apply transform  MNI mask → ACPC space
                mytx = reg['fwdtransforms']
                transformed_mask = ants.apply_transforms(
                    moving = wang_mask_img, 
                    fixed = acpc_t1_img, 
                    transformlist = mytx, 
                    interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
                )
                

                # # save transformed mask
                ants.image_write(transformed_mask, transformed_MT_path)

                ##Dilate and save
                input_mask = transformed_MT_path
                output_mask = op.join(save_dir,  participant+'_hemi-'+hemi+'_space-ACPC_desc-MT_mask_dilated.nii.gz')

                dilate_mask(input_mask, output_mask, dilate = 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACPC masks for a list of participants.")
    parser.add_argument(
        "--participants_file",
        type=str,
        required=True,
        help="Path to a text file containing participant IDs (one per line)."
    )
    parser.add_argument(
        "--bids_path",
        type=str,
        required=True,
        help="Path to bids formated data."
    )
    args = parser.parse_args()

    # Read participants from file
    with open(args.participants_file, "r") as f:
        participants = [line.strip() for line in f if line.strip()]

    main(participants_file = participants, bids_path = args.bids_path)