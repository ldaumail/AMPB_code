#In this script the goals are to create a Wang 2015 volumetric MT ROI for pyAFQ:
#1. combine MT and MST from Wang atlas
#2. Resample the new ROI into ACPC space
#3. Create an inflated version of the combined ROI.
import os
import os.path as op
import ants
import argparse
import sys
import subprocess
import numpy as np
current_dir = op.dirname(op.abspath(__file__))
project_dir = op.abspath(op.join(current_dir, '..'))  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.dilate_mask import dilate_mask

# -------------------------------
## Step 1: Load MT and MST from each hemisphere and combine them
# -------------------------------

def main(participants_file, bids_path):
    mni_wang_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases','Wang_2015')
    # bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
    # fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    # save_dir = op.join(bids_path, 'analysis','ROIs','wang_space-ACPC_rois', 'fsaverage')
    # os.makedirs(save_dir, exist_ok=True)
    # # #First: resample ROI to fsaverage
    # hemis = ["L", "R"]
    # for hemi in hemis:
    #     hemi_fs = "lh" if hemi == "L" else "rh"
    #     #Register and Transform mask from MNI to fs native space
    #     mni_mt_img = ants.image_read(op.join(mni_wang_path, 'subj_vol_all', f"perc_VTPM_vol_roi13_{hemi_fs}.nii.gz"))
    #     mni_mst_img = ants.image_read(op.join(mni_wang_path, 'subj_vol_all', f"perc_VTPM_vol_roi12_{hemi_fs}.nii.gz"))
    # #     mni_yba_data = mni_yba_img.numpy().astype(np.int32)
    #     # binarize
    #     mni_mt_img[mni_mt_img >= 1] = 1 #Peak probability for MT is about 50%
    #     mni_mt_img[mni_mt_img < 1] = 0

    #     mni_mst_img[mni_mst_img >= 1] = 1 #Peak probability for MST is about 40%
    #     mni_mst_img[mni_mst_img < 1] = 0

    #     img1_data = mni_mt_img.numpy()
    #     img2_data = mni_mst_img.numpy()

    #     union_img = (img1_data > 0) | (img2_data > 0) 
    #     union_img = union_img.astype(mni_mt_img.dtype)  # Keep same data type
        
    #     # Convert back to ANTs image
    #     wang_mask_img = ants.from_numpy(union_img, origin=mni_mt_img.origin, spacing=mni_mt_img.spacing, direction=mni_mt_img.direction)

    #     vol_hMT_file = op.join(save_dir, f"hemi-{hemi}_space-mni152_label-hMT_desc-wang.nii.gz")
    #     ants.image_write(wang_mask_img, vol_hMT_file)

    #     input_mask = vol_hMT_file
    #     output_mask = op.join(save_dir,  'hemi-'+hemi+'_space-mni152_desc-hMT_desc-wang_mask_dilated2.nii.gz')

    #     dilate_mask(input_mask, output_mask, dilate = 2)

    #     out_fsaverage_file = op.join(save_dir, f"hemi-{hemi}_space-fsaverage_label-hMT_desc-dil2wang0mm.mgh")

    # #     # Build mri_vol2surf command
    #     cmd = [
    #         "mri_vol2surf",
    #         "--mov", output_mask,
    #         "--regheader", 'fsaverage',
    #         "--hemi", hemi_fs,
    #         "--surf", "white",
    #         "--projdist", "0",
    #         "--sd", fs_path,
    #         "--out", out_fsaverage_file
    #     ]

    #     # Run the command
    #     print("Running:", " ".join(cmd))
    #     subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})

    #     #Save as a label file
    #     fsavg_mask = ants.image_read(out_fsaverage_file).numpy()
    #     output_label = op.join(save_dir, f"hemi-{hemi}_space-fsaverage_label-hMT_desc-dil2wang0mm.label")
    #     roi_verts = np.where(fsavg_mask == 1)[0]
    #     with open(output_label, "w") as f:
    #         f.write(f"#!ascii label  , from subject fsaverage vox2ras=TkReg\n")
    #         f.write(f"{len(roi_verts)}\n")
    #         for vertex in roi_verts: # for each vertex
    #             f.write(f"{vertex} 0 0 0 0\n")

    for participant in participants_file:
        # participant = 'sub-NSxGxHNx1952'
        # bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
        save_dir = op.join(bids_path, 'analysis','ROIs','wang_space-ACPC_rois', participant)
        os.makedirs(save_dir, exist_ok=True)
        qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'anat')
        acpc_t1_path       = op.join(qsiprep_path, participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
        acpc_t1_img       = ants.image_read(acpc_t1_path)
        acpc_brain_mask_img = ants.image_read(op.join(qsiprep_path, participant+'_space-ACPC_desc-brain_mask.nii.gz'))
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