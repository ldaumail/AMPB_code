#In this script the goals are to create a Wang 2015 volumetric MT ROI for pyAFQ:
#1. combine MT and MST from Wang atlas
#2. Resample the new ROI into ACPC space
#3. Create an inflated version of the combined ROI.
import os
import os.path as op
import ants
import numpy as np
import nibabel.freesurfer as fs
import argparse
import sys
import subprocess
current_dir = op.dirname(op.abspath(__file__))
project_dir = op.abspath(op.join(current_dir, '..'))  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.dilate_mask import dilate_mask

# -------------------------------
## Step 1: Load MT and MST from each hemisphere and combine them
# -------------------------------

def main(participants_file, bids_path):
    mni_yba_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases','YBA_696parcels')
    # bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
    # fs_path = op.join(bids_path, 'derivatives', 'freesurfer')

    # save_dir = op.join(bids_path, 'analysis','ROIs', 'YBA_space-ACPC_rois', 'fsaverage')
    # os.makedirs(save_dir, exist_ok=True)
    # #First: resample ROI to fsaverage
    # hemis = ["L", "R"]
    # for hemi in hemis:
    #     hemi_fs = "lh" if hemi == "L" else "rh"
    #     #Register and Transform mask from MNI to fs native space
    #     mni_yba_img = ants.image_read(op.join(mni_yba_path, f"YBA_696.nii"))
    #     mni_yba_data = mni_yba_img.numpy().astype(np.int32)
    #     #atlas_labels = np.unique(mni_yba_data)
    #     lookup = np.zeros(int(mni_yba_data.max()) + 1, dtype=bool)
    #     if hemi == "L":
    #         lookup[[31,32,38,39,45,60,69,70,71, 82, 91]] = True #, 109
    #     # 31: L_T2.1.F probably the most anterior (RH: 379)
    #     # 32: L_T2.1.G seems to overlap a lot with 31 once projected, and quite small but necessary (RH: 380)
    #     # 38: L_T2.2_F anterior, inferior to 31 (not sure if needed) (RH: 386)
    #     # 39: L_T2.2_G high overlap with NS func MT, absolutely needed (RH: 387)
    #     # 45: L_T3.1_F anterior, inferior to 38, very ventral (RH: 393)
    #     # 60: L_O2.1_E posterior, slightly dorsal (RH: 408)
    #     # 69: L_O2.2_I posterior, quite close to NS func MT (RH: 417)
    #     # 70: L_O2.2_J overlaps with NS func MT (RH: 418)
    #     # 71: L_O2.2_K overlaps with NS func MT (RH: 419)
    #     # 82: L_O3.I part overlaps with NS func MT, rest is very ventral (RH: 430)
    #     # 91: L_AN2_D superior, overlaps slightly with NS func MT (RH: 439)
    #     # 109: L_SM5_E superior need it to cover part of right hemisphere func MT (RH: 457)
    #     elif hemi == "R":
    #        lookup[[379,380,386,387,393,408,417,418,419, 430, 439]] = True #numbers for the right hemisphere , 457


    #     lh_MT_mask = lookup[mni_yba_data].astype(np.uint8)

    #     lh_MT_mask_img = ants.from_numpy(lh_MT_mask, origin=mni_yba_img.origin, spacing=mni_yba_img.spacing, direction=mni_yba_img.direction)
    #     vol_hMT_file = op.join(save_dir, f"hemi-{hemi}_space-mni152_label-hMT_desc-yba.nii.gz")
    #     ants.image_write(lh_MT_mask_img, vol_hMT_file)

    #     input_mask = vol_hMT_file
    #     output_mask = op.join(save_dir,  'hemi-'+hemi+'_space-mni152_desc-MT_desc-yba_mask_dilated.nii.gz')

    #     dilate_mask(input_mask, output_mask, dilate = 8)

    #     out_fsaverage_file = op.join(save_dir, f"hemi-{hemi}_space-fsaverage_label-hMT_desc-yba0mm.mgh")

    #     # Build mri_vol2surf command
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
    #     output_label = op.join(save_dir, f"hemi-{hemi}_space-fsaverage_label-hMT_desc-yba0mm.label")
    #     roi_verts = np.where(fsavg_mask == 1)[0]
    #     with open(output_label, "w") as f:
    #         f.write(f"#!ascii label  , from subject fsaverage vox2ras=TkReg\n")
    #         f.write(f"{len(roi_verts)}\n")
    #         for vertex in roi_verts: # for each vertex
    #             f.write(f"{vertex} 0 0 0 0\n")


    for participant in participants_file:
        # participant = 'sub-NSxGxHNx1952'
        # bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
        save_dir = op.join(bids_path, 'analysis','ROIs', 'YBA_space-ACPC_rois', participant)
        os.makedirs(save_dir, exist_ok=True)
        qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'anat')
        acpc_t1_path       = op.join(qsiprep_path, participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
        acpc_t1_img       = ants.image_read(acpc_t1_path)
        acpc_brain_mask_img = ants.image_read(op.join(qsiprep_path, participant+'_space-ACPC_desc-brain_mask.nii.gz'))
        mni_t1_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases','Wang_2015')
        mni_t1_img = ants.image_read(op.join(mni_t1_path, 'MNI152_T1_1mm.nii.gz'))
        
        # if os.path.exists(op.join(save_dir, participant+'_hemi-L_space-ACPC_desc-MT_mask.nii.gz')):
        #     continue
        # MNI to ACPC T1 registration
        reg = ants.registration(
            fixed = acpc_t1_img,
            moving = mni_t1_img,
            type_of_transform = 'SyN',#'SyN', #SyN here, as qsiprep T1 and MNI152NLin2009cAsym are different brains. For same brains, use 'Rigid'
            mask = acpc_brain_mask_img,  
            reg_iterations = (1000, 500, 250, 100),  
            verbose = True
        )

        #Load YBA atlas 
        mni_yba_img = ants.image_read(op.join(mni_yba_path, f"YBA_696.nii"))
        mni_yba_data = mni_yba_img.numpy().astype(np.int32)
        
        for hemi in ["L", "R"]:
        # #First: resample ROI to ACPC
            #Register and Transform mask from MNI to ACPC
            lookup = np.zeros(int(mni_yba_data.max()) + 1, dtype=bool) #Create boolean vector of atlas regions indices 
            if hemi == "L":
                lookup[[31,32,38,39,45,60,69,70,71, 82, 91]] = True 
            # 31: L_T2.1.F probably the most anterior
            # 32: L_T2.1.G seems to overlap a lot with 31 once projected, and quite small but necessary
            # 38: L_T2.2_F anterior, inferior to 31 (not sure if needed)
            # 39: L_T2.2_G high overlap with NS func MT, absolutely needed
            # 45: L_T3.1_F anterior, inferior to 38, very ventral
            # 60: L_O2.1_E posterior, slightly dorsal
            # 69: L_O2.2_I posterior, quite close to NS func MT
            # 70: L_O2.2_J overlaps with NS func MT
            # 71: L_O2.2_K overlaps with NS func MT
            # 82: L_O3.I part overlaps with NS func MT, rest is very ventral
            # 91: L_AN2_D superior, overlaps slightly with NS func MT
            elif hemi == "R":
               lookup[[379,380,386,387,393,408,417,418,419, 430, 439]] = True #numbers for the right hemisphere

            # create hMT mask and binarize
            yba_hMT_mask = lookup[mni_yba_data].astype(np.uint8)

            yba_hMT_mask_img = ants.from_numpy(yba_hMT_mask, origin=mni_yba_img.origin, spacing=mni_yba_img.spacing, direction=mni_yba_img.direction)
            vol_hMT_mask_file = op.join(save_dir, f"{participant}_hemi-{hemi}_space-mni152_label-hMT_desc-yba.nii.gz")
            ants.image_write(yba_hMT_mask_img, vol_hMT_mask_file)

            transformed_hMT_path = op.join(save_dir, participant+'_hemi-'+hemi+'_space-ACPC_desc-MT_mask.nii.gz')
            if os.path.exists(transformed_hMT_path):
                print("File exists!")
            else:
                print("File does not exist. Creating it now")
            

            #Resample binary mask into ACPC
            # Apply transform  MNI mask → ACPC space
            mytx = reg['fwdtransforms']
            transformed_mask = ants.apply_transforms(
                moving = yba_hMT_mask_img, 
                fixed = acpc_t1_img, 
                transformlist = mytx, 
                interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
            )
            # # save transformed mask
            ants.image_write(transformed_mask, transformed_hMT_path)

            #Apply transform to dilated mask
            dilated_mask_file = op.join(save_dir,  participant+'_hemi-'+hemi+'_space-mni152_desc-MT_desc-yba_mask_dilated.nii.gz')
            dilate_mask(vol_hMT_mask_file, dilated_mask_file, dilate = 8)
            dilated_mni_hMT_img = ants.image_read(dilated_mask_file)
            transformed_dilated_mask = ants.apply_transforms(
                moving = dilated_mni_hMT_img, 
                fixed = acpc_t1_img, 
                transformlist = mytx, 
                interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
            )
            # # save transformed mask
            transformed_dilated_hMT_path = op.join(save_dir, participant+'_hemi-'+hemi+'_space-ACPC_desc-MT_mask_dilated.nii.gz')
            ants.image_write(transformed_dilated_mask, transformed_dilated_hMT_path)


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