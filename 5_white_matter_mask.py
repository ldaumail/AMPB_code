#This script takes white matter segmentations from freesurfer and creates a mask in diffusion
#  (ACPC) space

import os
import os.path as op
import ants
import argparse
import sys
import numpy as np

def main(participant_file):

    for participant in participant_file:
        # participant = 'sub-NSxLxYKx1964'
        paths_local = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis','ampb')
        utils = op.join(paths_local, 'code', 'utils')
        sys.path.append(op.expanduser(f'{utils}'))

        paths_qsiprep = op.join(paths_local, 'derivatives', 'qsiprep', participant, 'anat')
        # paths_ACPC = op.join(paths_local, 'analysis', 'func_space-ACPC_rois', participant)
        # os.makedirs(paths_ACPC, exist_ok=True)

        ## Create freesurfer to ACPC space registration
        # load acpc t1 (fixed)
        acpc_t1 = op.join(paths_qsiprep, participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
        acpc_t1_img = ants.image_read(acpc_t1)

        # load acpc brain mask
        acpc_brain_mask = op.join(paths_qsiprep, participant+'_space-ACPC_desc-brain_mask.nii.gz')
        acpc_brain_mask_img = ants.image_read(acpc_brain_mask)

        # fs t1 (moving)
        fs_t1 = op.join(paths_local, 'derivatives', 'freesurfer', participant, 'mri', 'T1.mgz') #participant+'_ses-01b_acq-MEMPRvNav_rec-RMS_T1w.nii.gz')

        # T1 directory with NIfTI format
        paths_T1 = op.join(paths_local, 'analysis', 'functional_vol_roi', participant)
        fs_t1_nii = os.path.join(paths_T1, 'T1.nii')
        

        fs_t1_img = ants.image_read(fs_t1_nii)

        # registration
        reg = ants.registration(
            fixed = acpc_t1_img,
            moving = fs_t1_img,
            type_of_transform = 'Rigid',#'SyN'  For same brains, use 'Rigid'
            mask = acpc_brain_mask_img,  
            reg_iterations = (1000, 500, 250, 100),  
            verbose = True
        )

        #convert wm format to nifti
        paths_acpc_wm = op.join(paths_local, 'analysis', 'acpc_wm', participant)
        os.makedirs(paths_acpc_wm, exist_ok=True)

        fs_wm = os.path.join(paths_local, 'derivatives', 'freesurfer', participant, 'mri', 'wm.mgz')
        fs_wm_nii = os.path.join(paths_acpc_wm, 'wm.nii')
        # Run mri_convert
        freesurferCommand = f'mri_convert {fs_wm} {fs_wm_nii}'
        os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')
            
        ## Create binary masks resampled from freesurfer to ACPC space
        transformed_wm = participant+'_space-ACPC_label-WM_mask.nii.gz'
        transformed_mask_path = op.join(paths_acpc_wm, transformed_wm )

        if os.path.exists(transformed_mask_path):
            print("File exists!")
        else:
            print("File does not exist. Creating it now")
            # load wm mask
            wm_mask_img = ants.image_read(fs_wm_nii)

            #store matrix data in np array
            np_img = wm_mask_img.numpy()
            #Only keep white matter 
            np_img[np_img == 250] = 0 #250 here seems to label the CSF, we don't want it
            np_img[np_img != 0] = 1

            #convert back to an ants image object
            np_img = np_img.astype(wm_mask_img.dtype) 
            ants_mask_img = ants.from_numpy(np_img, origin=wm_mask_img.origin, spacing=wm_mask_img.spacing, direction=wm_mask_img.direction)

            # apply transformation: functional mask in freesurfer space → ACPC space
            mytx = reg['fwdtransforms']
            transformed_mask = ants.apply_transforms(
                moving = ants_mask_img, 
                fixed = acpc_t1_img, 
                transformlist = mytx, 
                interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
            )


            # save transformed mask
            ants.image_write(transformed_mask, transformed_mask_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate white matter ACPC masks for a list of participants.")
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
