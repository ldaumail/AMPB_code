#This script performs resampling of area MT from fs space to ACPC
import os
import os.path as op
import ants
import argparse
import sys


def main(participant_file):

    for participant in participant_file:
        # participant = 'sub-NSxLxYKx1964'
        # directories
        paths_local = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis','ampb')
        utils = op.join(paths_local, 'code', 'utils')
        sys.path.append(op.expanduser(f'{utils}'))

        paths_qsiprep = op.join(paths_local, 'derivatives', 'qsiprep', participant, 'anat')
        # paths_ACPC = op.join(paths_local, 'analysis', 'func_space-ACPC_rois', participant)
        # os.makedirs(paths_ACPC, exist_ok=True)
        paths_func = op.join(paths_local, 'analysis', 'functional_vol_roi', participant)

        ## Create freesurfer to ACPC space registration
        # load acpc t1 (fixed)
        acpc_t1 = op.join(paths_qsiprep, participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
        acpc_t1_img = ants.image_read(acpc_t1)

        # load acpc brain mask
        acpc_brain_mask = op.join(paths_qsiprep, participant+'_space-ACPC_desc-brain_mask.nii.gz')
        acpc_brain_mask_img = ants.image_read(acpc_brain_mask)

        # fs t1 (moving)
        fs_t1 = op.join(paths_local, 'derivatives', 'freesurfer', participant, 'mri', 'T1.mgz') #participant+'_ses-01b_acq-MEMPRvNav_rec-RMS_T1w.nii.gz')
        
        # Save directory with NIfTI format
        fs_t1_nii = os.path.join(paths_func, 'T1.nii')
        # Run mri_convert
        freesurferCommand = f'mri_convert {fs_t1} {fs_t1_nii}'
        os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')
                
        
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

        ## Create binary masks resampled from MNI to ACPC space
        roi_list = [participant+'_hemi-L_space-fsnative_label-MT_desc-vol_mask.nii.gz', participant+'_hemi-R_space-fsnative_label-MT_desc-vol_mask.nii.gz']
        transformed_list = [participant+'_hemi-L_space-ACPC_label-MT_mask.nii.gz', participant+'_hemi-R_space-ACPC_label-MT_mask.nii.gz']


        for roi, transmask in zip(roi_list, transformed_list):
            transformed_mask_path = op.join(paths_func, transmask)
            if os.path.exists(transformed_mask_path):
                print("File exists!")
            else:
                print("File does not exist. Creating it now")
                # load julich mask
                func_mask = op.join(paths_func, roi)
                func_mask_img = ants.image_read(func_mask)


                # apply transformation: functional mask in freesurfer space → ACPC space
                mytx = reg['fwdtransforms']
                transformed_mask = ants.apply_transforms(
                    moving = func_mask_img, 
                    fixed = acpc_t1_img, 
                    transformlist = mytx, 
                    interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
                )


                # save transformed mask
                ants.image_write(transformed_mask, transformed_mask_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACPC masks for a list of participants.")
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