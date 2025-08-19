 #Write a script that loops through each participant .label MT files 
 # that were functionally defined by Yang Yang, reformats them into binary masks 
 # and converts them into volumetric ROIs with label2vol 
#Loic Daumail 04/18/2024

import os
import os.path as op
import argparse
import sys

def main(participant_file, roi_name):
        paths_local = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis', 'ampb')

        utils = op.join(paths_local, 'code','utils')
        sys.path.append(op.expanduser(f'{utils}'))

        surf_ROI_path =  op.join(paths_local,'analysis','functional_surf_roi')
        #subjNames = [f for f in os.listdir(surf_ROI_path ) if f not in {".DS_Store"} and not f.endswith((".html"))]

        vol_ROI_path = op.join(paths_local,'analysis','functional_vol_roi')
        os.makedirs(vol_ROI_path, exist_ok=True) 

        fs_path =  op.join(paths_local,'derivatives', 'freesurfer')

        # Loop through each subject folder in the directory
        for participant in participant_file:

                # define participant
                # participant='sub-NSxLxYKx1964'
                label_list = [participant+'_hemi-L_space-fsnative_label-'+roi_name+'_mask.label', participant+'_hemi-R_space-fsnative_label-'+roi_name+'_mask.label'] # 
                vol_list = [participant+'_hemi-L_space-fsnative_label-'+roi_name+'_desc-vol_mask.nii.gz', participant+'_hemi-R_space-fsnative_label-'+roi_name+'_desc-vol_mask.nii.gz'] #
                #vol_list = [participant+'_hemi-L_space-fsnative_label-'+roi_name+'_mask.nii.gz', participant+'_hemi-R_space-fsnative_label-'+roi_name+'_mask.nii.gz']

                hemi = ['lh','rh'] #
                temp_path = op.join(fs_path, participant, 'mri', 'brain.mgz')

                for label_file, vol_file, hID in zip(label_list, vol_list, hemi):
                

                        surf_label_file = op.join(surf_ROI_path, participant, label_file)
                        out_mask_file = op.join(vol_ROI_path, participant, vol_file)
                        os.makedirs(op.join(vol_ROI_path, participant), exist_ok=True) 
                        # First, we have to convert labels into binary mask files
                        #freesurferCommand = f'mri_label2label --sd {paths_fsdata} --srclabel {surf_label_file} --trglabel {surf_label_file} --outmask {out_mask_file} --hemi {hID} --s {participant} --regmethod surface'
                        freesurferCommand = f'mri_label2vol --sd {fs_path} --subject {participant} --hemi {hID} --label {surf_label_file} --temp {temp_path} --o {out_mask_file} --proj frac 0 1 .1 --fillthresh 0.75 --fill-ribbon --identity'
                        os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')
                        
                        # call the mri_surf2vol command  
                        # paths_template = op.join(paths_fsdata, participant, 'mri', 'brain.mgz')
                        # output_vol_file = op.join(vol_ROI_path, participant,vol_file)
                        # os.makedirs(op.join(vol_ROI_path, participant), exist_ok=True) 
                        # freesurferCommand = f'mri_surf2vol --sd {paths_fsdata} --surfval {out_mask_file} --o {output_vol_file} --identity {participant} --hemi {hID} --fillribbon --template {paths_template}'
                        # os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate volumetric masks for a list of participants .label files.")
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