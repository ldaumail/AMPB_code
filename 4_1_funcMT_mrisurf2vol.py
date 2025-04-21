 #Write a script that loops through each participant .label MT files 
 # that were functionally defined by Yang Yang, reformats them into binary masks 
 # with label2label and converts them into
#volumetric ROIs using mrisurf2vol
#Loic Daumail 04/18/2024

import os
import os.path as op
import sys


paths_local = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis', 'ampb')

utils = op.join(paths_local, 'code','utils')
sys.path.append(op.expanduser(f'{utils}'))

surf_ROI_path =  op.join(paths_local,'analysis','functional_surf_roi')
subjNames = [f for f in os.listdir(surf_ROI_path ) if f not in {".DS_Store"} and not f.endswith((".html"))]

vol_ROI_path = op.join(paths_local,'analysis','functional_vol_roi')
os.makedirs(vol_ROI_path, exist_ok=True) 


# Loop through each subject folder in the directory
for participant in subjNames:

        # define participant
        # participant='sub-NSxLxYKx1964'
        label_list = [participant+'_hemi-L_space-fsnative_label-MT_mask.label', participant+'_hemi-R_space-fsnative_label-MT_mask.label'] # 
        mask_list = [participant+'_hemi-L_space-fsnative_label-MT_mask.nii.gz', participant+'_hemi-R_space-fsnative_label-MT_mask.nii.gz'] #
        vol_list = [participant+'_hemi-L_space-fsnative_label-MT_desc-vol_mask.nii.gz', participant+'_hemi-R_space-fsnative_label-MT_desc-vol_mask.nii.gz'] #
        hemi = ['lh','rh'] #

        for label_file, mask_file, vol_file, hID in zip(label_list, mask_list, vol_list, hemi):
                paths_fsdata = op.join(paths_local, 'derivatives', 'freesurfer')

                surf_label_file = op.join(surf_ROI_path, participant, label_file)
                out_mask_file = op.join(surf_ROI_path, participant, mask_file)

                # First, we have to convert labels into binary mask files
                freesurferCommand = f'mri_label2label --sd {paths_fsdata} --srclabel {surf_label_file} --trglabel {surf_label_file} --outmask {out_mask_file} --hemi {hID} --s {participant} --regmethod surface'
                os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"') #--trglabel could also be a junk file as not needed here

                # call the mri_surf2vol command  
                paths_template = op.join(paths_fsdata, participant, 'mri', 'brain.mgz')
                output_vol_file = op.join(vol_ROI_path, participant,vol_file)
                os.makedirs(op.join(vol_ROI_path, participant), exist_ok=True) 
                freesurferCommand = f'mri_surf2vol --sd {paths_fsdata} --surfval {out_mask_file} --o {output_vol_file} --identity {participant} --hemi {hID} --fillribbon --template {paths_template}'
                os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')

