
 #Write a script that loops through each participant .label files and converts them into
#volumetric ROIs using mrisurf2vol

import os
import os.path as op
import sys

utils = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils'
sys.path.append(op.expanduser(f'{utils}'))

surf_ROI_path = op.join("/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/fonctional_surf_roi")
subjNames = [f for f in os.listdir(surf_ROI_path ) if f not in {} and not f.endswith((".html"))]

vol_ROI_path = op.join("/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/fonctional_vol_roi")
os.makedirs(vol_ROI_path, exist_ok=True) 
# Loop through each subject folder in the directory
for participant in subjNames:
        output_volume = op.join(vol_ROI_path, participant, participant+"_hemi-L_space-fsnative_label-MT_mask.nii.gz")
        os.makedirs(op.join(vol_ROI_path, participant), exist_ok=True) 
        input_label = op.join(surf_ROI_path, participant,participant+"_hemi-L_space-fsnative_label-MT_mask.label" )
        t1_template = op.join("/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/freesurfer", participant, "mri/T1.mgz")
        hemi = "lh"
        freesurferCommand = f'mri_surf2vol --o {output_volume} --subject {participant} --so {input_label} --hemi {hemi} --template {t1_template}'
        os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')



