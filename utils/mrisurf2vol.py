 #Write a script that loops through each participant .label files and converts them into
#volumetric ROIs using mrisurf2vol

import os
import os.path as op
import sys

utils = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils'
sys.path.append(op.expanduser(f'{utils}'))

surf_ROI_path = op.join("/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/tests/pyAFQ_tests/wp-afq/analysis/functional_surf_roi")
subjNames = [f for f in os.listdir(surf_ROI_path ) if f not in {".DS_Store"} and not f.endswith((".html"))]

vol_ROI_path = op.join("/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/fonctional_vol_roi")
os.makedirs(vol_ROI_path, exist_ok=True) 
# Loop through each subject folder in the directory
for participant in subjNames:

        #Create registration
        freesurferCommand = f'bbregister --s {participant} --mov {regDir}/refFuncImage.nii.gz --init-fsl --reg {regPath} --bold'
        os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')
        #Create volume 
        output_volume = op.join(vol_ROI_path, participant, participant+"_hemi-L_space-fsnative_label-MT_mask.nii.gz")
        os.makedirs(op.join(vol_ROI_path, participant), exist_ok=True) 
        input_label = op.join(surf_ROI_path, participant,participant+"_hemi-L_space-fsnative_label-MT_mask.label" )
        wm_surface = op.join("/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/freesurfer", participant, "surf/lh.white")

        freesurferCommand = f'mri_surf2vol --o {output_volume} --subject {participant} --so {wm_surface} {input_label}'
        os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')
        mri_surf2vol  --reg anat2epi.dat  --hemi lh --vtxvol vtxvol.nii.gz --fillribbon --outvol foo.nii.gz --template epi.nii.gz --mkmask


regPath = os.path.join(regDir, 'func2surf.dat')
if not os.path.isfile(regPath):
        freesurferCommand = f'bbregister --s {subject} --mov {regDir}/refFuncImage.nii.gz --init-fsl --reg {regPath} --bold'

volumePath = os.path.join(retDir, f'retino_{mapType}.nii.gz')

'''
# special conversion for left hemi angle map
if mapType == 'ang' and hemi == 'lh':
# change range of map (used for left hemisphere)
newVolPath = f'{volumePath[:-7]}_mod.nii.gz'
os.system(f'fslmaths {volumePath} -add 180 -rem 360 {newVolPath}')
volumePath = newVolPath
'''
# convert  to surface
for hemi in ['lh','rh']:
surfacePath = f'{volumePath[:-7]}_{hemi}.mgh'
freesurferCommand = f'mri_vol2surf --mov {volumePath} --out {surfacePath} --reg {regPath} --hemi {hemi} --interp nearest'

