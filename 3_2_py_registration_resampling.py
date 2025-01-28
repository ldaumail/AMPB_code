# This script aims at performing a transformation using Antspy
# The transformation of the parcellation from freesurfer to the diffusion space of qsiprep is done
# via the registration of the fs brain to the qsiprep T1.
# the registration is then applied to 

#Loic Daumail 01/21/2024
##use the anatomical output from qsiprep as a reference 
import ants
import os
import os.path as op
import sys

utils = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils'
sys.path.append(op.expanduser(f'{utils}'))

WORK_DIR = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/work'

FIXED_QSIPREPT1 = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-NSxGxHNx1952/anat/sub-NSxGxHNx1952_space-ACPC_desc-preproc_T1w.nii.gz'
QSIPREP_T1_MASK = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-NSxGxHNx1952/anat/sub-NSxGxHNx1952_space-ACPC_desc-brain_mask.nii.gz'

MOV_FSBRAIN_MGZ = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/freesurfer/sub-NSxGxHNx1952/mri/brain.mgz'
MOV_FSPARC_MGZ = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/freesurfer/sub-NSxGxHNx1952/mri/aparc.a2009s+aseg.mgz'

## We need to convert fs brain .mgz to nifti format for registration
# Save directory with NIfTI format
MOV_FSBRAIN_NII = os.path.join(WORK_DIR, "sub-NSxGxHNx1952/brain.nii")
# Run mri_convert
freesurferCommand = f'mri_convert {MOV_FSBRAIN_MGZ} {MOV_FSBRAIN_NII}'
os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')

##2 Run the registration
# regAnat2Diff = ants.registration( movImg, targImg, 'SyN', initial_transform = [targImg, movImg, 1], aff_metric = mattes,reg_iterations = [100,100,20] )

qsiprep_t1 = ants.image_read(FIXED_QSIPREPT1)
fs_brain_nii = ants.image_read(MOV_FSBRAIN_NII)
qsiprep_t1_mask = ants.image_read(QSIPREP_T1_MASK)

# Perform registration
reg = ants.registration(
    fixed=qsiprep_t1,
    moving=fs_brain_nii,
    type_of_transform='Rigid', #"Rigid",  # Corresponds to --transform Rigid[ 0.1 ]
    mask=qsiprep_t1_mask,  # Corresponds to --masks [ ${QSIPREP_T1_MASK} NULL ]
    reg_iterations=(1000, 500, 250, 100),  # Corresponds to --convergence [ 1000x500x250x100, 1e-06, 10 ]
    verbose=True
)

# Save outputs
ants.image_write(reg["warpedmovout"], os.path.join(WORK_DIR, "sub-NSxGxHNx1952/transform_Warped.nii.gz"))
# ants.write_transform(reg["fwdtransforms"], os.path.join(WORK_DIR, "sub-NSxGxHNx1952/fwdtransform"))


mytx = reg['fwdtransforms']

##3 Apply the transformation to parcellation
# Convert .mgz parcellation to nifti
MOV_FSPARC_NII = os.path.join(WORK_DIR, "sub-NSxGxHNx1952/aparc.a2009s+aseg.nii")
# Run mri_convert
freesurferCommand = f'mri_convert {MOV_FSPARC_MGZ} {MOV_FSPARC_NII}'
os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')

fs_parc_nii = ants.image_read(MOV_FSPARC_NII)
fsparc_warped = ants.apply_transforms( fixed = qsiprep_t1, 
                                       moving = fs_parc_nii , 
                                       transformlist = mytx,                                       
                                       interpolator  = 'genericLabel', 
                                       ) #whichtoinvert = [True, False]

ants.plot( qsiprep_t1, fsparc_warped, overlay_alpha = 0.5 )
ants.plot(fsparc_warped )

ants.image_write(fsparc_warped, os.path.join(WORK_DIR, "sub-NSxGxHNx1952/transform_Warped_aparc.a2009s+aseg_to_qsiprepT1.nii.gz"))

# ants.plot(movImg,atlaswarpedimage, overlay_alpha = 0.5)

# Register FreeSurfer brain to QSIPrep T1w with bash shell command line 
# antsRegistration --collapse-output-transforms 1 \
#     --dimensionality 3 --float 0 \
#     --initial-moving-transform [ ${QSIPREP_T1}, ${FS_BRAIN_NII}, 1 ] \
#     --initialize-transforms-per-stage 0 --interpolation BSpline \
#     --output [ ${OUTDIR}/transform, ${OUTDIR}/transform_Warped.nii.gz ] \
#     --transform Rigid[ 0.1 ] \
#     --metric Mattes[ ${QSIPREP_T1}, ${FS_BRAIN_NII}, 1, 32, Random, 0.25 ] \
#     --convergence [ 1000x500x250x100, 1e-06, 10 ] \
#     --smoothing-sigmas 3.0x2.0x1.0x0.0mm --shrink-factors 8x4x2x1 \
#     --use-histogram-matching 0 \
#     --masks [ ${QSIPREP_T1_MASK} NULL ] \
#     --winsorize-image-intensities [ 0.002, 0.998 ] \
#     --write-composite-transform 0

# import ants
# import numpy as np

# #Directories of interest: the dir of the file we want to apply the transformation to, 
# movDir = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/freesurfer/sub-NSxGxHNx1952/mri/aparc.a2009s+aseg.mgz'
# targetDir = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-NSxGxHNx1952/ses-04/dwi/sub-NSxGxHNx1952_ses-04_acq-HCPdir99_space-ACPC_desc-preproc_dwi.nii.gz'
# transformFile = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-NSxGxHNx1952/anat/sub-NSxGxHNx1952_from-MNI152NLPC_mode-image_xfm.h5'

# movImg = ants.image_read(movDir)
# targImg = ants.image_read(targetDir)

# ##1: We need the images to have the same dimensionnality in order to perform the registration.
# #Given that diffusion data 
# # Get the number of volumes in the 4th dimension
# _, _, _, num_volumes = targImg.shape 

# # Find the middle index of the 4th dimension
# middle_index = num_volumes // 2

# # Convert the 4D ANTs image to a NumPy array for slicing
# targ_data = targImg.numpy()

# # Extract the 3D middle volume
# middle_volume_data = targ_data[:, :, :, middle_index]

# midTargImg = ants.from_numpy(middle_volume_data, 
#                             spacing=targImg.spacing[:3],
#                             origin=targImg.origin[:3],  # Use the first 3 elements of the origin
#                             direction=targImg.direction[:3, :3])  # Extract the 3x3 direction matrix)

# ##2 Run the registration
# regAnat2Diff = ants.registration( movImg, midTargImg, 'SyN', reg_iterations = [100,100,20] )
# mytx = regAnat2Diff['invtransforms']

# ##3 Apply the transformation
# atlaswarpedimage = ants.apply_transforms( fixed = midTargImg, 
#                                        moving = movImg , 
#                                        transformlist = mytx, 
#                                        interpolator  = 'genericLabel', 
#                                        whichtoinvert = [True,False])

# ants.plot( midTargImg, atlaswarpedimage, overlay_alpha = 0.5 )

# ants.plot(movImg,atlaswarpedimage, overlay_alpha = 0.5)
