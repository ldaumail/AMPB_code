#This script aims at performing a transformation using Antspyx.
#Loic Daumail 01/21/2024

import ants
import numpy as np
# img1 = ants.image_read( ants.get_ants_data('r16') )
# img2 = ants.image_read( ants.get_ants_data('r27') )
# img3 = ants.image_read( ants.get_ants_data('r64') )

# seg1 = ants.kmeans_segmentation( img1, 3 )
# ants.plot( img1, seg1['segmentation'] )

# reg12 = ants.registration( img1, img2, 'SyN', reg_iterations = [100,100,20] )
# reg23 = ants.registration( img2, img3, 'SyN', reg_iterations = [100,100,20] )

# mytx = reg23[ 'invtransforms'] + reg12[ 'invtransforms'] 

# mywarpedimage = ants.apply_transforms( fixed = img3, 
#                                        moving = seg1['segmentation'] , 
#                                        transformlist = mytx, 
#                                        interpolator  = 'nearestNeighbor', 
#                                        whichtoinvert = [True,False,True,False])
# ants.plot( img3, mywarpedimage, overlay_alpha = 0.5 )



#Directories of interest: the dir of the file we want to apply the transformation to, 
movDir = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/freesurfer/sub-NSxGxHNx1952/mri/aparc.a2009s+aseg.mgz'
targetDir = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-NSxGxHNx1952/ses-04/dwi/sub-NSxGxHNx1952_ses-04_acq-HCPdir99_space-ACPC_desc-preproc_dwi.nii.gz'
transformFile = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-NSxGxHNx1952/anat/sub-NSxGxHNx1952_from-MNI152NLPC_mode-image_xfm.h5'

movImg = ants.image_read(movDir)
targImg = ants.image_read(targetDir)

##1: We need the images to have the same dimensionnality in order to perform the registration.
#Given that diffusion data 
# Get the number of volumes in the 4th dimension
_, _, _, num_volumes = targImg.shape 

# Find the middle index of the 4th dimension
middle_index = num_volumes // 2

# Convert the 4D ANTs image to a NumPy array for slicing
targ_data = targImg.numpy()

# Extract the 3D middle volume
middle_volume_data = targ_data[:, :, :, middle_index]

midTargImg = ants.from_numpy(middle_volume_data, 
                            spacing=targImg.spacing[:3],
                            origin=targImg.origin[:3],  # Use the first 3 elements of the origin
                            direction=targImg.direction[:3, :3])  # Extract the 3x3 direction matrix)

##2 Run the registration
regAnat2Diff = ants.registration( movImg, midTargImg, 'SyN', reg_iterations = [100,100,20] )
mytx = regAnat2Diff['invtransforms']

##3 Apply the transformation
atlaswarpedimage = ants.apply_transforms( fixed = midTargImg, 
                                       moving = movImg , 
                                       transformlist = mytx, 
                                       interpolator  = 'nearestNeighbor', 
                                       whichtoinvert = [True,False])

ants.plot( midTargImg, atlaswarpedimage, overlay_alpha = 0.5 )

ants.plot(atlaswarpedimage)

##3 bis: Apply the transform using the transform file output from qsiprep
# mytx = ['/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-NSxGxHNx1952/anat/sub-NSxGxHNx1952_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5']
# atlaswarpedimage = ants.apply_transforms( fixed = midTargImg, 
#                                        moving = movImg , 
#                                        transformlist = mytx, 
#                                        interpolator  = 'nearestNeighbor', 
#                                        whichtoinvert = [False])
# ants.plot( midTargImg, atlaswarpedimage, overlay_alpha = 0.5 )

