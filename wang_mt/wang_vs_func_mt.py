
#This script goal is to check whether MT from Wang 2015 atls is at least as big or bigger than functional MT

import os.path as op
import ants
from fury import window, actor


participant = 'sub-NSxGxHNx1952'
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
mni_wang_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases','Wang_2015')
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
mask_images = {}
hemisphere = ['L', 'R']
roi_list = ['roi12', 'roi13']

for roi in roi_list:
    for h, label in enumerate(hemisphere): # for each label to keep
        hemi = hemisphere[h] # 'R'
        hemi_fs = "lh" if hemi == "L" else "rh"
        # roi = roi_list[h]
        mni_mask_img = ants.image_read(op.join(mni_wang_path, 'subj_vol_all', f"perc_VTPM_vol_{roi}_{hemi_fs}.nii.gz"))

        mask_name = f"{hemi_fs}_MST" if '12' in roi else f"{hemi_fs}_MT"
        #Register and Transform mask from MNI to fs native space

            # Apply transform  MNI mask → ACPC space
        mytx = reg['fwdtransforms']
        transformed_mask = ants.apply_transforms(
            moving = mni_mask_img, 
            fixed = acpc_t1_img, 
            transformlist = mytx, 
            interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
        )

        mask_images[mask_name] = transformed_mask


            # # save transformed mask
            # os.makedirs(op.join(paths_ACPC, 'ses-concat', 'anat'), exist_ok=True)
            # ants.image_write(transformed_mask, transformed_mask_path)



# ----------------------------
# Convert ANTs masks -> FURY volume actors
# ----------------------------
# Background anatomical T1 actor
t1_actor = actor.slicer(acpc_t1_img.numpy())

# ROI actors
wang_lh_mt_actor = actor.contour_from_roi(
    mask_images["lh_MT"].numpy(), color=(0, 0, 1), opacity=0.8
)
wang_rh_mt_actor = actor.contour_from_roi(
    mask_images["rh_MT"].numpy(), color=(0, 0, 1), opacity=0.8
)

wang_lh_mst_actor = actor.contour_from_roi(
    mask_images["lh_MST"].numpy(), color=(1, 0, 0), opacity=0.8
)
wang_rh_mst_actor = actor.contour_from_roi(
    mask_images["rh_MST"].numpy(), color=(1, 0, 0), opacity=0.8
)

# ---------------
## Functional MT
# ---------------
func_mt_path = op.join(bids_path, 'analysis', 'functional_vol_roi', participant)
func_mask_images = {}
for hemi in hemisphere:
    mask_name = f"{hemi}_mt"
    mask_img = ants.image_read(op.join(func_mt_path, f"{participant}_hemi-{hemi}_space-ACPC_label-MT_mask.nii.gz"))
    func_mask_images[mask_name] = mask_img
# Func ROI actors
L_mt_actor = actor.contour_from_roi(
    func_mask_images["L_mt"].numpy(), color=(0, 1, 0), opacity=0.8
)
R_mt_actor = actor.contour_from_roi(
    func_mask_images["R_mt"].numpy(), color=(0, 1, 0), opacity=0.8
)

# -----------
## WMGMI
# -----------

wmgmi_path =  op.join(bids_path, 'derivatives', 'pyAFQ', 'wmgmi', 'LeftMTxFEF', participant, participant+'_ses-concat_acq-HCPdir99_desc-wmgmi_mask.nii.gz' )
# --- Load your ROI (WMGMI file) ---

wmgmi_img = ants.image_read(wmgmi_path)
wmgmi_resampled = ants.resample_image_to_target(
    image=wmgmi_img, 
    target=acpc_t1_img, # Use the T1w as the spatial target
    interp_type='nearestNeighbor' # Crucial for binary masks
)

# Smooth contour for GMWMI (binary mask)
wmgmi_actor = actor.contour_from_roi(
    wmgmi_resampled.numpy(),
    color=(1, 0, 0),   # red surface
    opacity=0.5
)
# ----------------------------
# Scene
# ----------------------------
scene = window.Scene()
scene.add(t1_actor)       # anatomical background
scene.add(wang_lh_mt_actor)    # left MT ROI
scene.add(wang_rh_mt_actor)    # right MT ROI
scene.add(wang_lh_mst_actor)    # left MST ROI
scene.add(wang_rh_mst_actor)    # right MST ROI
scene.add(L_mt_actor)    # left MT ROI
scene.add(R_mt_actor)    # right MT ROI
scene.add(wmgmi_actor)
window.show(scene)

