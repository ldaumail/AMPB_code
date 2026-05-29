#Here we generate the thalamocortical rois and visuzalize them within MNI space

import os.path as op
import ants
import numpy as np
from fury import window, actor

mni_aicha_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases','AICHA')

#Load atlas file
mni_aicha = ants.image_read(op.join(mni_aicha_path, 'AICHA1mm.nii'))

# lh_rois = [367, 369, 371, 373, 375, 377, 379, 381, 383]
# rh_rois = [368, 370, 372, 374, 376, 378, 380, 382, 384]
#
#
lh_rois = [##-----ROIs for PTRs:
           77, 79, 81, 83, 85, #Parietal sup (right above inferior parietal)
           107, #Parietal inferior (pretty much located on parietal cortex)
           109, 111, 113, ##Intraparietal 
           115, # Intra occipital (superior part of occipital)
           117, #occipital pole
           119, 121, 123, 125, 127, #occipital regions/lateral
           129, 131, #Occipital sup
           133, 135, 137, 139, #Occipital mid (close to MT)
           141, 143, #Occipital inf
           283, 285, 287, 291, 293, #Parietooccipital (towards the superior edge of hemisphere)
           
           ##-----Additional ROIs for hMT+ overlap
           193, 195,#G_Temporal_Inf-4-L/5-L
           185, #G_Temporal_Mid-4-L
           177, 175, #S_Sup_Temporal-5-L/4-L
           99, #G_SupraMarginal-7-L
           103,105,# #G_Angular
           ]
rh_rois = [
           78, 80, 82, 84, 86,
           108, 
           110, 112, 114, 
           116, 
           118, 
           120, 122, 124, 126, 128, 
           130, 132, 
           134, 136, 138, 140, 
           142, 144, 
           284, 286, 288, 292, 294,
           ##
           194,196,
           186,
           178, 176,
           100,
           104,106
           ]
lh_cand_roi = [195]
rh_cand_roi = [196]
roi_name = "posterior"
for hemi_fs in ['lh', 'rh']:

    # Register and Transform mask from MNI to fs native space
    rois = lh_rois if hemi_fs == "lh" else rh_rois
    hemi = "L" if hemi_fs == "lh" else "R"

    # ---------------------------------------
    # Build binary ROI mask
    # ---------------------------------------
    mni_img = mni_aicha * 0
    for roi in rois:
        mni_img = mni_img + (mni_aicha == roi)

    # Ensure binary uint8 mask
    mni_img = (mni_img > 0).astype("uint8")

    print(f"\nOriginal resolution for hemi-{hemi}:")
    print(f"Shape: {mni_img.shape}")
    print(f"Spacing: {mni_img.spacing}")


    # ---------------------------------------
    # Save 1 mm ROI mask
    # ---------------------------------------
    mni_roi_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases', 'AICHA', 'mni_rois',
        f"hemi-{hemi}_space-mni_desc-{roi_name}_mask.nii.gz"
    )

    ants.image_write(mni_img, mni_roi_path)
    print(f"\nSaved: {mni_roi_path}")

    candidate_mni_roi = mni_aicha * 0
    cand_roi = lh_cand_roi if hemi_fs == "lh" else rh_cand_roi
    candidate_mni_roi = candidate_mni_roi + (mni_aicha == cand_roi)
    cand_roi_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases', 'AICHA', 'mni_rois',
        f"hemi-{hemi}_space-mni_desc-candidate_mask.nii.gz"
    )

    ants.image_write(candidate_mni_roi, cand_roi_path)
    print(f"\nSaved: {cand_roi_path}")
    
# -------------------------------------------------------------------------
# 6. ROI loading helper
# -------------------------------------------------------------------------
#bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research','ampb_mt_tractometry_analysis', 'ampb')
def roi_actor(roi_path, color):
    if not op.exists(roi_path):
        print(f"⚠️ Missing ROI: {roi_path}")
        return None
    roi_img = ants.image_read(roi_path)
    # roi_resampled = ants.resample_image_to_target(roi_img, t1w_ants_img, interp_type='nearestNeighbor')
    return actor.contour_from_roi(roi_img.numpy(), color=color, opacity=0.4)

# -------------------------------------------------------------------------
# 7. ROI definitions
# -------------------------------------------------------------------------
roi_defs = {"MT":    (op.join('/Users','ldaumail3', 'Documents','research','brain_atlases', 'Wang_2015', 'hmtplus'), 'hMT', (1, 0, 0)),
            "posteriorCortex": (op.join('/Users','ldaumail3', 'Documents','research','brain_atlases','AICHA', 'mni_rois'), 'posterior', (0,1,0)),
            "candidateRoi": (op.join('/Users','ldaumail3', 'Documents','research','brain_atlases','AICHA', 'mni_rois'), 'candidate', (0,0,1))
            }
roi_actors = []

for hemi in ["L", "R"]: # "R"
    for roi_name, (mni_roi_path, label, color) in roi_defs.items():
        if roi_name == "MT":
            roi_path = op.join(mni_roi_path,
                               f"hemi-{hemi}_space-mni_label-{label}_desc-wangvol_dilated.nii.gz")
                                 
        else:
            roi_path = op.join(mni_roi_path, f"hemi-{hemi}_space-mni_desc-{label}_mask.nii.gz")
        roi_act = roi_actor(roi_path, color)
        if roi_act:
            roi_actors.append(roi_act)
                # -----------------------------
        # Load ROI image and print info
        # -----------------------------
        if op.exists(roi_path):
            roi_img = ants.image_read(roi_path)

            print(f"\nLoaded ROI: {roi_name}")
            print(f"Path: {roi_path}")
            print(f"Shape: {roi_img.shape}")
            print(f"Spacing (resolution): {roi_img.spacing}")
            print(f"Origin: {roi_img.origin}")
            print(f"Direction:\n{roi_img.direction}")

        else:
            print(f"\nROI file does not exist: {roi_path}")

t1w_mni_file = op.join("/Users/ldaumail3/Documents/research/brain_atlases/Wang_2015/MNI152_T1_1mm.nii.gz")
t1w_ants_img = ants.image_read(t1w_mni_file)
t1_actor = actor.slicer(t1w_ants_img.numpy())

# -------------------------------------------------------------------------
# wmgmi_resampled = ants.resample_image_to_target(
#     image=wmgmi_img, 
#     target=t1w_ants_img, # Use the T1w as the spatial target
#     interp_type='nearestNeighbor' # Crucial for binary masks
# )

# # Smooth contour for GMWMI (binary mask)
# wmgmi_actor = actor.contour_from_roi(
#     wmgmi_resampled.numpy(),
#     color=(1, 0, 0),   # red surface
#     opacity=0.5
# )
# -------------------------------------------------------------------------
# 8. Build and render scene
# -------------------------------------------------------------------------
scene = window.Scene()
for act in roi_actors:
    scene.add(act)
scene.add(t1_actor)
# scene.add(wmgmi_actor)
scene.reset_camera_tight()
scene.background((0, 0, 0))
window.show(scene)

