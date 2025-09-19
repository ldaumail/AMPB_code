import ants
import os.path as op
import os
import numpy as np
from fury import window, actor, colormap
from nibabel.freesurfer import read_geometry

participant = 'sub-EBxGxCCx1986'
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
parc_seg_path = op.join(bids_path, 'derivatives', 'freesurfer', participant, 'mri', 'aparc+aseg.mgz')
t1_path       = op.join(fs_path, participant, 'mri', 'T1.mgz')

parc_seg_img = ants.image_read(parc_seg_path)
t1_img       = ants.image_read(t1_path)

hemisphere = ['L', 'R']
keep_labels  = [1015,  # left hemisphere
                2015] # right nemisphere
mask_images = {}
for h, label in enumerate(keep_labels): # for each label to keep
    mt_mask = np.zeros(parc_seg_img.shape) # initialize
    mt_mask = np.logical_or(mt_mask, parc_seg_img.numpy() == label)
    hemi = hemisphere[h] # 'R'
    hemi_fs = "lh" if hemi == "L" else "rh"
    mask_name = f"{hemi_fs}_mt_mask"
    # globals()[mask_name] = ants.new_image_like(image = parc_seg_img, data = mt_mask * 1.0)  

    # output_fname = op.join(bids_path, 'analysis', 'freesurfer_rois', participant, f"{participant}_hemi-{hemi}_space-freesurfer_label-MT.mgh")
    # os.makedirs(op.join(bids_path, 'analysis', 'freesurfer_rois', participant), exist_ok=True)
    # ants.image_write( globals()[mask_name], output_fname)
    # print(f"Saved: {output_fname}")
    # print(rh_mt_mask)
    mask_img = ants.new_image_like(parc_seg_img, mt_mask.astype(float))
    mask_images[mask_name] = mask_img

# ----------------------------
# Convert ANTs masks -> FURY volume actors
# ----------------------------
# Background anatomical T1 actor
t1_actor = actor.slicer(t1_img.numpy())

# ROI actors
lh_mt_actor = actor.contour_from_roi(
    mask_images["lh_mt_mask"].numpy(), color=(1, 0, 0), opacity=0.8
)
rh_mt_actor = actor.contour_from_roi(
    mask_images["rh_mt_mask"].numpy(), color=(0, 0, 1), opacity=0.8
)

# ----------------------------
# Scene
# ----------------------------
scene = window.Scene()
scene.add(t1_actor)       # anatomical background
scene.add(lh_mt_actor)    # left MT ROI
scene.add(rh_mt_actor)    # right MT ROI

window.show(scene)

##

parc_seg_path = op.join(bids_path, 'derivatives', 'freesurfer', participant, 'mri', 'aparc.a2009s+aseg.mgz')
t1_path       = op.join(fs_path, participant, 'mri', 'T1.mgz')

parc_seg_img = ants.image_read(parc_seg_path)
t1_img       = ants.image_read(t1_path)

hemisphere = ['L', 'R']
keep_labels  = [11138,  # left hemisphere
                12138] # right nemisphere
mask_images = {}
for h, label in enumerate(keep_labels): # for each label to keep
    mt_mask = np.zeros(parc_seg_img.shape) # initialize
    mt_mask = np.logical_or(mt_mask, parc_seg_img.numpy() == label)
    hemi = hemisphere[h] # 'R'
    hemi_fs = "lh" if hemi == "L" else "rh"
    mask_name = f"{hemi_fs}_mt_mask"
    # globals()[mask_name] = ants.new_image_like(image = parc_seg_img, data = mt_mask * 1.0)  

    # output_fname = op.join(bids_path, 'analysis', 'freesurfer_rois', participant, f"{participant}_hemi-{hemi}_space-freesurfer_label-MT.mgh")
    # os.makedirs(op.join(bids_path, 'analysis', 'freesurfer_rois', participant), exist_ok=True)
    # ants.image_write( globals()[mask_name], output_fname)
    # print(f"Saved: {output_fname}")
    # print(rh_mt_mask)
    mask_img = ants.new_image_like(parc_seg_img, mt_mask.astype(float))
    mask_images[mask_name] = mask_img

# ----------------------------
# Convert ANTs masks -> FURY volume actors
# ----------------------------
# Background anatomical T1 actor
t1_actor = actor.slicer(t1_img.numpy())

# ROI actors
lh_mt_actor = actor.contour_from_roi(
    mask_images["lh_mt_mask"].numpy(), color=(1, 0, 0), opacity=0.8
)
rh_mt_actor = actor.contour_from_roi(
    mask_images["rh_mt_mask"].numpy(), color=(0, 0, 1), opacity=0.8
)

# ----------------------------
# Scene
# ----------------------------
scene = window.Scene()
scene.add(t1_actor)       # anatomical background
scene.add(lh_mt_actor)    # left MT ROI
scene.add(rh_mt_actor)    # right MT ROI

window.show(scene)


