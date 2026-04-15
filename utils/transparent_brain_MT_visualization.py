


import ants
import numpy as np
import os
import os.path as op
from fury import window, actor

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
participant = 'sub-NSxGxHNx1952'

bids_path = op.join(
    '/Users','ldaumail3','Documents','research',
    'ampb_mt_tractometry_analysis','ampb'
)

fs_path = op.join(bids_path, 'derivatives', 'freesurfer')

# FreeSurfer volumes
wm_path = op.join(fs_path, participant, 'mri', 'wm.mgz')

# Functional MT ROIs
func_mt_path = op.join(
    bids_path,
    'analysis',
    'ROIs',
    'func_roi',
    'functional_vol_roi',
    participant
)

# ------------------------------------------------------------
# Load FreeSurfer WM volume
# ------------------------------------------------------------
wm_img = ants.image_read(wm_path)

# ------------------------------------------------------------
# Create WM-GMI mask (approx cortical surface)
# ------------------------------------------------------------
wm_data = wm_img.numpy()

# threshold white matter
wm_mask = wm_data > 0

# convert to ANTs image
wm_mask_img = ants.new_image_like(wm_img, wm_mask.astype(float))

# ------------------------------------------------------------
# FURY actors
# ------------------------------------------------------------

# Transparent WM surface
wmgmi_actor = actor.contour_from_roi(
    wm_mask_img.numpy(),
    color=(0.85, 0.85, 0.85),
    opacity=0.2
)

# ------------------------------------------------------------
# Functional MT ROIs
# ------------------------------------------------------------
hemisphere = ['L', 'R']

func_mask_images = {}

for hemi in hemisphere:

    mask_name = f"{hemi}_mt"

    mask_img = ants.image_read(
        op.join(
            func_mt_path,
            f"{participant}_hemi-{hemi}_space-fsnative_label-MT_desc-vol_mask.nii.gz"
        )
    )

    func_mask_images[mask_name] = mask_img


# ROI actors
L_mt_actor = actor.contour_from_roi(
    func_mask_images["L_mt"].numpy(),
    color=(0, 1, 0),
    opacity=0.9
)

R_mt_actor = actor.contour_from_roi(
    func_mask_images["R_mt"].numpy(),
    color=(0, 1, 0),
    opacity=0.9
)

# ------------------------------------------------------------
# Scene
# ------------------------------------------------------------
scene = window.Scene()

# transparent brain surface
scene.add(wmgmi_actor)

# functional MT
scene.add(L_mt_actor)
scene.add(R_mt_actor)

scene.background((0, 0, 0))

scene.reset_camera_tight()

window.show(scene)

#=============================================

import ants
import nibabel as nib
import numpy as np
import os.path as op
from fury import window, actor
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import transform_streamlines

# ------------------------------------------------------------
# 1. Define paths
# ------------------------------------------------------------
participant = 'sub-EBxLxTZx1956'

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
afq_path = op.join(bids_path, 'derivatives', 'pyafq', 'wmgmi_wang')
qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant)


t1w_acpc_file = op.join(
    qsiprep_path,
    'anat',
    f"{participant}_space-ACPC_desc-brain_mask.nii.gz"
)

# ------------------------------------------------------------
# 2. Load base images
# ------------------------------------------------------------
t1w_img = nib.load(t1w_acpc_file)
t1w_ants_img = ants.image_read(t1w_acpc_file)


# ------------------------------------------------------------
# 3. Helper to create tube streamlines
# ------------------------------------------------------------
def lines_as_tubes(streamlines, line_width, color):

    line_actor = actor.line(streamlines, colors=color)
    line_actor.GetProperty().SetRenderLinesAsTubes(1)
    line_actor.GetProperty().SetLineWidth(line_width)

    return line_actor



# ------------------------------------------------------------
# 6. ROI helper
# ------------------------------------------------------------
def roi_actor(roi_path, color):

    if not op.exists(roi_path):
        print(f"⚠️ Missing ROI: {roi_path}")
        return None

    roi_img = ants.image_read(roi_path)

    roi_resampled = ants.resample_image_to_target(
        roi_img,
        t1w_ants_img,
        interp_type='nearestNeighbor'
    )

    return actor.contour_from_roi(
        roi_resampled.numpy(),
        color=color,
        opacity=0.6
    )

# ------------------------------------------------------------
# 7. ROI definitions
# ------------------------------------------------------------
roi_defs = {

    "MT": ("analysis/ROIs/wang_space-ACPC_rois", "MT_mask_dilated", (0, 0, 1)),
    # "LGNxPU": ("analysis/ROIs/julich_space-ACPC_rois", "LGNxPU_mask", (0, 0.8, 0.2)),
    # "PTxSTS1": ("analysis/ROIs/julich_space-ACPC_rois", "PTxSTS1_mask", (1, 0, 0)),
    # "FEF": ("analysis/ROIs/julich_space-ACPC_rois", "FEF_mask", (0.2, 0.6, 1)),

}

roi_actors = []

for hemi in ["L", "R"]:

    for roi_name, (subdir, label, color) in roi_defs.items():

        if roi_name == "MT":

            roi_path = op.join(
                bids_path,
                subdir,
                participant,
                f"{participant}_hemi-{hemi}_space-ACPC_label-{roi_name}_mask_dilated.nii.gz"
            )

        else:

            roi_path = op.join(
                bids_path,
                subdir,
                participant,
                'ses-concat',
                'anat',
                f"{participant}_hemi-{hemi}_space-ACPC_desc-{roi_name}_mask.nii.gz"
            )

        roi_act = roi_actor(roi_path, color)

        if roi_act:
            roi_actors.append(roi_act)

# ------------------------------------------------------------
# 8. Transparent brain surface
# ------------------------------------------------------------

t1_data = t1w_ants_img.numpy()

brain_mask = t1_data > np.percentile(t1_data, 20)

brain_actor = actor.contour_from_roi(
    brain_mask.astype(np.uint8),
    color=(0.85, 0.85, 0.85),
    opacity=0.08
)


# ------------------------------------------------------------
# 10. Build scene
# ------------------------------------------------------------

scene = window.Scene()
scene.add(brain_actor)

for act in roi_actors:
    scene.add(act)


scene.reset_camera_tight()

scene.background((0, 0, 0))

window.show(scene)