
##----
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
participant = 'sub-NSxGxHNx1952' #'sub-EBxLxTZx1956'

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research',
                    'ampb_mt_tractometry_analysis', 'ampb')
afq_path = op.join('/Volumes', 'cos-lab-wpark78', 'LoicDaumail', 'ampb', 'derivatives', 'pyafq', 'wmgmi')
qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant)

# Files
gmwmi_mask_file = op.join(afq_path, 'LeftMTxLGN', participant,
                          f"{participant}_ses-concat_acq-HCPdir99_desc-wmgmi_mask.nii.gz")
t1w_acpc_file = op.join(qsiprep_path, 'anat',
                        f"{participant}_space-ACPC_desc-preproc_T1w.nii.gz")

# -------------------------------------------------------------------------
# 2. Load base images
# -------------------------------------------------------------------------
t1w_img = nib.load(t1w_acpc_file)
t1w_ants_img = ants.image_read(t1w_acpc_file)
gmwmi_img = ants.image_read(gmwmi_mask_file)

# -------------------------------------------------------------------------
# 3. Helper to make Fury line actors
# -------------------------------------------------------------------------
def lines_as_tubes(streamlines, line_width, color):
    line_actor = actor.line(streamlines, colors=color)
    line_actor.GetProperty().SetRenderLinesAsTubes(1)
    line_actor.GetProperty().SetLineWidth(line_width)
    return line_actor

# -------------------------------------------------------------------------
# 4. Define tracts and colors
# -------------------------------------------------------------------------
tracts = {
    "LeftMTxLGN": (1, 0.2, 0.2),
    "LeftMTxPT": (1, 0.5, 0),
    "LeftMTxSTS1": (0.9, 0.8, 0),
    "LeftMTxPU": (0, 0.8, 0.2),
    "LeftMTxFEF": (0.2, 0.6, 1),
    "LeftMTxV1": (0.8, 0.2, 1),
    "LeftMTxhIP": (0.5, 0.5, 0.5),
    "RightMTxLGN": (1, 0.2, 0.2),
    "RightMTxPT": (1, 0.5, 0),
    "RightMTxSTS1": (0.9, 0.8, 0),
    "RightMTxPU": (0, 0.8, 0.2),
    "RightMTxFEF": (0.2, 0.6, 1),
    "RightMTxV1": (0.8, 0.2, 1),
    "RightMTxHIP": (0.5, 0.5, 0.5)
}

# -------------------------------------------------------------------------
# 5. Load and transform tracts
# -------------------------------------------------------------------------
tract_actors = []
for tract_name, color in tracts.items():
    hemi = "Left" if tract_name.startswith("Left") else "Right"
    mask_name = tract_name.replace("MT", "MTmask")
    tract_file = op.join(afq_path, tract_name, participant, "bundles",
                         f"{participant}_ses-concat_acq-HCPdir99_desc-{mask_name}_tractography.trx")
    if not op.exists(tract_file):
        print(f"⚠️ Missing: {tract_file}")
        continue

    trk = load_tractogram(tract_file, t1w_img)
    trk.to_rasmm()
    trk_xfm = transform_streamlines(trk.streamlines, np.linalg.inv(t1w_img.affine))
    tract_actor = lines_as_tubes(trk_xfm, 5, color=color)
    tract_actors.append(tract_actor)

# -------------------------------------------------------------------------
# 6. ROI loading helper
# -------------------------------------------------------------------------
def roi_actor(roi_path, color):
    if not op.exists(roi_path):
        print(f"⚠️ Missing ROI: {roi_path}")
        return None
    roi_img = ants.image_read(roi_path)
    roi_resampled = ants.resample_image_to_target(roi_img, t1w_ants_img, interp_type='nearestNeighbor')
    return actor.contour_from_roi(roi_resampled.numpy(), color=color, opacity=0.4)

# -------------------------------------------------------------------------
# 7. ROI definitions
# -------------------------------------------------------------------------
roi_defs = {
    "LGN":   ("analysis/julich_space-ACPC_rois", "LGN_mask", (1, 0, 0)),
    "PT":    ("analysis/julich_space-ACPC_rois", "PT_mask", (1, 0.5, 0)),
    "STS1":  ("analysis/julich_space-ACPC_rois", "STS1_mask", (0.9, 0.8, 0)),
    "PU":    ("analysis/julich_space-ACPC_rois", "PU_mask", (0, 0.8, 0.2)),
    "FEF":   ("analysis/julich_space-ACPC_rois", "FEF_mask", (0.2, 0.6, 1)),
    "V1":    ("analysis/julich_space-ACPC_rois", "V1_mask", (0.8, 0.2, 1)),
    "hIP":   ("analysis/julich_space-ACPC_rois", "hIP_mask", (0.5, 0.5, 0.5))
} #    "MT":    ("analysis/functional_vol_roi", "MT_mask_dilated", (0, 0, 1)),

roi_actors = []

for hemi in ["L", "R"]:
    for roi_name, (subdir, label, color) in roi_defs.items():
        if roi_name == "MT":
            roi_path = op.join(bids_path, subdir, participant,
                               f"{participant}_hemi-{hemi}_space-ACPC_label-{roi_name}_mask_dilated.nii.gz")
        else:
            roi_path = op.join(bids_path, subdir, participant, 'ses-concat', 'anat',
                               f"{participant}_hemi-{hemi}_space-ACPC_desc-{roi_name}_mask.nii.gz")
        roi_act = roi_actor(roi_path, color)
        if roi_act:
            roi_actors.append(roi_act)



t1_actor = actor.slicer(t1w_ants_img.numpy())
# -------------------------------------------------------------------------
# 8. Build and render scene
# -------------------------------------------------------------------------
scene = window.Scene()
for act in tract_actors + roi_actors:
    scene.add(act)
#scene.add(t1_actor)
scene.reset_camera_tight()
scene.background((0, 0, 0))
window.show(scene)