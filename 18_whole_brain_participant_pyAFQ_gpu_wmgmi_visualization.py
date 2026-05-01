
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
participant = 'sub-EBxLxTZx1956' #'sub-EBxGxZAx1990'#'sub-EBxGxEYx1965' #'sub-EBxGxZAx1990' #'sub-EBxLxTZx1956'

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research',
                    'ampb_mt_tractometry_analysis', 'ampb')
afq_path = op.join(bids_path, 'derivatives', 'pyafq', 'wmgmi_wb', 'afq-Left') ##op.join('/Volumes', 'cos-lab-wpark78', 'LoicDaumail', 'ampb', 'derivatives', 'pyafq', 'wmgmi_wang') #op.join(bids_path, 'derivatives', 'pyafq', 'wmgmi_wang')#
qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant)

# Files
gmwmi_mask_file = op.join(afq_path, participant,
                          f"{participant}_ses-concat_acq-HCPdir99_desc-wmgmi_mask.nii.gz")
t1w_acpc_file = op.join(qsiprep_path, 'anat',
                        f"{participant}_space-ACPC_desc-preproc_T1w.nii.gz")

# -------------------------------------------------------------------------
# 2. Load base images
# -------------------------------------------------------------------------
t1w_img = nib.load(t1w_acpc_file)
t1w_ants_img = ants.image_read(t1w_acpc_file)
#wmgmi_img = ants.image_read(gmwmi_mask_file)


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
    #"CallosumAnteriorFrontal": (0.2, 0.6, 1),
    # "CallosumMotor": (1, 0.2, 0.2),
    # "CallosumOccipital": (1, 0.5, 0),
    # "CallosumOrbital": (0.9, 0.8, 0),
    # "CallosumPosteriorParietal": (0, 0.8, 0.2),
    # "CallosumSuperiorFrontal": (0.8, 0.2, 1),
    # "CallosumSuperiorParietal": (0.5, 0.5, 0.5),
    "CallosumTemporal": (1, 0.2, 0.2),
    # "LeftAnteriorThalamic": (1, 0.5, 0),
    # "LeftArcuate": (0.9, 0.8, 0),
    # "LeftCingulumCingulate": (0, 0.8, 0.2),
    # "LeftCorticospinal": (0.2, 0.6, 1),
    # "LeftInferiorFrontooccipital": (0.8, 0.2, 1),
    # "LeftInferiorLongitudinal": (0.7, 0.7, 0.7),
    # "LeftPosteriorArcuate": (0.2, 0.6, 1),
    # "LeftSuperiorLongitudinal": (0.2, 1, 1),
    # "LeftUncinate": (0.2, 0.8, 1),
    # "LeftVerticalOccipital": (0.2, 0, 1),
    # "RightAnteriorThalamic": (0.2, 0.6, 0),
    # "RightArcuate": (0.2, 0.6, 0.9),
    # "RightCingulumCingulate": (0.4, 0.6, 0.8),
    # "RightCorticospinal": (0.5, 0.5, 0.9),
    # "RightInferiorFrontooccipital": (0.7, 0.6, 1),
    # "RightInferiorLongitudinal": (0.9, 0.6, 0.9),
    # "RightPosteriorArcuate": (0.8, 0.8, 1),
    # "RightSuperiorLongitudinal": (0.2, 0.3, 1),
    # "RightUncinate": (0.1, 0.1, 1),
    # "RightVerticalOccipital": (0.3, 0.3, 1),

}
# -------------------------------------------------------------------------
# 5. Load and transform tracts
# -------------------------------------------------------------------------
tract_actors = []
for tract_name, color in tracts.items():
    # hemi = "Left" if tract_name.startswith("Left") else "Right"
    # mask_name = tract_name.replace("MT", "MTmask")
    # mask_name = mask_name.replace("afq-", "")
    tract_file = op.join(afq_path, participant, "bundles",
                         f"{participant}_ses-concat_acq-HCPdir99_desc-{tract_name}_tractography.trx")
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
     "MT":    ("analysis/ROIs/wang_space-ACPC_rois", "MT_mask_dilated", (1, 0, 0)),
    # "LGNxPU":    ("analysis/ROIs/julich_space-ACPC_rois", "LGNxPU_mask", (0, 0.8, 0.2)),
    # "PTxSTS1":   ("analysis/ROIs/julich_space-ACPC_rois", "PTxSTS1_mask", (1, 0, 0)),
    # "FEF":   ("analysis/ROIs/julich_space-ACPC_rois", "FEF_mask", (0.2, 0.6, 1)),
}
roi_actors = []

for hemi in ["L", "R"]: # "R"
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
for act in tract_actors + roi_actors:
    scene.add(act)
scene.add(t1_actor)
# scene.add(wmgmi_actor)
scene.reset_camera_tight()
scene.background((0, 0, 0))
window.show(scene)




