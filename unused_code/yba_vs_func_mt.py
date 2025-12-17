
#This script goal is to assess the overlap between YBA hMT+ and functionally defined MT
#Loic Daumail 12/11/2025

import os.path as op
import ants
from fury import window, actor
import numpy as np
import scipy.ndimage as ndi


# -----------------------------------------------
# 1: Look at dilated YBA hMT+ mask + func MT with the wmgmi
# -----------------------------------------------

participant = 'sub-EBxLxQPx1957' #'sub-NSxLxYKx1964' #'sub-EBxGxZAx1990' #'sub-NSxLxATx1954' 
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'anat')
acpc_t1_path       = op.join(qsiprep_path, participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
acpc_t1_img       = ants.image_read(acpc_t1_path)

yba_lh_mt_path = op.join(bids_path, 'analysis', 'ROIs', 'yba_space-ACPC_rois',participant, f"{participant}_hemi-L_space-ACPC_desc-MT_mask_dilated.nii.gz")
yba_lh_mt_img       = ants.image_read(yba_lh_mt_path)
yba_rh_mt_path = op.join(bids_path, 'analysis', 'ROIs', 'yba_space-ACPC_rois',participant, f"{participant}_hemi-R_space-ACPC_desc-MT_mask_dilated.nii.gz")
yba_rh_mt_img       = ants.image_read(yba_rh_mt_path)

func_lh_mt_path = op.join(bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_vol_roi', participant, f"{participant}_hemi-L_space-ACPC_label-MT_mask.nii.gz")
func_lh_mt_img = ants.image_read(func_lh_mt_path)
func_rh_mt_path = op.join(bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_vol_roi', participant, f"{participant}_hemi-R_space-ACPC_label-MT_mask.nii.gz")
func_rh_mt_img = ants.image_read(func_rh_mt_path)

# -----------
## WMGMI
# -----------

# wmgmi_path =  op.join(bids_path, 'derivatives', 'pyAFQ', 'wmgmi_wang', 'afq-LeftMTxFEF', participant, participant+'_ses-concat_acq-HCPdir99_desc-wmgmi_mask.nii.gz' )
# op.exists(wmgmi_path)
# # --- Load your ROI (WMGMI file) ---

# wmgmi_img = ants.image_read(wmgmi_path)
# wmgmi_resampled = ants.resample_image_to_target(
#     image=wmgmi_img, 
#     target=acpc_t1_img, # Use the T1w as the spatial target
#     interp_type='nearestNeighbor' # Crucial for binary masks
# )

# # Smooth contour for GMWMI (binary mask)
# wmgmi_actor = actor.contour_from_roi(
#     wmgmi_resampled.numpy(),
#     color=(1, 0, 0),   # red surface
#     opacity=0.2
# )

# --------------------------------------------------------
# RESTRICT WMGMI TO WANG MT MASK (per hemisphere)
# --------------------------------------------------------

# --------------------------------------------------------
# EXPAND (DILATE) WANG MT MASKS BY 1–2 VOXELS
# --------------------------------------------------------

# Convert ANTs images -> numpy arrays
# wmgmi_arr = wmgmi_resampled.numpy().astype(bool)

# wang_lh_arr = ants.resample_image_to_target(
#     yba_lh_mt_img, acpc_t1_img, interp_type="nearestNeighbor"
# ).numpy().astype(bool)

# wang_rh_arr = ants.resample_image_to_target(
#     yba_rh_mt_img, acpc_t1_img, interp_type="nearestNeighbor"
# ).numpy().astype(bool)

# Dilation (choose iterations=1 or 2)
# dilate_by = 0  # <-- change to 1 or 2 depending on how much spill you want
# structure = ndi.generate_binary_structure(3, 1)

# wang_lh_dil = ndi.binary_dilation(wang_lh_arr, structure=structure, iterations=dilate_by)
# wang_rh_dil = ndi.binary_dilation(wang_rh_arr, structure=structure, iterations=dilate_by)

# Intersection with WMGMI mask
# wmgmi_lh_masked = np.logical_and(wmgmi_arr, wang_lh_dil)
# wmgmi_rh_masked = np.logical_and(wmgmi_arr, wang_rh_dil)

# Convert to FURY surfaces
# wmgmi_lh_actor = actor.contour_from_roi(
#     wmgmi_lh_masked.astype(np.uint8),
#     color=(1, 0.3, 0),
#     opacity=0.5
# )

# wmgmi_rh_actor = actor.contour_from_roi(
#     wmgmi_rh_masked.astype(np.uint8),
#     color=(1, 0.3, 0),
#     opacity=0.5
# )

# ----------------------------
# Convert ANTs masks -> FURY volume actors
# ----------------------------
# Background anatomical T1 actor
t1_actor = actor.slicer(acpc_t1_img.numpy())

# ROI actors
yba_lh_mt_actor = actor.contour_from_roi(
    yba_lh_mt_img.numpy(), color=(0, 0, 1), opacity=0.5
)
yba_rh_mt_actor = actor.contour_from_roi(
    yba_rh_mt_img.numpy(), color=(0, 0, 1), opacity=0.5
)

func_lh_mt_actor = actor.contour_from_roi(
    func_lh_mt_img.numpy(), color=(0, 1, 0), opacity=1
)
func_rh_mt_actor = actor.contour_from_roi(
    func_rh_mt_img.numpy(), color=(0, 1, 0), opacity=1
)

scene = window.Scene()
scene.add(t1_actor)       # anatomical background
scene.add(yba_lh_mt_actor)    # left MT ROI
scene.add(yba_rh_mt_actor)    # right MT ROI
scene.add(func_lh_mt_actor)    # left MT ROI
scene.add(func_rh_mt_actor)    # right MT ROI
# scene.add(wmgmi_lh_actor)
# scene.add(wmgmi_rh_actor)
window.show(scene)



#---------------------------------------------------------------------------------
## Check %overlap between YBA MT and func MT
#---------------------------------------------------------------------------------
import os.path as op
import ants
import numpy as np
import pandas as pd
#participant = 'sub-NSxLxYKx1964' #'sub-NSxLxYKx1964' #'sub-EBxGxZAx1990' #'sub-NSxLxATx1954' 
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
participants_file = op.join(bids_path, 'code', 'utils', 'study2_subjects_updated.txt')
with open(participants_file, "r") as f:
    participants = [line.strip() for line in f if line.strip()] 
all_rows = []
N = len(participants)   # or whatever loop length you're using
overlap  = [0] * N
for i, participant in enumerate(participants):
    qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'anat')

    for hemi in ['L', 'R']:
        yba_mt_path = op.join(bids_path, 'analysis', 'ROIs', 'yba_space-ACPC_rois', participant, f"{participant}_hemi-{hemi}_space-ACPC_desc-MT_mask_dilated.nii.gz")
        yba_mt_img       = ants.image_read(yba_mt_path)

        func_mt_path = op.join(bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_vol_roi', participant, f"{participant}_hemi-{hemi}_space-ACPC_label-MT_mask.nii.gz")
        func_mt_img = ants.image_read(func_mt_path)

        overlap[i] = np.sum(yba_mt_img.numpy()*func_mt_img.numpy())/np.sum(func_mt_img.numpy())

        all_rows.append({
                    "participant": participant,
                    "hemisphere": hemi,
                    "proportion": overlap[i],
                    "group": ("EB" if participant.startswith("sub-EB") else "NS" if participant.startswith("sub-NS") else "Other")
                    })

df_results = pd.DataFrame(all_rows)

#Perform stats
from scipy.stats import mannwhitneyu

# Load your dataframe (example)
# df = pd.read_csv("your_file.csv")

results = {}

for hemi in ["L", "R"]:
    df_hemi = df_results[df_results["hemisphere"] == hemi]

    eb = df_hemi[df_hemi["group"] == "EB"]["proportion"]
    ns = df_hemi[df_hemi["group"] == "NS"]["proportion"]

    stat, p = mannwhitneyu(eb, ns, alternative='two-sided')

    results[hemi] = {
        "EB_n": len(eb),
        "NS_n": len(ns),
        "U_statistic": stat,
        "p_value": p
    }

print(results)

df_stats = pd.DataFrame(results)


#------------------------------------------------------------------------
# Look at overlap of wang MT with V1 and hIP
#---------------------------------------------------------------------------------
import os.path as op
import ants
import numpy as np
#participant = 'sub-NSxLxYKx1964' #'sub-NSxLxYKx1964' #'sub-EBxGxZAx1990' #'sub-NSxLxATx1954' 
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
participants_file = op.join(bids_path, 'code', 'utils', 'study2_subjects_updated.txt')
with open(participants_file, "r") as f:
    participants = [line.strip() for line in f if line.strip()] 
N = len(participants)   # or whatever loop length you're using

left_overlap  = [0] * N
right_overlap = [0] * N
for i, participant in enumerate(participants):
    qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'anat')

    wang_lh_mt_path = op.join(bids_path, 'analysis', 'wang_space-ACPC_rois',participant, f"{participant}_hemi-L_space-ACPC_desc-MT_mask_dilated.nii.gz")
    wang_lh_mt_img       = ants.image_read(wang_lh_mt_path)
    wang_rh_mt_path = op.join(bids_path, 'analysis', 'wang_space-ACPC_rois',participant, f"{participant}_hemi-R_space-ACPC_desc-MT_mask_dilated.nii.gz")
    wang_rh_mt_img       = ants.image_read(wang_rh_mt_path)

    func_lh_mt_path = op.join(bids_path, 'analysis', 'julich_space-ACPC_rois', participant, 'ses-concat', 'anat', f"{participant}_hemi-L_space-ACPC_desc-hIP_mask.nii.gz")
    func_lh_mt_img = ants.image_read(func_lh_mt_path)
    func_rh_mt_path = op.join(bids_path, 'analysis', 'julich_space-ACPC_rois', participant, 'ses-concat', 'anat', f"{participant}_hemi-R_space-ACPC_desc-hIP_mask.nii.gz")
    func_rh_mt_img = ants.image_read(func_rh_mt_path)


    left_overlap[i] = np.sum(wang_lh_mt_img.numpy()*func_lh_mt_img.numpy())/np.sum(func_lh_mt_img.numpy())
    right_overlap[i] = np.sum(wang_rh_mt_img.numpy()*func_rh_mt_img.numpy())/np.sum(func_rh_mt_img.numpy())

