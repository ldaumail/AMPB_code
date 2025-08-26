import os
import os.path as op
import nibabel as nib
from nibabel.freesurfer.io import read_label
import numpy as np
import subprocess


participant = 'sub-NSxGxBAx1970'
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
analysis_path = op.join(bids_path, 'analysis')
fs_path = op.join(bids_path, 'derivatives','freesurfer')
out_dir = op.join(analysis_path, 'tdi_maps', 'dipy_proj_surf', participant)
# Ensure output dir exists
os.makedirs(out_dir, exist_ok=True)

# template_file = op.join(bids_path, 'derivatives', 'pyAFQ', 'cleaning_rounds2', 'afq-LeftMTxLGN', participant, f"{participant}_ses-concat_acq-HCPdir99_b0ref.nii.gz")
template_file = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'anat',f"{participant}_space-ACPC_desc-preproc_T1w.nii.gz")
# template_img = nib.load(template_file)
# print(template_img.affine)

# fs_t1 = op.join(bids_path, 'derivatives', 'freesurfer', participant, 'mri', 'T1.mgz')
# fs_t1_img = nib.load(fs_t1)
# print(fs_t1_img.affine)

registration_file = op.join(out_dir, 'ACPC2fsnative_register.dat')
# 1) Generate registration file .dat
cmd = ["bbregister",
        "--s", participant, 
        "--sd", fs_path,
        "--mov", template_file,
        "--reg", registration_file,
        "--init-fsl",
        "--t1"
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})

#2) Apply ACPC2fsNative registration and project density map to surface 
hemisphere = ['L', 'R']
roi = ['LeftMTxWMxLGN', 'RightMTxWMxLGN']
for h, hemi in enumerate(hemisphere):
    mask = roi[h] 
    density_map = op.join(analysis_path, 'tdi_maps', 'dipy_tdi_maps', participant, f"{participant}_ses-concat_desc-{mask}_tdi_map.nii.gz")
    # density_img = nib.load(density_map)
    # print(density_img.affine)

    # Output file name
    out_file = os.path.join(out_dir, f"{participant}_hemi-{hemi}_space-fsnative_label-{mask}_tdi_on_surf.mgh")

    if hemi == 'L':
        hm = 'lh'
    else:
        hm = 'rh'
    # Build mri_vol2surf command
    cmd = [
        "mri_vol2surf",
        "--mov", density_map,
        "--regheader", participant,
        "--hemi", hm,
        "--reg", registration_file,
        "--surf", "white",
        "--projfrac", "0",
        "--sd", fs_path,
        "--out", out_file
    ]

    # Run the command
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})


#------- visualize complete hemisphere data ----------- #
import nibabel as nib
from nibabel.freesurfer import read_geometry
from fury import window, actor, colormap
import numpy as np

fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
wm_surf = op.join(fs_path, participant, 'surf', "lh.inflated")   # FreeSurfer surface (white or pial)

# Load surface (coords = vertices, faces = triangles)
coords, faces = read_geometry(wm_surf)

# Load projected surface data
surf_map = nib.load(out_file).get_fdata().squeeze()

# Normalize values for colormap
values = surf_map.copy()

values = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))

# Map values to colors (e.g., plasma colormap)
colors = colormap.create_colormap(values, name='plasma')

# Create mesh actor
mesh_actor = actor.surface(coords, faces, colors)

# Create interactive window
scene = window.Scene()
scene.add(mesh_actor)

# Show
window.show(scene)

# ----- Visualize MT ROI data

import nibabel as nib
from nibabel.freesurfer import read_geometry, read_label
from fury import window, actor, colormap
import numpy as np
import os.path as op

fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
wm_surf = op.join(fs_path, participant, 'surf', "lh.inflated")   # FreeSurfer surface (inflated)
coords, faces = read_geometry(wm_surf)

# Load projected surface data
surf_map = nib.load(out_file).get_fdata().squeeze()

# Load label vertices (ROI indices)
label_file = op.join(analysis_path, 'functional_surf_roi', participant, participant+"_hemi-L_space-fsnative_label-MT_mask.label")  # FreeSurfer label file for MT
label_vertices = read_label(label_file)

# Extract values inside ROI
mt_values = surf_map[label_vertices]

# ----------------------------
# Base surface (gray, inflated)
# ----------------------------
base_colors = np.ones((coords.shape[0], 4)) * 0.8  # light gray RGBA
base_colors[:, -1] = 0.3   # alpha (transparency)
base_actor = actor.surface(coords, faces, base_colors)

# ----------------------------
# ROI overlay (colored by values)
# ----------------------------
# assign colors only to ROI vertices
roi_colors = np.zeros((coords.shape[0], 4))   # RGBA
roi_colors[:, -1] = 0.0  # fully transparent background

# normalize ROI values
roi_values = surf_map[label_vertices]
norm_vals = (roi_values - np.nanmin(roi_values)) / (np.nanmax(roi_values) - np.nanmin(roi_values) + 1e-8)

# plasma colormap (returns Nx3 RGB)
roi_colormap = colormap.create_colormap(norm_vals, name='plasma')

# add alpha channel (opaque ROI)
roi_colormap_rgba = np.c_[roi_colormap, np.ones(roi_colormap.shape[0])]

# assign only to ROI vertices
roi_colors[label_vertices, :] = roi_colormap_rgba


roi_actor = actor.surface(coords, faces, roi_colors)

# ----------------------------
# Scene
# ----------------------------
scene = window.Scene()
scene.add(base_actor)
scene.add(roi_actor)

window.show(scene)