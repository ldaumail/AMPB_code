import os
import os.path as op
import nibabel as nib
from nibabel.freesurfer.io import read_label
import numpy as np
import subprocess


participant = 'sub-EBxGxCCx1986'
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
analysis_path = op.join(bids_path, 'analysis')
label_file = op.join(analysis_path, 'functional_surf_roi', participant, participant+"_hemi-L_space-fsnative_label-MT_mask.label")  # FreeSurfer label file for MT
density_map = op.join(analysis_path, 'tdi_maps', 'dipy_tdi_maps', participant, participant+'_ses-04_desc-STS1xMTL_tdi_map.nii.gz')
out_dir = op.join(analysis_path, 'tdi_maps', 'dipy_proj_surf')
# Ensure output dir exists
os.makedirs(out_dir, exist_ok=True)

hemisphere = ['lh', 'rh']
hemi = 'lh'

# Output file name
out_file = os.path.join(out_dir, f"{participant}_{hemi}_tdi_on_surf.mgh")

# Build mri_vol2surf command
cmd = [
    "mri_vol2surf",
    "--mov", density_map,
    "--regheader", participant,
    "--hemi", hemi,
    "--surf", "white",
    "--projfrac", "0",
    "--out", out_file
]

# Run the command
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})

# Load projected surface map
surf_map = nib.load(out_file).get_fdata().squeeze()

# Load label vertices
label_vertices = read_label(label_file)

# Extract values inside ROI
mt_values = surf_map[label_vertices]

#------- visualize data ----------- #
import nibabel as nib
from nibabel.freesurfer import read_geometry
from fury import window, actor, colormap
import numpy as np

fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
wm_surf = op.join(fs_path, participant, 'surf', "lh.white")   # FreeSurfer surface (white or pial)

# Load surface (coords = vertices, faces = triangles)
coords, faces = read_geometry(wm_surf)

# Load projected surface data
# surf_map = nib.load(out_file).get_fdata().squeeze()

# Normalize values for colormap
# values = surf_map.copy()
values = (mt_values - np.nanmin(mt_values)) / (np.nanmax(mt_values) - np.nanmin(mt_values))

# Map values to colors (e.g., plasma colormap)
colors = colormap.create_colormap(values, name='plasma')

# Create mesh actor
mesh_actor = actor.surface(coords, faces, colors)

# Create interactive window
scene = window.Scene()
scene.add(mesh_actor)

# Show
window.show(scene)