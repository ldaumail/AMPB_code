import os
import os.path as op
import nibabel as nib
from nibabel.freesurfer.io import read_label
from nibabel.freesurfer import read_geometry
import numpy as np
from scipy.ndimage import map_coordinates


participant = 'sub-EBxGxCCx1986'
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
analysis_path = op.join(bids_path, 'analysis')
label_file = op.join(analysis_path, 'functional_surf_roi', participant, participant+"_hemi-L_space-fsnative_label-MT_mask.label")  # FreeSurfer label file for MT

fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
surf_file = op.join(fs_path, participant, 'surf', "lh.white")   # FreeSurfer surface (white or pial)

density_map = op.join(analysis_path, 'tdi_maps', 'dipy_tdi_maps', participant, participant+'_ses-04_desc-STS1xMTL_tdi_map.nii.gz')

# ---- 3. Load ROI label (FreeSurfer .label gives list of vertex indices) ----
label_vertices = read_label(label_file)

# ---- 4. Project volumetric density onto surface ----
# Simplest way: sample density map onto surface vertices
# Requires surface coordinates in voxel space
coords, faces = read_geometry(surf_file)

# Transform surface coords (RAS) -> voxel indices
# Load density map ----
img = nib.load(density_map)
data = img.get_fdata()

affine = np.linalg.inv(img.affine)
vox_coords = nib.affines.apply_affine(affine, coords)

# Sample density map at vertex locations (nearest neighbor)

samples = map_coordinates(data, vox_coords.T, order=0)

# ---- 5. Restrict to MT ROI ----
mt_values = samples[label_vertices]

print("Mean streamline endpoint density in MT ROI:", np.mean(mt_values))
print("Distribution (first 10 values):", mt_values[:10])