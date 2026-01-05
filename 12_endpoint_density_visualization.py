
#Diffusion endpoint density maps visualization
#Loic Daumail 09/11/2025
# -------------------
# Visualize projection 
# -------------------

#Still need to add tract to the figure

import nibabel as nib
from nibabel.freesurfer import read_geometry, read_label
from fury import window, actor, colormap
import numpy as np
import os.path as op
from scipy.ndimage import binary_erosion
import ants

# ----------------------------
# Load data
# ----------------------------
tract_name = 'wangMTxLGNxPU'
participant = 'sub-NSxGxHKx1965' #'sub-EBxGxCCx1986'
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
proj_density_path = op.join(bids_path, 'analysis', 'tdi_maps','dipy_wmgmi_tdi_maps', participant, 'wang_MT')
analysis_path = op.join(bids_path, 'analysis')
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')

hemi = 'L' if 'Left' in tract_name else 'R'
hemi_fs = "lh" if hemi == "L" else "rh"
wm_surf = op.join(fs_path, participant, 'surf', f"{hemi_fs}.inflated")    # FreeSurfer surface
coords, faces = read_geometry(wm_surf)

# Load projected density map (per vertex values)
output_file = op.join(proj_density_path, f"{participant}_hemi-{hemi}_space-fsnative_label-{tract_name}_desc-fsprojdensity0mm2.mgh")
surf_map = nib.load(output_file).get_fdata().squeeze()

# Load MT label vertices
label_file = op.join(
    analysis_path, 'ROIs', 'func_roi',
    'functional_surf_roi',
    participant,
    f"{participant}_hemi-{hemi}_space-fsnative_label-MT_mask.label"
)
label_vertices = read_label(label_file)

# ----------------------------
# Base density map overlay (all vertices) actor
# ----------------------------
# normalize surface map
norm_vals = (surf_map - np.nanmin(surf_map)) / (np.nanmax(surf_map) - np.nanmin(surf_map) + 1e-8)

# colormap for full surface
surf_colors = colormap.create_colormap(norm_vals, name='plasma')  # Nx3 RGB
surf_colors = np.c_[surf_colors, np.ones(surf_colors.shape[0])]   # add alpha

density_actor = actor.surface(coords, faces, surf_colors)

# ----------------------------
# MT boundary as line overlay actor
# ----------------------------
# Extract MT boundary edges (faces that include MT + non-MT vertices)
mask = np.zeros(coords.shape[0], dtype=bool)
mask[label_vertices] = True

# find edges between MT and non-MT vertices
edges = []
for f in faces: #Faces contains triangles of 3 corner IDs each.
    in_mask = mask[f]
    if np.any(in_mask) and not np.all(in_mask):
        # collect boundary edges
        for i in range(3):
            v1, v2 = f[i], f[(i+1) % 3] #use modulo 3 to take care of last corner i =2 with corner 0
            if mask[v1] != mask[v2]:
                edges.append((v1, v2))

edges = np.unique(np.array(edges), axis=0) #This to avoid edges counted twice over all triangles
line_coords = coords[edges]

# draw boundary as polyline actor (red)
boundary_actor = actor.line(line_coords, colors=(0, 1, 0), linewidth=3)

## ---------------------------------------------------
# ## Volumetric MT ROI actor in fs RAS-tkr space
## ---------------------------------------------------
# STEP 1: registration from acpc to fs RAS-tkr
qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant)
acpc_t1 = op.join(qsiprep_path, 'anat', participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
acpc_t1_img = ants.image_read(acpc_t1)
fs_ref_path = op.join(fs_path, participant, 'mri', 'orig.mgz')  # or 'T1.mgz'
fs_ref_img  = ants.image_read(fs_ref_path)
fs_brain_mask = op.join(fs_path, participant, 'mri', 'brain.mgz')
fs_brain_mask_img = ants.image_read(fs_brain_mask)

reg = ants.registration(
    fixed = fs_ref_img,
    moving = acpc_t1_img,
    type_of_transform = 'Rigid',#'SyN', #SyN here, as qsiprep T1 and MNI152NLin2009cAsym are different brains. For same brains, use 'Rigid'
    mask = fs_brain_mask_img,  
    reg_iterations = (1000, 500, 250, 100),  
    verbose = True
)
# ----------------------------
# STEP 2: Load volumetric ROI and resample to fs native space
# ----------------------------
roi_path = op.join(analysis_path, 'ROIs', 'func_roi', 'functional_vol_roi', participant)
mt_mask_img = ants.image_read(op.join(roi_path, participant+'_hemi-'+hemi+'_space-ACPC_label-MT_mask_dilated.nii.gz'))
mt_mask = mt_mask_img.numpy()
#Resample mask into freesurfer native space
mytx = reg['fwdtransforms']
mt_mask_fs_img = ants.apply_transforms(
    moving = mt_mask_img, 
    fixed = fs_ref_img, 
    transformlist = mytx, 
    interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
)
# -----------------------------
### --- STEP 3: generate volumetric ROI surface voxels ---
# -----------------------------
mt_mask_fs = mt_mask_fs_img.numpy()
mt_eroded_fs = binary_erosion(mt_mask_fs.astype(bool))
outer_mt_fs  =  mt_mask_fs.astype(bool) & (~mt_eroded_fs) #mt_mask_fs - mt_eroded_fs
outer_vox    = np.argwhere(outer_mt_fs)    # (N,3) in FS voxel indices (i,j,k)

# ----------------------------
# --- STEP  4: Convert FS voxel indices -> RAS-tkr
# ----------------------------
# RAS-tkr matrix for the FS reference
nib_fs_ref_img  = nib.load(fs_ref_path)
vox2ras_tkr = nib.freesurfer.mghformat.MGHHeader.get_vox2ras_tkr(nib_fs_ref_img.header)  # 4x4

ones = np.ones((outer_vox.shape[0], 1))
ijk1 = np.hstack([outer_vox[:, [0,1,2]], ones])           # i,j,k,1
ras_tkr = (vox2ras_tkr @ ijk1.T).T[:, :3]                 # (N,3) in RAS-tkr
vol_actor = actor.point(ras_tkr, colors=(1, 0, 0), point_radius=0.3)  # red dots

# ----------------------------
# Scene
# ----------------------------
scene = window.Scene()
scene.add(density_actor)
scene.add(boundary_actor)
# scene.add(vol_actor)

window.show(scene)