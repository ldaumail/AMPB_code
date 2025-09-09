import os
import os.path as op
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from skimage import measure
from nibabel.processing import resample_from_to
import nibabel.freesurfer.io as fsio
from sklearn.neighbors import NearestNeighbors
from nilearn import plotting
import ants


participant = 'sub-EBxGxCCx1986'
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
analysis_path = op.join(bids_path, 'analysis')
roi_path = op.join(analysis_path, 'functional_vol_roi', participant)
fs_path = op.join(bids_path, 'derivatives', 'freesurfer', participant)
out_path = op.join(bids_path, 'analysis', 'tdi_maps','dipy_proj_surf', participant)
os.makedirs(out_path, exist_ok=True)

# 1) Load the FreeSurfer conformed reference volume
fs_ref_path = op.join(fs_path, 'mri', 'orig.mgz')  # or 'T1.mgz'
fs_ref_img  = ants.image_read(fs_ref_path)

# RAS-tkr matrix for the FS reference
nib_fs_ref_img  = nib.load(fs_ref_path)
vox2ras_tkr = nib.freesurfer.mghformat.MGHHeader.get_vox2ras_tkr(nib_fs_ref_img.header)  # 4x4

# registration
qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant)
acpc_t1 = op.join(qsiprep_path, 'anat', participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
acpc_t1_img = ants.image_read(acpc_t1)
fs_brain_mask = op.join(fs_path, 'mri', 'brain.mgz')
fs_brain_mask_img = ants.image_read(fs_brain_mask)

reg = ants.registration(
    fixed = fs_ref_img,
    moving = acpc_t1_img,
    type_of_transform = 'Rigid',#'SyN', #SyN here, as qsiprep T1 and MNI152NLin2009cAsym are different brains. For same brains, use 'Rigid'
    mask = fs_brain_mask_img,  
    reg_iterations = (1000, 500, 250, 100),  
    verbose = True
)

hemisphere = ['L', 'R']
roi_names = ['LeftMTxWMxLGN', 'RightMTxWMxLGN']
for h, hemi in enumerate(hemisphere):
    # ----------------------------
    # STEP 1: Load volumetric ROI
    # ----------------------------
    mt_mask_img = ants.image_read(op.join(roi_path, participant+'_hemi-'+hemi+'_space-ACPC_label-MT_mask_dilated.nii.gz'))
    mt_mask = mt_mask_img.numpy()

    # mt_mask_nib = nib.load(op.join(roi_path, participant+'_hemi-'+hemi+'_space-ACPC_label-MTxWM_mask.nii.gz'))
    # print(mt_mask_nib.header)
# mt_mask.shape
# non_zero_count = np.count_nonzero(mt_mask)
# (2000000/353)**(1/3)

# --- Check the ROIs created ---
# plot 
# plotting.plot_roi(mt_roi, bg_img=t1w_img, alpha=0.5, cmap="autumn", title="MT")
# plt.show()
# plotting.plot_roi(mt_eroded, bg_img=fa_img, alpha=0.5, cmap="autumn", title="MT")
# plt.show()

#mt_roi_surface = nib.Nifti1Image(surface_mt.astype(np.float32), affine=mt_roi.affine)
# plotting.plot_roi(mt_roi_surface, bg_img=fa_img, alpha=0.5, cmap="autumn", title="MT")
# plt.show()
    # -----------------------------
    ### --- STEP 2: Load streamline density map
    # -----------------------------
    mask_name = roi_names[h] 
    density_path = op.join(analysis_path, 'tdi_maps', 'dipy_tdi_maps', participant, f"{participant}_ses-concat_desc-{mask_name}_tdi_map.nii.gz")
    density_map_img = ants.image_read(density_path)

    # -----------------------------
    ### --- STEP 3: Resample mask and density into FS reference grid
    # -----------------------------

    # apply transformation: ACPC → FS space
    mytx = reg['fwdtransforms']
    density_fs_img = ants.apply_transforms(
        moving = density_map_img, 
        fixed = fs_ref_img, 
        transformlist = mytx, 
        interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
    )

    mt_mask_fs_img = ants.apply_transforms(
        moving = mt_mask_img, 
        fixed = fs_ref_img, 
        transformlist = mytx, 
        interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
    )

    ants.plot(fs_ref_img, overlay=density_fs_img, cmap="autumn", alpha=0.5, axis=2)
    # ants.plot(fs_ref_img, overlay=density_map_img, cmap="autumn", alpha=0.5, axis=2)

    # plotting.plot_roi(density_fs_img, bg_img=fs_ref_img, alpha=0.5, cmap="autumn", title="Density")
    # plt.show()

    # plotting.plot_roi(mt_mask_fs_img, bg_img=fs_ref_img, alpha=0.5, cmap="autumn", title="Density")
    # plt.show()

    # -----------------------------
    ### --- STEP 4: generate volumetric ROI surface voxels ---
    # -----------------------------
    mt_mask_fs = mt_mask_fs_img.numpy()
    mt_eroded_fs = binary_erosion(mt_mask_fs.astype(bool))
    outer_mt_fs  =  mt_mask_fs.astype(bool) & (~mt_eroded_fs) #mt_mask_fs - mt_eroded_fs
    outer_vox    = np.argwhere(outer_mt_fs)    # (N,3) in FS voxel indices (i,j,k)
    
    if outer_vox.size == 0:
        print(f'[{hemi}] No surface voxels found after resampling; skipping.')
        continue
    # -----------------------------
    ### --- STEP 4: Extract intensities from density map ---
    # -----------------------------
    # --- Get surface voxel coordinates ---
    density_fs = density_fs_img.numpy()
    intensities = density_fs[
        outer_vox[:, 0],
        outer_vox[:, 1],
        outer_vox[:, 2]
    ]
    # ----------------------------
    # --- STEP  5: Convert FS voxel indices -> RAS-tkr
    # ----------------------------
    ones = np.ones((outer_vox.shape[0], 1))
    ijk1 = np.hstack([outer_vox[:, [0,1,2]], ones])           # i,j,k,1
    ras_tkr = (vox2ras_tkr @ ijk1.T).T[:, :3]                 # (N,3) in RAS-tkr

    # --- Plotting ---
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # img = ax.scatter(
    #     outer_vox[:, 0],
    #     outer_vox[:, 1],
    #     outer_vox[:, 2],
    #     c=intensities,
    #     cmap='Spectral',
    #     s=60
    # )

    # ax.view_init(elev=20, azim=0)
    # hemi_name = "Left" if hemi == "L" else "Right"
    # ax.set_title(f"LGNxMT Endpoint Density on {hemi_name} MT ROI Surface")
    # fig.colorbar(img, ax=ax, label='Endpoint Count')

    # plt.show()

    # ----------------------------
    # STEP 6: Load FreeSurfer WM surface (coords already in RAS-tkr)
    # ----------------------------
    hemi_fs = "lh" if hemi == "L" else "rh"
    surf_path = os.path.join(fs_path, "surf", f"{hemi_fs}.white")
    fs_coords, fs_faces = fsio.read_geometry(surf_path)

    # ----------------------------
    # STEP 6: Project ROI surface onto WM surface
    # ----------------------------
    nn = NearestNeighbors(n_neighbors=1).fit(fs_coords)
    distances, indices = nn.kneighbors(ras_tkr)
    roi_indices = indices[:, 0]  # nearest FS vertex for each ROI vertex

    # ----------------------------
    # STEP 7: Assign values to surface vertices
    # ----------------------------
    # Initialize surface values
    surf_values = np.zeros(len(fs_coords), dtype=np.float32)
   
    # initialize to NaN to detect untouched vertices (optional)
    surf_values[:] = 0.0
    # If multiple voxels map to the same vertex, we’ll take the max intensity
    for fs_idx in np.unique(roi_indices):
        mapped_vals = intensities[roi_indices == fs_idx] #Find intensity values for the ROI indices that are nearest to the cortical surface
        surf_values[fs_idx] = mapped_vals.max() #Only keep the max value to build the projected surface density map
    # surf_values.size

    # ----------------------------
    # STEP 8: Save as FreeSurfer overlay (.mgh)
    # ----------------------------
    mgh_img = nib.freesurfer.mghformat.MGHImage(
        surf_values[:, np.newaxis, np.newaxis],  # shape (N,1,1,1)
        np.eye(4)  # dummy affine
    )

    output_file = op.join(out_path, f"{participant}_hemi-{hemi_fs}_label-MT_desc-projdensity.mgh")
    nib.save(mgh_img, output_file)


    print(f"✅ Saved projected ROI surface to {output_file}")
    print("👉 Load this in Freeview with:")
    print(f"freeview -f {surf_path}:overlay={output_file}")

# -------------------
# Visualize projection 
# -------------------

import nibabel as nib
from nibabel.freesurfer import read_geometry, read_label
from fury import window, actor, colormap
import numpy as np
import os.path as op

fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
wm_surf = op.join(fs_path, participant, 'surf', f"{hemi_fs}.white")    # FreeSurfer surface (inflated)
coords, faces = read_geometry(wm_surf)

# Load projected surface data
surf_map = nib.load(output_file).get_fdata().squeeze()
# print(surf_map.size)

# Load label vertices (ROI indices)
label_file = op.join(analysis_path, 'functional_surf_roi', participant, f"{participant}_hemi-{hemi}_space-fsnative_label-MT_mask.label")  # FreeSurfer label file for MT
label_vertices = read_label(label_file)
# label_vertices.size

# ----------------------------
# Base surface (gray, inflated)
# ----------------------------
base_colors = np.ones((coords.shape[0], 4)) * 0.8  # light gray RGBA
base_colors[:, -1] = 0.3   # alpha (transparency)
base_actor = actor.surface(coords, faces, base_colors)

# ----------------------------
# MT ROI overlay (colored by values of density map)
# ----------------------------
# normalize ROI values
roi_values = surf_map[label_vertices]
norm_vals = (roi_values - np.nanmin(roi_values)) / (np.nanmax(roi_values) - np.nanmin(roi_values) + 1e-8)

# color map
roi_colormap = colormap.create_colormap(norm_vals, name='plasma') # plasma colormap (returns Nx3 RGB)
roi_colormap_rgba = np.c_[roi_colormap, np.ones(roi_colormap.shape[0])] # add alpha channel (opaque ROI)

# assign colors only to ROI vertices
roi_colors = np.zeros((coords.shape[0], 4))   # RGBA
roi_colors[:, -1] = 0.0  # fully transparent background
roi_colors[label_vertices, :] = roi_colormap_rgba

#Define surface actors
roi_actor = actor.surface(coords, faces, roi_colors)

#Define Volumetric MT ROI actor in fs RAS-tkr space
vol_actor = actor.point(ras_tkr, colors=(1, 0, 0), point_radius=0.3)  # red dots


# ----------------------------
# Scene
# ----------------------------
scene = window.Scene()
scene.add(base_actor)
scene.add(roi_actor)
scene.add(vol_actor)

window.show(scene)

# -------------------------------------------- #

#Check coordinates and space for each file
def center_world(img):
    center_voxel = np.array(img.shape) // 2
    center_world = img.affine @ np.append(center_voxel, 1)
    return center_world[:3]
mt_roi = nib.load(op.join(roi_path, participant+'_hemi-'+hemi+'_space-ACPC_label-MT_mask_dilated.nii.gz'))
mt_roi.affine
print("Center world coords image 1:", center_world(mt_roi))
mt_roi =  nib.load(op.join(bids_path, 'derivatives', 'pyAFQ','cleaning_rounds2', 'afq-RightMTxLGN', participant, 'ROIs', participant+'_ses-concat_acq-HCPdir99_space-ACPC_desc-RightMTxWMxLGNStart_mask.nii.gz'))
mt_roi.affine
print("Center world coords image 2:", center_world(mt_roi))

julich_path = op.join('/Users','ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb', 'analysis', 'julich_space-ACPC_rois')
lgn_roi = nib.load(op.join(julich_path, participant, 'ses-concat', 'anat', participant+'_hemi-R_space-T1w_label-LGN_mask.nii.gz'))
lgn_roi.affine
print("Center world coords image 3:", center_world(lgn_roi))

fa_path = op.join(base_path,'models')
fa_img = nib.load(op.join(fa_path, participant+'_ses-04_acq-HCPdir99_model-dki_param-fa_dwimap.nii.gz'))
fa = fa_img.get_fdata()
print("Center world coords image 3:", center_world(fa_img))
print(fa_img.header)
t1w_img.affine
print(t1w_img.header)
print(sts1_mtR.space)




########### Visualize Bundles #############
from fury import actor, window
from fury.colormap import create_colormap

import AFQ.data.fetch as afd
from AFQ.viz.utils import PanelFigure
from dipy.tracking.streamline import transform_streamlines

qsiprep_path = op.join('/Volumes','cos-lab-wpark78','LoicDaumail','ampb','derivatives', 'qsiprep', participant, 'anat')
# read in the bundles, trasnform into RASMM coordinates and then into anatomical coordinates
t1w_img = nib.load(op.join(qsiprep_path,
                           participant+'_space-ACPC_desc-preproc_T1w.nii.gz'))
t1w = t1w_img.get_fdata()
sts1_mtL.to_rasmm()
sts1_mtR.to_rasmm()
sts1_mtL_t1w = transform_streamlines(sts1_mtL.streamlines,
                                np.linalg.inv(t1w_img.affine))
sts1_mtR_t1w = transform_streamlines(sts1_mtR.streamlines,
                                np.linalg.inv(t1w_img.affine))

#Visualizing bundles with principal direction coloring

def lines_as_tubes(sl, line_width, **kwargs):
    line_actor = actor.line(sl, **kwargs)
    line_actor.GetProperty().SetRenderLinesAsTubes(1)
    line_actor.GetProperty().SetLineWidth(line_width)
    return line_actor


sts1_mtL_actor = lines_as_tubes(sts1_mtL_t1w, 8)
sts1_mtR_actor = lines_as_tubes(sts1_mtR_t1w, 8)

#Slice anatomical image using slicer actors

def slice_volume(data, x=None, y=None, z=None):
    slicer_actors = []
    slicer_actor_z = actor.slicer(data)
    if z is not None:
        slicer_actor_z.display_extent(
            0, data.shape[0] - 1,
            0, data.shape[1] - 1,
            z, z)
        slicer_actors.append(slicer_actor_z)
    if y is not None:
        slicer_actor_y = slicer_actor_z.copy()
        slicer_actor_y.display_extent(
            0, data.shape[0] - 1,
            y, y,
            0, data.shape[2] - 1)
        slicer_actors.append(slicer_actor_y)
    if x is not None:
        slicer_actor_x = slicer_actor_z.copy()
        slicer_actor_x.display_extent(
            x, x,
            0, data.shape[1] - 1,
            0, data.shape[2] - 1)
        slicer_actors.append(slicer_actor_x)

    return slicer_actors


slicers = slice_volume(t1w, x=t1w.shape[0] // 2, z=t1w.shape[-1] // 3)

#Add actors to 3D window scen object
scene = window.Scene()

#scene.add(sts1_mtL_actor)
scene.add(sts1_mtR_actor)

#mtL_actor = actor.contour_from_roi(mtL_dat, color=(1, 0, 0), opacity=0.5)
mtR_actor = actor.contour_from_roi(mtR_dat, color=(1, 0, 0), opacity=0.5)

# Show in interactive window
#scene.add(mtL_actor)
scene.add(mtR_actor)


for slicer in slicers:
    scene.add(slicer)

#Now visualize:
window.show(scene, size=(1200, 1200), reset_camera=False)
