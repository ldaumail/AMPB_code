import os
import os.path as op
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from skimage import measure
import nibabel.freesurfer.io as fsio
from sklearn.neighbors import NearestNeighbors


participant = 'sub-EBxGxCCx1986'
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
analysis_path = op.join(bids_path, 'analysis')
roi_path = op.join(analysis_path, 'functional_vol_roi', participant)
fs_path = op.join(bids_path, 'derivatives', 'freesurfer', participant)

hemisphere = ['L', 'R']
roi = ['LeftMTxWMxLGN', 'RightMTxWMxLGN']
for h, hemi in enumerate(hemisphere):

    mt_roi_img = nib.load(op.join(roi_path, participant+'_hemi-'+hemi+'_space-ACPC_label-MT_mask_dilated.nii.gz'))
    mt_dat = mt_roi_img.get_fdata()
    affine = mt_roi_img.affine


    ### --- Step 1: obtain surface ROI coordinates -- ###
    # --- generate volumetric ROI surface voxels ---
    mt_eroded = binary_erosion(mt_dat) #If no structuring element is provided, an element is generated with a square connectivity equal to one. Connectivity = 1 keeps the erosion conservative, shrinking only along face-adjacent voxels
    outer_mt = mt_dat - mt_eroded  # Surface voxels are 1; others are 0

# --- Check the ROIs created ---
# plot 
# plotting.plot_roi(mt_roi, bg_img=t1w_img, alpha=0.5, cmap="autumn", title="MT")
# plt.show()

# plotting.plot_roi(mt_eroded, bg_img=fa_img, alpha=0.5, cmap="autumn", title="MT")
# plt.show()

#mt_roi_surface = nib.Nifti1Image(surface_mt.astype(np.float32), affine=mt_roi.affine)
# plotting.plot_roi(mt_roi_surface, bg_img=fa_img, alpha=0.5, cmap="autumn", title="MT")
# plt.show()

    ### --- Step 2: Load streamline density map
    mask = roi[h] 
    density_path = op.join(analysis_path, 'tdi_maps', 'dipy_tdi_maps', participant, f"{participant}_ses-concat_desc-{mask}_tdi_map.nii.gz")
    density_map_img = nib.load(density_path)
    density_map = density_map_img.get_fdata()
    # print(density_map.space)
    # --- Get surface voxel coordinates ---
    outer_voxel_mtcoords = np.argwhere(outer_mt > 0)

# --- Extract intensities from density map ---
    intensities = density_map[
        outer_voxel_mtcoords[:, 0],
        outer_voxel_mtcoords[:, 1],
        outer_voxel_mtcoords[:, 2]
    ]

    # --- Plotting ---
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # img = ax.scatter(
    #     surface_voxel_mtcoords[:, 0],
    #     surface_voxel_mtcoords[:, 1],
    #     surface_voxel_mtcoords[:, 2],
    #     c=intensities,
    #     cmap='Spectral',
    #     s=60
    # )

    # ax.view_init(elev=20, azim=180)
    # hemi_name = "Left" if hemi == "L" else "Right"
    # ax.set_title(f"LGNxMT Endpoint Density on {hemi_name} MT ROI Surface")
    # fig.colorbar(img, ax=ax, label='Endpoint Count')

    # plt.show()



    # ----------------------------
    # USER INPUTS
    # ----------------------------

    # ----------------------------
    # STEP 1: Load volumetric ROI
    # ----------------------------

    # Generate surface mesh from ROI volume
    verts, faces, normals, values = measure.marching_cubes(mt_dat, level=0.5)

    # Convert voxel coordinates → world coordinates
    verts_world = nib.affines.apply_affine(affine, verts)

    # ----------------------------
    # STEP 2: Load FreeSurfer WM surface
    # ----------------------------
    hemi  = "lh" if hemi == "L" else "rh"
    surf_path = os.path.join(fs_path, "surf", f"{hemi}.inflated")
    fs_coords, fs_faces = fsio.read_geometry(surf_path)

    # ----------------------------
    # STEP 3: Project ROI surface onto WM surface
    # ----------------------------
    nn = NearestNeighbors(n_neighbors=1).fit(fs_coords)
    distances, indices = nn.kneighbors(verts_world)

    roi_indices = indices[:, 0]  # nearest FS vertex for each ROI vertex

    # ----------------------------
    # STEP 4: Assign values to surface vertices
    # ----------------------------
    surf_values = np.zeros(len(fs_coords))  # one value per vertex
    for idx in roi_indices:
        surf_values[idx] = 1  # or use intensities if you have a map

# ----------------------------
# STEP 5: Save as FreeSurfer overlay (.mgh)
# ----------------------------
mgh_img = nib.freesurfer.mghformat.MGHImage(
    surf_values[:, np.newaxis, np.newaxis, np.newaxis],  # shape (N,1,1,1)
    np.eye(4)  # dummy affine
)
nib.save(mgh_img, output_file)

print(f"✅ Saved projected ROI surface to {output_file}")
print("👉 Load this in Freeview with:")
print(f"freeview -f {surf_path}:overlay={output_file}")




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
