import os
import os.path as op
import numpy as np
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt
from skimage import measure
from nibabel.processing import resample_from_to
from nilearn import plotting
import ants
import subprocess

participant = 'sub-EBxGxCCx1986'
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
analysis_path = op.join(bids_path, 'analysis')
roi_path = op.join(analysis_path, 'functional_vol_roi', participant)
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
out_path = op.join(bids_path, 'analysis', 'tdi_maps','dipy_proj_surf', participant)
os.makedirs(out_path, exist_ok=True)



# registration
# qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant)
# acpc_t1 = op.join(qsiprep_path, 'anat', participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
# acpc_t1_img = ants.image_read(acpc_t1)

# registration_file = op.join(out_path, 'ACPC2fsnative_register.dat')
# # 1) Generate registration file .dat
# cmd = ["bbregister",
#         "--s", participant, 
#         "--sd", fs_path,
#         "--mov", acpc_t1,
#         "--reg", registration_file,
#         "--init-fsl",
#         "--t1"
# ]
# print("Running:", " ".join(cmd))
# subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})
# registration
qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant)
acpc_t1 = op.join(qsiprep_path, 'anat', participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
acpc_t1_img = ants.image_read(acpc_t1)
fs_brain_mask = op.join(fs_path, participant, 'mri', 'brain.mgz')
fs_brain_mask_img = ants.image_read(fs_brain_mask)
# 1) Load the FreeSurfer conformed reference volume
fs_ref_path = op.join(fs_path, participant, 'mri', 'T1.mgz')  # or 'T1.mgz'
fs_ref_img  = ants.image_read(fs_ref_path)
# fs_ref_img.numpy().shape
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

    # -----------------------------
    ### --- STEP 2: load streamline density map
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
        interpolator = "genericLabel"
     ) # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)

   # -----------------------------
    ### --- STEP 3: generate volumetric ROI surface voxels ---
    # -----------------------------
    mt_mask_fs = mt_mask_fs_img.numpy()
    mt_eroded_fs = binary_erosion(mt_mask_fs.astype(bool))
    outer_mt_fs  =  mt_mask_fs.astype(bool) & (~mt_eroded_fs) #mt_mask_fs - mt_eroded_fs
    outer_vox_fs   = np.argwhere(outer_mt_fs)    # ROI voxel coordinates (N,3) in FS voxel indices (i,j,k)
    
    # mt_mask_fs.shape
    # outer_vox_fs.shape
    # -----------------------------
    ### --- STEP 4: Extract intensities from density map ---
    # -----------------------------
    # --- Get surface voxel intensities ---
    density = density_fs_img.numpy()
    intensities = density[
        outer_vox_fs[:, 0],
        outer_vox_fs[:, 1],
        outer_vox_fs[:, 2]
    ]
    # density.shape
    # Make an empty array with same shape as reference
    out_data = np.zeros(density_fs_img.shape, dtype=np.float32)
    # Fill only the outer voxels with your intensities
    out_data[outer_vox_fs[:, 0], outer_vox_fs[:, 1], outer_vox_fs[:, 2]] = intensities
    # out_data.shape
    out_img = ants.from_numpy(out_data, origin=density_fs_img.origin, spacing=density_fs_img.spacing, direction=density_fs_img.direction)

    # -----------------------------
    # STEP 6: Save as NIfTI (.nii.gz)
    # -----------------------------
    mt_density_path = op.join(analysis_path, "tdi_maps", "surface_voxels", participant,f"{participant}_hemi-{hemi}_label-MT_surface_density.nii.gz")
    os.makedirs(op.dirname(mt_density_path), exist_ok=True)
    ants.image_write(out_img, mt_density_path)

    # --- Plotting ---
    mt_density_map_img = ants.image_read(mt_density_path)
    intensities = mt_density_map_img.numpy()
    intensities.shape
    # # Get voxel coordinates of nonzero entries
    # coords = np.argwhere(intensities > 0)   # (N, 3) -> i, j, k voxel indices
    # values = intensities[intensities > 0]   # (N,)   -> corresponding density values

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # img = ax.scatter(
    #     coords[:, 0],
    #     coords[:, 1],
    #     coords[:, 2],
    #     c=values,
    #     cmap='Spectral',
    #     s=60
    # )
    # ax.view_init(elev=20, azim=0)
    # hemi_name = "Left" if hemi == "L" else "Right"
    # ax.set_title(f"LGNxMT Endpoint Density on {hemi_name} MT ROI Surface")
    # fig.colorbar(img, ax=ax, label='Endpoint Count')

    # plt.show()

    ## STEP 7: Perform volume to surface projection of MT ROI density values
    # Output file name
    out_file = os.path.join(out_path, f"{participant}_hemi-{hemi}_space-fsnative_label-{mask_name}_desc-fsprojdensity.mgh")

    hemi_fs  = "lh" if hemi == "L" else "rh"
    # Build mri_vol2surf command
    cmd = [
        "mri_vol2surf",
        "--mov", mt_density_path,
        "--regheader", participant,
        "--hemi", hemi_fs,
        "--surf", "white",
        "--projfrac", "-1",
        "--sd", fs_path,
        "--out", out_file
    ]

    # Run the command
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})






