import os
import os.path as op
import nibabel as nib
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

# 1) Load the FreeSurfer conformed reference volume
fs_ref_path = op.join(fs_path, participant, 'mri', 'orig.mgz')  # or 'T1.mgz'
fs_ref_img  = ants.image_read(fs_ref_path)

# registration
qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant)
acpc_t1 = op.join(qsiprep_path, 'anat', participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
acpc_t1_img = ants.image_read(acpc_t1)

registration_file = op.join(out_path, 'ACPC2fsnative_register.dat')
# 1) Generate registration file .dat
cmd = ["bbregister",
        "--s", participant, 
        "--sd", fs_path,
        "--mov", acpc_t1,
        "--reg", registration_file,
        "--init-fsl",
        "--t1"
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})

hemisphere = ['L', 'R']
roi_names = ['LeftMTxWMxLGN', 'RightMTxWMxLGN']
for h, hemi in enumerate(hemisphere):
    # ----------------------------
    # STEP 1: Load volumetric ROI
    # ----------------------------
    mt_mask_img = ants.image_read(op.join(roi_path, participant+'_hemi-'+hemi+'_space-ACPC_label-MT_mask_dilated.nii.gz'))
    mt_mask = mt_mask_img.numpy()

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
    ### --- STEP 2: source streamline density map
    # -----------------------------
    mask_name = roi_names[h] 
    density_path = op.join(analysis_path, 'tdi_maps', 'dipy_tdi_maps', participant, f"{participant}_ses-concat_desc-{mask_name}_tdi_map.nii.gz")
    density_map_img = ants.image_read(density_path)


   # -----------------------------
    ### --- STEP 4: generate volumetric ROI surface voxels ---
    # -----------------------------
    mt_eroded = binary_erosion(mt_mask.astype(bool))
    outer_mt  =  mt_mask.astype(bool) & (~mt_eroded) #mt_mask_fs - mt_eroded_fs
    outer_vox    = np.argwhere(outer_mt)    # (N,3) in FS voxel indices (i,j,k)
    
    # -----------------------------
    ### --- STEP 5: Extract intensities from density map ---
    # -----------------------------
    # --- Get surface voxel coordinates ---
    density = density_map_img.numpy()
    intensities = density[
        outer_vox[:, 0],
        outer_vox[:, 1],
        outer_vox[:, 2]
    ]
    
    # Make an empty array with same shape as reference
    out_data = np.zeros(density_map_img.shape, dtype=np.float32)

    # Fill only the outer voxels with your intensities
    out_data[outer_vox[:, 0], outer_vox[:, 1], outer_vox[:, 2]] = intensities

    out_img = ants.from_numpy(out_data, origin=density_map_img.origin, spacing=density_map_img.spacing, direction=density_map_img.direction)

    # -----------------------------
    # STEP 6: Save as NIfTI (.nii.gz)
    # -----------------------------
    mt_density_path = op.join(
        analysis_path, "tdi_maps", "surface_voxels", participant,
        f"{participant}_hemi-{hemi}_label-MT_surface_density.nii.gz"
    )
    os.makedirs(op.dirname(mt_density_path), exist_ok=True)
    ants.image_write(out_img, mt_density_path)

    # Output file name
    out_file = os.path.join(out_path, f"{participant}_hemi-{hemi}_space-fsnative_label-{mask_name}_desc-fsprojdensity.mgh")

    hemi_fs  = "lh" if hemi == "L" else "rh"
    # Build mri_vol2surf command
    cmd = [
        "mri_vol2surf",
        "--mov", mt_density_path,
        "--regheader", participant,
        "--hemi", hemi_fs,
        "--reg", registration_file,
        "--surf", "white",
        "--projfrac", "-1",
        "--sd", fs_path,
        "--out", out_file
    ]

    # Run the command
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})






