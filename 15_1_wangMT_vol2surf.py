#Resample Wang MT from volume MNI to surface fsaverage space
#Loic Daumail 
#11/12/2025
import os
import os.path as op
import ants
import subprocess

mni_wang_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases','Wang_2015')
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
hemisphere = ["L", "R"]
for h, hemi in enumerate(hemisphere):
    hemi_fs = "lh" if hemi == "L" else "rh"

    #First make the hMT+ ROI:
    mni_mt_img = ants.image_read(op.join(mni_wang_path, 'subj_vol_all', f"perc_VTPM_vol_roi13_{hemi_fs}.nii.gz"))
    mni_mst_img = ants.image_read(op.join(mni_wang_path, 'subj_vol_all', f"perc_VTPM_vol_roi12_{hemi_fs}.nii.gz"))

    # binarize
    mni_mt_img[mni_mt_img >= 1] = 1 #Peak probability for MT is about 50%
    mni_mt_img[mni_mt_img < 1] = 0

    mni_mst_img[mni_mst_img >= 1] = 1 #Peak probability for MST is about 40%
    mni_mst_img[mni_mst_img < 1] = 0


    img1_data = mni_mt_img.numpy()
    img2_data = mni_mst_img.numpy()

    union_img = (img1_data > 0) | (img2_data > 0) 
    union_img = union_img.astype(mni_mt_img.dtype)  # Keep same data type
    
    # Convert back to ANTs image
    out_path = op.join(mni_wang_path, 'hmtplus')
    os.makedirs(out_path, exist_ok=True)
    wang_mask_img = ants.from_numpy(union_img, origin=mni_mt_img.origin, spacing=mni_mt_img.spacing, direction=mni_mt_img.direction)
    # # save transformed mask
    vol_hMT_file = op.join(out_path, f"hemi-{hemi}_space-mni_label-hMT_desc-wangvol.nii.gz")
    ants.image_write(wang_mask_img, vol_hMT_file)


    out_fsaverage_file = op.join(out_path, f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang.mgh")

    hemi_fs  = "lh" if hemi == "L" else "rh"
    # Build mri_vol2surf command
    cmd = [
        "mri_vol2surf",
        "--mov", vol_hMT_file,
        "--regheader", 'fsaverage',
        "--hemi", hemi_fs,
        "--surf", "white",
        "--projdist", "0",
        "--sd", fs_path,
        "--out", out_fsaverage_file
    ] #"--projfrac", "-0.3",  "--regheader", participant

    # Run the command
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})

    # # Resample to fsaverage space
    # source_mt_file = out_mni_file
    # out_fsaverage_file = op.join(out_path, f"hemi-{hemi_fs}_space-fsaverage_label-MT_desc-wang.mgh")

    # cmd = ["mri_surf2surf",
    # "--srcsubject", participant, 
    # "--trgsubject", "fsaverage",
    # "--hemi", hemi_fs, 
    # "--sval", source_density_file, 
    # "--tval", out_fsaverage_file ]

    # # Run the command
    # print("Running:", " ".join(cmd))
    # subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})
