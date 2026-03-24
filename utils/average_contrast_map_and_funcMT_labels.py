#Compute average MotionxStationary heatmap of PT localizer in NS
#The first part of this script loads tstat contrast maps from functional activity (BOLD) of PT localizer
#It then computes an average map.
#The second part of this script computes an average func MT and creates a label file out of it
import os
import os.path as op
import numpy as np
import nibabel as nib

bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')

hemis = ["L", "R"]
func_data_dir = op.join(bids_path, 'analysis', 'fMRI_data')
participants = sorted([p for p in os.listdir(func_data_dir) if p.startswith("sub-") and "sub-EBxGxYZx1949" not in p])
# Dictionary to store per-hemisphere data
hemi_maps = {hemi: [] for hemi in hemis}
for hemi in hemis:
    print(f"\n=== Loading {hemi} hemisphere ===")
    for participant in participants:
        if "NS" in participant:
            print(f"\n=== Loading participant {participant} ===")

            mgz_file = op.join(bids_path,"analysis", "fMRI_data", participant, "glm", "contrasts",
                f"{participant}_task-ptlocal_hemi-{hemi}_space-fsaverage_desc-motionXstationary_tstat.mgz")

            # Load with nibabel (freesurfer .mgz)
            img = nib.load(mgz_file)
            data = np.squeeze(img.get_fdata())  # (n_vertices,) or (1, n_vertices)

            hemi_maps[hemi].append(data)

            print(f"{hemi} shape: {hemi_maps[hemi][0].shape}")
    # convert list → array **after** loading all participants
    hemi_maps[hemi] = np.vstack(hemi_maps[hemi])
    print(f"{hemi} shape: {hemi_maps[hemi].shape}")

# ---------------------------------------------------------
# Compute average map for each hemisphere
# ---------------------------------------------------------
avg_maps = {}

for hemi in hemis:
    if hemi_maps[hemi].size == 0:
        print(f"No maps found for {hemi}")
        continue

    avg_maps[hemi] = np.mean(hemi_maps[hemi], axis=0)
    print(f"Avg {hemi} shape: {avg_maps[hemi].shape}")

# ---------------------------------------------------------
# (Optional) Save average maps as .mgz in fsaverage space
# ---------------------------------------------------------
save_dir = op.join(bids_path, "analysis", "group_averages")
os.makedirs(save_dir, exist_ok=True)

for hemi in hemis:
    if hemi not in avg_maps:
        continue

    # reuse a "template" MGZ header so the output matches fsaverage
    template_file = op.join(bids_path, "analysis", "fMRI_data", participants[0],"glm", "contrasts",
        f"{participants[0]}_task-ptlocal_hemi-{hemi}_space-fsaverage_desc-motionXstationary_tstat.mgz"
    )

    template = nib.load(template_file)
    avg_img = nib.MGHImage(avg_maps[hemi][None, :], affine=template.affine, header=template.header)

    out_path = op.join(save_dir, f"group_hemi-{hemi}_space-fsaverage_desc-motionXstationary_avg_tstat.mgz")
    nib.save(avg_img, out_path)
    print(f"Saved: {out_path}")


#----------------------------------------------------------
#Compute average functionally defined hMT+ in NS
#----------------------------------------------------------
# +
#----------------------------------------------------------
#Compute average functionally defined hMT+ in EB
#----------------------------------------------------------

import os
import os.path as op
import nibabel.freesurfer as fs
from nibabel.freesurfer.io import write_annot
import numpy as np
import sys
import shutil

bids_path = "/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb"
utils = op.join(bids_path, 'code','utils')
sys.path.append(op.expanduser(f'{utils}'))

fs_dir = op.join(bids_path, "derivatives", "freesurfer")
roi_dir = op.join(bids_path, "analysis", "ROIs", "func_roi", "functional_surf_roi")

hemis = ["L", "R"]
participants = sorted([d for d in os.listdir(roi_dir) if d.startswith("sub-") and "sub-EBxGxYZx1949" not in d])

# fsaverage (or fsnative) has a fixed number of vertices per hemi
# For fsnative, you must load one participant's surface to get n_vertices
# FS average surface sizes:
fsaverage_surf = op.join(fs_dir, "fsaverage", "surf", "rh.white")
fsavg_coords, _ = fs.read_geometry(fsaverage_surf)
N_VERTS_FSAVG = fsavg_coords.shape[0]
print("fsaverage vertices per hemisphere:", N_VERTS_FSAVG)

average_masks = {hemi: None for hemi in hemis}

# ------------------------------
# PROCESS EACH HEMISPHERE
# ------------------------------
for hemi in hemis:
    print(f"\n=== Processing hemisphere {hemi} ===")
    hemi_masks = []
    hemi_fs = "lh" if hemi == "L" else "rh"

    for participant in participants:
        if "EB" in participant:
            # Input fsnative label

            # native_label = next(
            #     (p for p in [
            #         op.join(roi_dir, participant, f"{participant}_hemi-{hemi}_space-fsnative_label-PT_mask.label"),
            #         op.join(roi_dir, participant, f"{participant}_hemi-{hemi}_space-fsnative_label-PT_desc-NS_mask.label")
            #     ] if op.exists(p)),
            #     None
            # )

            # # Output: label transformed to fsaverage
            # out_label = op.join(
            #     roi_dir, participant,
            #     f"{participant}_hemi-{hemi}_space-fsaverage_label-PT_mask.label"
            # ) if "label-PT_mask" in native_label else op.join(
            #     roi_dir, participant,
            #     f"{participant}_hemi-{hemi}_space-fsaverage_label-PT_desc-NS_mask.label"
            # )
            native_label =  op.join(roi_dir, participant, f"{participant}_hemi-{hemi}_space-fsnative_label-MT_mask.label")
            out_label = op.join(roi_dir, participant, f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label")
            # ---------------------------------------------------
            # (1) RESAMPLE using FreeSurfer mri_label2label
            # ---------------------------------------------------
            cmd = (
                f"mri_label2label "
                f"--srclabel {native_label} "
                f"--srcsubject {participant} "
                f"--trgsubject fsaverage "
                f"--trglabel {out_label} "
                f"--hemi {hemi_fs} "
                f"--regmethod surface "
                f"--sd {fs_dir}"
            )
            print(f"Running:\n{cmd}")
            os.system(cmd)
            os.system(f'bash {utils}/callFreesurferFunction.sh -s "{cmd}"')

            # ---------------------------------------------------
            # (2) LOAD back the fsaverage label file
            # ---------------------------------------------------
            verts_fsavg = fs.read_label(out_label, read_scalars=False)

            # Create binary mask in fsaverage space
            mask = np.zeros(N_VERTS_FSAVG, dtype=np.float32)
            mask[verts_fsavg] = 1.0

            hemi_masks.append(mask)

    # Convert to array → shape (n_subjects, n_vertices)
    hemi_masks = np.stack(hemi_masks)

    # Average
    avg_mask = hemi_masks.mean(axis=0)
    average_masks[hemi] = avg_mask

    print(f"Computed average mask for {hemi}: shape {avg_mask.shape}, mean value: {avg_mask.mean():.4f}")

    print(f"{hemi} average map shape = {avg_mask.shape}, min={avg_mask.min()}, max={avg_mask.max()}")


    # Threshold at 0.5 for group-level ROI
    consensus_mask_binary = (average_masks[hemi] >= 0.1).astype(int)

    # Find vertices of consensus ROI
    roi_verts = np.where(consensus_mask_binary == 1)[0]
    save_dir = op.join(bids_path, "analysis", "ROIs", "func_roi", "functional_surf_roi", "group_averages")
    os.makedirs(save_dir, exist_ok=True)
    # txt_out = op.join(save_dir,f"{hemi_fs}_consensus_temp.txt")
    # np.savetxt(txt_out, roi_verts, fmt="%d")

    fsaverage_surf = op.join(fs_dir, "fsaverage", "surf", f"{hemi_fs}.white")
    fsavg_coords, _ = fs.read_geometry(fsaverage_surf)
    # value=1.0
    output_label = op.join(save_dir,f"group-EB_hemi-{hemi}_space-fsaverage_label-MT_mask.label")

    with open(output_label, "w") as f:
        f.write(f"#!ascii label  , from subject fsaverage vox2ras=TkReg\n")
        f.write(f"{len(roi_verts)}\n")
        for vertex in roi_verts: # for each vertex
            f.write(f"{vertex} 0 0 0 0\n")

    # shutil.copy(txt_out, final_label)
    print(f"Saved final label to {output_label}")
   