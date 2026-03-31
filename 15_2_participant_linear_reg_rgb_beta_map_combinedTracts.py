#The goal of this script is to plot the beta values projected on the endpoint density maps in 
#and RGB scheme 

import os
import os.path as op
import numpy as np
import nibabel as nib

from nilearn import plotting
from nibabel.freesurfer import read_label
import matplotlib.pyplot as plt

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
density_dir = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps')

fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
# -----------------------
#1 Generate endpoint densities array
#-------------------------

# ✅ Fixed tract order (keep consistent across subjects!)
tract_order = ['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF'] #'MTxLGNxPU', 'MTxPTxSTS1', 
participants = sorted([p for p in os.listdir(density_dir) if p.startswith("sub-")])
hemis = ["L", "R"]

# Initialize storage dictionary
density_data = {hemi: [] for hemi in hemis}

for participant in participants:
    if not participant.startswith("sub-"):
        continue
    print(f"\n🔹 Participant: {participant}")
    # -----------------
    # Loop by hemisphere
    # -----------------
    for hemi in hemis:
        # for tract in ['MTmaskxLGN', 'MTmaskxPT', 'MTmaskxSTS1', 'MTmaskxPU', 'MTmaskxFEF', 'MTmaskxhIP', 'MTmaskxV1']:
        print(f"   🧩 Hemisphere: {hemi}")
        hemi_fs = "lh" if hemi == "L" else "rh"
        subj_dir = op.join(density_dir, participant, 'wang_MT')
        subj_densities = []

        # Loop through *tracts in fixed order*
        for tract in tract_order:
            
            # Find file matching this tract and hemisphere
            matches = [f for f in os.listdir(subj_dir) if f"wang{tract}" in f and f"hemi-{hemi_fs}" in f and "fsaverage" in f and f.endswith("fsprojdensity0mm2.mgh")]

            if not matches:
                print(f"   ⚠️ Missing: {tract} ({hemi}) for {participant}")
                subj_densities.append(np.zeros_like(subj_densities[0]) if subj_densities else None)
                continue

            # Load the file
            file_path = op.join(subj_dir, matches[0])
            img = nib.load(file_path)
            data = img.get_fdata().astype(np.float32)
            subj_densities.append(data)

        # Stack into one array: shape (n_tracts, n_vertices)
        subj_densities = np.stack(subj_densities, axis=0)  # (7, n_vertices)
        density_data[hemi].append(subj_densities)

# for i, arr in enumerate(density_data[hemi]):
#     print(f"{hemi} element {i}: shape = {arr.shape}")
    
# Convert to numpy arrays

for hemi in hemis:
    density_data[hemi] = np.squeeze(np.stack(density_data[hemi], axis=0))  # (n_subjects, n_tracts, n_vertices)
    print(f"✅ {hemi}-hemisphere shape: {density_data[hemi].shape}")

#========================
#Load linreg beta coefs
#========================
beta_dir = op.join(bids_path, 'analysis', 'diff2func_model_fits', 'linearcv_group_loso_predicted_maps', 'combined', 'betas_contrast-motionXstationary_combined_tracts.csv')

#=============================================================
# Plot
#=============================================================

for h, hemi in enumerate(hemis):
    hemi_fs = "lh" if hemi == "L" else "rh"
    infl_surf = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.inflated")
    # ----------------------------
    # Load curvature map (sulci/gyri)
    # ----------------------------
    curv_file = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv")
    curv = nib.freesurfer.read_morph_data(curv_file)

    # normalize curvature for nicer background display
    curv_norm = (curv - np.percentile(curv, 5)) / (
        np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8
    )
    curv_norm = np.clip(curv_norm, 0, 1)

    # ----------------------------
    # Load Wang MT ROI
    # ----------------------------
    wang_hmt_path = op.join(
        '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
        f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
    )
    surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
    wang_hmt_vertices = np.where(surf_roi > 0)[0]

    for s, participant in enumerate(participants):
        # ----------------------------
        # Functional MT ROI (binary surface map)
        # ----------------------------
        label_file = op.join(bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi', participant,
            f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label")

        func_mt_vertices = read_label(label_file)

        func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
        func_mt_roi[func_mt_vertices] = 1

        # ----------------------------
        # Density Map visualization
        # ----------------------------
        vmin, vmax = -1.0, 1.0
        # -------------------------------------------------
        # Tract endpoint Density map
        # -------------------------------------------------
        for t, tract in enumerate(tract_order):
            # Build full-surface vector 
            surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
            surf_map[:] = density_data[hemi][s][t][:,] #wang_hmt_vertices

            # Output filename (no run index)
            img_out_dir = op.join(bids_path, "analysis", "plots", "surface_pngs", participant)
            os.makedirs(img_out_dir, exist_ok=True)
            out_png = op.join(
                img_out_dir,
                f"{participant}_hemi-{hemi}_desc-{tract}_density_inflated.png"
            )

            # -------------------------------------------------
            # Plot only once
            # -------------------------------------------------

            display = plotting.plot_surf_stat_map(
                surf_mesh=infl_surf,
                stat_map=surf_map,
                hemi="left" if hemi == "L" else "right",
                view="lateral",
                cmap="plasma",
                colorbar=True,
                vmin=vmin,
                vmax=vmax,
                threshold=None,
                bg_map=curv_norm,
                bg_on_data=True,
                darkness=0.6,
            )

            # ---- MT boundary overlay ----
            plotting.plot_surf_contours(
                surf_mesh=infl_surf,
                roi_map=func_mt_roi,
                levels=[1],
                colors=["lightgray"],
                linewidths=2.0,
                figure=display.figure,
                axes=display.axes[0]
            )

            # ---- save + close ----
            display.savefig(out_png, dpi=300)
            plt.close(display.figure)