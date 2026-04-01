#The goal of this script is to plot the beta values projected on the endpoint density maps in 
#and RGB scheme 

import os
import os.path as op
import numpy as np
import nibabel as nib
import pandas as pd

from nilearn import plotting
from nibabel.freesurfer import read_label
import matplotlib.pyplot as plt
from nilearn.surface import load_surf_mesh


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

beta_dir = op.join(bids_path, 'analysis', 'diff2func_model_fits', 'participants_linearreg','combined', 'participant_betas_contrast-motionXstationary_combined_tracts.csv')
df = pd.read_csv(beta_dir)

dims = ["Participant", "Tract", "Hemisphere"]  # order matters!
value_col = "Beta"

# get unique values per dimension (sorted for consistency)
levels = [sorted(df[d].unique()) for d in dims]

# create full grid index
multi_index = pd.MultiIndex.from_product(levels, names=dims)

# align data to full grid
df_indexed = df.set_index(dims)[value_col].reindex(multi_index)

# reshape into N-D array
shape = [len(l) for l in levels]
beta_coeffs = df_indexed.to_numpy().reshape(shape)



#=============================================================
# Plot
#=============================================================
global_max = np.max(np.abs(beta_coeffs)) + 1e-8
for h, hemi in enumerate(hemis):
    n_subjects, n_tracts, n_vertices = density_data[hemi].shape
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

        # -------------------------------------------------
        # Build RGB map (3 tracts → RGB)
        # -------------------------------------------------

        rgb_map = np.zeros((n_vertices, 3), dtype=np.float32)

        for t, tract in enumerate(tract_order):
            surf_map = np.zeros(n_vertices, dtype=np.float32)

            # replace density > 0 with beta
            mask = density_data[hemi][s][t] > 0
            surf_map[mask] = beta_coeffs[s, t, h]

            # restrict to Wang ROI
            surf_map_full = np.zeros(n_vertices, dtype=np.float32)
            surf_map_full[wang_hmt_vertices] = surf_map[wang_hmt_vertices]

            # assign to RGB channel
            rgb_map[:, t] = surf_map_full
        # normalize betas → 0–1
        vmin, vmax = -1.0, 1.0
        # normalize per channel
        for t in range(3):
            max_val = np.max(rgb_map[:, t]) + 1e-8
            rgb_map[:, t] /= max_val

        rgb_map = np.clip(rgb_map, 0, 1)

        # Output filename (no run index)
        img_out_dir = op.join(bids_path, "analysis", "plots", "surface_pngs", participant)
        os.makedirs(img_out_dir, exist_ok=True)
        out_png = op.join(
            img_out_dir,
            f"{participant}_hemi-{hemi}_desc-tracts_betas_inflated.png"
        )

        # -------------------------------------------------
        # Plot only once
        # -------------------------------------------------

        # display = plotting.plot_surf(
        #     surf_mesh=infl_surf,
        #     surf_map=rgb_map,
        #     hemi="left" if hemi == "L" else "right",
        #     view="lateral",
        #     bg_map=curv_norm,
        #     bg_on_data=True,
        #     darkness=0.6,
        # )   

        # # ---- MT boundary overlay ----
        # plotting.plot_surf_contours(
        #     surf_mesh=infl_surf,
        #     roi_map=func_mt_roi,
        #     levels=[1],
        #     colors=["lightgray"],
        #     linewidths=2.0,
        #     figure=display.figure,
        #     axes=display.axes[0]
        # )

        # # ---- save + close ----
        # display.savefig(out_png, dpi=300)
        # plt.close(display.figure)


        coords, faces = load_surf_mesh(infl_surf)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # plot mesh with RGB colors
        mesh = ax.plot_trisurf(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            triangles=faces,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )

        # assign RGB colors per vertex → per face
        face_colors = rgb_map[faces].mean(axis=1)
        mesh.set_facecolors(face_colors)

        # background curvature (optional blend)
        curv_faces = curv_norm[faces].mean(axis=1)
        blended = 0.6 * face_colors + 0.4 * plt.cm.gray(curv_faces)[:, :3]
        blended = np.clip(blended, 0, 1)
        mesh.set_facecolors(blended)

        if hemi == "L":
            ax.view_init(elev=0, azim=180)   # lateral left
        else:
            ax.view_init(elev=0, azim=0)     # lateral right
        ax.set_axis_off()

        plt.savefig(out_png, dpi=300)
        plt.close(fig)



