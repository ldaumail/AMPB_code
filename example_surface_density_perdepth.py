#Plot surface density map of multiple projection depths.
import os.path as op
import os
import numpy as np
import nibabel as nib

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
density_dir = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps')
func_dir = op.join(bids_path, 'analysis', 'fMRI_data')
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')

participants_file = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb', 'code', 'utils', 'subjects.txt')
with open(participants_file, "r") as f:
    participants = [line.strip().lstrip("/") for line in f if line.strip()]

# -----------------------
#1 Generate endpoint densities array
#-------------------------
# ✅ Fixed tract order (keep consistent across subjects!)
tract_order = ['InferiorLongitudinal'] 

hemis = ["L", "R"]
projdist = '10'
# hemis = ["L"]
# tract_order = ['PTR']
# participants = ['sub-EBxLxHHx1949']
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
        subj_dir = op.join(density_dir, participant, 'pyAFQ33_wb_red')
        subj_densities = []

        # Loop through *tracts in fixed order*
        for tract in tract_order:
            
            # Find file matching this tract and hemisphere
            matches = [f for f in os.listdir(subj_dir) if f"{tract}" in f and f"hemi-{hemi_fs}" in f and "fsaverage" in f and f.endswith(f"fsprojdensity{projdist}mm2.mgh")]

            if not matches:
                print(f"   ⚠️ Missing: {tract} ({hemi}) for {participant}")
                # subj_densities.append(np.zeros_like(subj_densities[0]) if subj_densities else None)
                subj_densities.append(np.zeros([163842, 1, 1])) #if there is no match, replace by map of zeros
                continue

            # Load the file
            file_path = op.join(subj_dir, matches[0])
            img = nib.load(file_path)
            data = img.get_fdata().astype(np.float32)
            subj_densities.append(data)

        # Stack into one array: shape (n_tracts, n_vertices)
        subj_densities = np.stack(subj_densities, axis=0)  # (n_tracts, n_vertices)
        density_data[hemi].append(subj_densities)

# for i, arr in enumerate(density_data[hemi]):
#     print(f"{hemi} element {i}: shape = {arr.shape}")
    
# Convert to numpy arrays

for hemi in hemis:
    density_data[hemi] = np.squeeze(np.stack(density_data[hemi], axis=0))  # (n_subjects, n_tracts, n_vertices)
    print(f"✅ {hemi}-hemisphere shape: {density_data[hemi].shape}")

#================== Prepare and Normalize X data =============
hemis = ["L", "R"]
n_subj = len(participants)
n_tracts = len(tract_order)
norm_density_data = {hemi: {} for hemi in hemis}
norm_contrast_data = {hemi: {} for hemi in hemis}
for h, hemi in enumerate(hemis):
    densities = density_data[hemi]        # X; (subj, tract, vertices)

    # ----------------------------
    # Load MT ROI
    # ----------------------------
    wang_hmt_path = op.join(
        '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
        f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
    )
    surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
    wang_hmt_vertices = np.where(surf_roi > 0)[0]
    print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")
    # Densities within Wang MT only (subj, tract, masked_vertices)
    if n_tracts == 1:
        densities_masked = densities[wang_hmt_vertices]
    else:
        densities_masked = densities[:, wang_hmt_vertices]

    n_masked = len(wang_hmt_vertices)

    #Prepare subject's X = zscored density maps for each tract
    zscored_densities = np.full((n_tracts, n_masked), np.nan)
    for tract_idx in range(n_tracts):

        print(f" Tract {tract_idx+1}/{n_tracts} z-scored")

        # anatomical vector for this subject and tract (length = n_masked)
        # anat_vec = densities_masked[s, tract_idx, :]  # shape (n_masked,)
        #Zscore the density data of the tract of this participant
        if n_tracts == 1:
            if np.std(densities_masked[ :]) == 0:
                zscored_densities[tract_idx,:] = 0
            else:
                zscored_densities[tract_idx,:] = (densities_masked[ :] - np.mean(densities_masked[ :]))/np.std(densities_masked[ :])
        else:
            if np.std(densities_masked[tract_idx, :]) == 0:
                zscored_densities[tract_idx,:] = 0
            else:
                zscored_densities[tract_idx,:] = (densities_masked[tract_idx, :] - np.mean(densities_masked[tract_idx, :]))/np.std(densities_masked[tract_idx, :])


        norm_density_data[hemi].setdefault(0, {})
        norm_density_data[hemi][0] = zscored_densities



#-------------------------------------
## Plot density map
#-------------------------------------
from nilearn import plotting
from nibabel.freesurfer import read_label
import matplotlib.pyplot as plt
from pathlib import Path
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
empty_check = {}
for h, hemi in enumerate(hemis):
    hemi_fs = "lh" if hemi == "L" else "rh"
    infl_surf = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.inflated")
    n_vertices  = density_data[hemi].shape 
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

    os.makedirs(op.join(bids_path, "analysis", "diff2func_model_fits", "pyAFQ33_wb_participants_linearreg", f"surface_pngs_{projdist}mm"), exist_ok=True)

    for s, participant in enumerate(participants):
        # s =1
        # participant= 'sub-EBxGxEYx1965'
        # # ----------------------------
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
       
        img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "pyAFQ33_wb_participants_linearreg", f"surface_pngs_{projdist}mm", participant)
        os.makedirs(img_out_dir, exist_ok=True)

        vmin, vmax = 0, 5.0

        for t, tract in enumerate(tract_order):

            if not np.all(norm_density_data[hemi][s][t] == 0):
                key = (s, t, h)
                empty_check[key] = empty_check.get(key, 0) + 1
                # Build full-surface vector 
                surf_map = np.full(n_vertices, np.nan, dtype=np.float32)
                surf_map[wang_hmt_vertices] = norm_density_data[hemi][s][t]#predicted_maps[hemi][s, :]

                # Output filename (no run index)
                out_png = Path(op.join(img_out_dir,f"{participant}_hemi-{hemi}_desc-{tract}_inflated_scaled.png"
                ))
                if not out_png.exists():
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
            # elif np.all(norm_density_data[hemi][s][t] == 0):
                # key = (s, t, h)
                # empty_check[key] = empty_check.get(key, 0) + 1
