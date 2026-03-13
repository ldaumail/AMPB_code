#Here, same approach as 15_3_participant_linear_reg_combined_tracts_nested.py
#In addition, we add a permutation analysis of pearson's r 

import os
import os.path as op
import pandas as pd
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# ----------------------------
# Inputs preparation
# ----------------------------

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
density_dir = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps')
func_dir = op.join(bids_path, 'analysis', 'fMRI_data')
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

#-------------------------
# Generate Beta contrasts array
#-------------------------

contrast_order = ["motionXstationary"]
contrast_data = {hemi: [] for hemi in hemis}

for participant in participants:
    if not participant.startswith("sub-"):
        continue

    print(f"\n🔹 Participant: {participant}")
    contrasts_dir = op.join(func_dir, participant, 'glm', 'contrasts')

    for hemi in hemis:
        print(f"   🧩 Hemisphere: {hemi}")

        # Dictionary for this subject: {contrast → list of runs}
        subj_contrast_runs = {contrast: [] for contrast in contrast_order}

        for run in range(0, 6):
            print(f"  🧩 Run: {run+1}")

            for contrast in contrast_order:
                print(f"      🧩 Contrast: {contrast}")

                # Build required filename parts
                if "EB" in participant:
                    required = ["ptlocal", f"hemi-{hemi}", f"run-{run+1}", "fsaverage", contrast, "tstat"]
                else:
                    required = ["mtlocal", f"hemi-{hemi}", f"run-{run+1}", "fsaverage", contrast, "tstat"]

                # Match file
                matches = [f for f in os.listdir(contrasts_dir) if all(k in f for k in required)]

                if not matches:
                    print("        ⚠️ No matching contrast file")
                    continue

                # Load run file
                contrast_file = op.join(contrasts_dir, matches[0])
                img = nib.load(contrast_file)
                data = img.get_fdata().astype(np.float32)

                # Add run for this contrast
                subj_contrast_runs[contrast].append(data)

        # After all runs → convert each contrast’s list into an array
        subj_final = {}
        for contrast in contrast_order:
            runs = subj_contrast_runs[contrast]
            if len(runs) == 0:
                continue
            subj_final[contrast] = np.stack(runs, axis=0)  
            # shape = (n_runs, n_vertices)

        # Save this subject's data for this hemisphere
        contrast_data[hemi].append(subj_final)
#contrast_data[hemi][0][contrast][0].shape #dim 1 = hemisphere initial, dim 2 = participant number, dim 3 = contrast type, dim 4: run number of each contrast map

#================== Prepare and Normalize X and Y data for model fit =============
hemis = ["L", "R"]
contrast = contrast_order[0]   # e.g. "motionXstationary"
n_subj = len(participants)
n_tracts = len(tract_order)
norm_density_data = {hemi: {} for hemi in hemis}
norm_contrast_data = {hemi: {} for hemi in hemis}
for h, hemi in enumerate(hemis):
    densities = density_data[hemi]        # X; (subj, tract, vertices)
    subj_contrasts = contrast_data[hemi]  # Y; list: one dict per participant
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
        densities_masked = densities[:, wang_hmt_vertices]
    else:
        densities_masked = densities[:, :, wang_hmt_vertices]

    n_masked = len(wang_hmt_vertices)

    for s, participant in enumerate(participants):
        #Prepare subject's Y = functional maps
        # get this subject's run maps of the chosen contrast
        subj_dict = subj_contrasts[s]
        if contrast not in subj_dict:
            print(f"Subject {s} missing contrast {contrast} for hemi {hemi}, skipping")
            continue
        
        C_full = subj_dict[contrast]              # (n_runs, n_vertices_fullspace)
 
        # mask ROI
        C = np.squeeze(C_full[:, wang_hmt_vertices] )         # (n_runs, n_masked)
        #zscore the runs for a given participant
        n_runs = C.shape[0]
        print(f"\nSubject {s+1}: {n_runs} runs (hemi {hemi})")
        zscored_C = np.array([(C[r_num,:] - np.mean(C[r_num,:]))/np.std(C[r_num,:]) for r_num in range(n_runs)])

        norm_contrast_data[hemi].setdefault(s, {})
        norm_contrast_data[hemi][s][contrast] = zscored_C
        
        #Prepare subject's X = zscored density maps for each tract
        zscored_densities = np.full((n_tracts, n_masked), np.nan)
        for tract_idx in range(n_tracts):

            print(f" Tract {tract_idx+1}/{n_tracts} z-scored")

            # anatomical vector for this subject and tract (length = n_masked)
            # anat_vec = densities_masked[s, tract_idx, :]  # shape (n_masked,)
            #Zscore the density data of the tract of this participant
            if n_tracts == 1:
                if np.std(densities_masked[s, :]) == 0:
                    zscored_densities[tract_idx,:] = 0
                else:
                    zscored_densities[tract_idx,:] = (densities_masked[s, :] - np.mean(densities_masked[s, :]))/np.std(densities_masked[s, :])
            else:
                if np.std(densities_masked[s, tract_idx, :]) == 0:
                    zscored_densities[tract_idx,:] = 0
                else:
                    zscored_densities[tract_idx,:] = (densities_masked[s, tract_idx, :] - np.mean(densities_masked[s, tract_idx, :]))/np.std(densities_masked[s, tract_idx, :])


        norm_density_data[hemi].setdefault(s, {})
        norm_density_data[hemi][s] = zscored_densities

#================Useful functions ============
#Define a goodness of fit function
def r2_score(y_t, y_p):
    ss_res = np.sum((y_t - y_p)**2)
    ss_tot = np.sum((y_t - np.mean(y_t))**2)
    return 1 - ss_res / ss_tot


def vertex_bootstrap_reliability(C, n_boot=1000, frac=1):
    """
    C: array (n_runs, n_vertices)
    n_boot: number of bootstrap samples
    frac: fraction of vertices sampled per bootstrap
    Returns mean bootstrap reliability
    """
    n_runs, n_vertices = C.shape

    n_sample = int(frac * n_vertices)
    rs = []

    for _ in range(n_boot):
        verts = np.random.choice(n_vertices, n_sample, replace=True)

        # split runs as before (fixed run split)
        half = n_runs // 2
        A = C[:half, verts].mean(axis=0)
        B = C[half:, verts].mean(axis=0)

        r, _ = pearsonr(A, B)
        if not np.isnan(r):
            rs.append(r)

    return np.mean(rs)


def noise_normalized_r(y_true, y_pred, reliability):
    """
    y_true: (n_vertices,)
    y_pred: (n_vertices,)
    reliability: split-half reliability of y_true

    Returns noise-normalized r
    """
    if reliability <= 0 or np.isnan(reliability):
        return np.nan

    r, _ = pearsonr(y_true, y_pred)
    return r / np.sqrt(reliability)


def null_pearson_r2(y_true, y_pred):

    y_perm = np.random.permutation(y_pred)

    r, _ = pearsonr(y_perm, y_true)
    r2 = r2_score(y_true, y_perm)

    return r, r2

## Fit Regression Models to each participant
hemis = ["L", "R"]
contrast = contrast_order[0]   # e.g. "motionXstationary"
groups = ["EB", "NS"]
# get n_subj, n_tracts
n_bootstrap = 1000
n_subj = len(participants)
n_tracts = len(tract_order)
rs   = np.full((n_subj, len(hemis)), np.nan)
rsquared = np.full(( n_subj, len(hemis)), np.nan) #goodness of fit
reliability = np.full((n_subj, len(hemis)), np.nan)
r_all = np.full((n_subj, len(hemis)), np.nan)
r_rand = np.full((n_bootstrap, n_subj, len(hemis)), np.nan)
r2_rand = np.full((n_bootstrap, n_subj, len(hemis)), np.nan)
rnd_run_idx = np.full((n_subj, 3, len(hemis)), np.nan)
trained_coefs = np.zeros((n_tracts, n_subj, len(hemis)))  # scalar summary per tract/run
delta_mse = np.full((n_tracts, n_subj, len(hemis)), np.nan) #
predicted_maps = {hemi: [] for hemi in hemis}
for h, hemi in enumerate(hemis):
        #h = 0
        #hemi = "L"
        #s = 0
        #'sub-EBxGxCCx1986'

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
    n_masked = len(wang_hmt_vertices)

    # Predicted and coef storage preallocation
    predicted = np.full((n_subj, n_masked), np.nan)  # predicted maps per run
    _, _, n_vertices  = density_data[hemi].shape #get total number of vertices within fsaverage hemisphere
    
    #select runs and average func maps across runs
    n_target_runs = 3
    all_C = []              # will store (n_subj, 3, n_vertices)
    rnd_run_idx = np.full((n_subj, n_target_runs, len(hemis)), np.nan)
    for s, participant in enumerate(participants):

        C_full = norm_contrast_data[hemi][s][contrast]   # (n_runs_available, n_vertices)
        n_runs_available = C_full.shape[0]

        # ---- choose runs ----
        if "NS" in participant:
            run_idx = np.arange(3)
        elif "EB" in participant:
            run_idx = np.random.choice(
                n_runs_available, size=3, replace=False
            )

        rnd_run_idx[s, :, h] = run_idx

        # ---- extract runs ----
        C_sel = C_full[run_idx, :]     # (3, n_vertices)
        all_C.append(C_sel)
    all_C = np.stack(all_C, axis=0)
    C_mean = all_C.mean(axis=1)


    # -------------------------
    # main loop
    # -------------------------
    for s, participant in enumerate(participants):

        print(f"Fitting participant {participant}")

        # X: density maps for this participant
        # shape: (n_vertices_masked, n_tracts)
        X = norm_density_data[hemi][s].T

        # y: functional contrast map
        # shape: (n_vertices_masked,)
        y = C_mean[s, :]

        # Fit regression
        linreg = LinearRegression()
        linreg.fit(X, y)

        # Store coefficients
        trained_coefs[:, s, h] = linreg.coef_.copy()

        # Predict (optional)
        y_pred = linreg.predict(X)
        predicted[s, :] = y_pred

        # Reliability (if needed)
        reliability[s, h] = vertex_bootstrap_reliability(all_C[s,:,:])

        # Optional evaluation
        r, p = pearsonr(y, y_pred)
        rs[s,h] = r 

        rsquared[s,h] = r2_score(y, y_pred)

        mse_full = mean_squared_error(y, y_pred)
        print(f"r={r:.4f}, MSE={mse_full:.4e}, p={p:.4e}")

        ## 1. shuffling across predicted maps vertices
        # for i in range(n_bootstrap):
            # r_rand[i,s,h], r2_rand[i,s,h] = null_pearson_r2(y, y_pred)
    ## 2. Shuffling accross predicted maps of participants
    # for i in range(n_bootstrap):
    #     subj_perm = np.random.permutation(np.arange(len(participants)))
    #     for s, participant in enumerate(participants):
    #             r_rand[i,s,h], _ = pearsonr(predicted[subj_perm[s],:], C_mean[s, :])
    #             r2_rand[i,s,h] =r2_score(C_mean[s, :], predicted[subj_perm[s],:])
    ## 3. Shuffling accross participants within each group

    for i in range(n_bootstrap):
        
        for g, group in enumerate(groups):
            participants_group = [p for p in participants if group in p]
            if "EB" in group:
                group1_size = len(participants_group)
                participant_idx =[p for p in range(len(participants_group))]
                shuf_participants1 = np.random.permutation(participant_idx)
            elif "NS" in group:
                participant_idx =[p for p in group1_size+np.arange(len(participants_group))]
                shuf_participants2 = np.random.permutation(participant_idx)
        subj_perm = np.concatenate((shuf_participants1, shuf_participants2),
        axis=0)
        for s, participant in enumerate(participants):
            r_rand[i,s,h], _ = pearsonr(predicted[subj_perm[s],:], C_mean[s, :])
            r2_rand[i,s,h] =r2_score(C_mean[s, :], predicted[subj_perm[s],:])

#===============================
# Figures
#===============================

#Plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_null_vs_actual(
    rs, rsquared,
    r_rand, r2_rand,
    hemis,
    n_bins=40
):

    n_subj = rs.shape[0]
    n_hemis = len(hemis)

    fig, axes = plt.subplots(
        2, n_hemis,
        figsize=(6 * n_hemis, 8),
        constrained_layout=True
    )

    for h, hemi in enumerate(hemis):

        # ======================================================
        # ---------------- PEARSON r ---------------------------
        # ======================================================

        ax = axes[0, h] if n_hemis > 1 else axes[0]

        null_r = r_rand[:, :, h].mean(axis = 1).flatten()
        actual_r = rs[:, h]

        # Shared bins
        # combined = np.concatenate([null_r, actual_r])
        bins = np.histogram_bin_edges(null_r, bins=n_bins)

        # Null histogram
        null_hist = ax.hist(
            null_r,
            bins=bins,
            density=True,
            alpha=0.3,
            label="Null (hist)"
        )

        # Null KDE
        kde_null = gaussian_kde(null_r)
        x_vals = np.linspace(null_r.min(), null_r.max(), 500)
        y_null_kde = kde_null(x_vals)
        ax.plot(x_vals, y_null_kde, linewidth=2, color="blue", label="Null (KDE)")


        # 95% CI (null)
        ci_low, ci_high = np.percentile(null_r, [2.5, 97.5])
        ax.axvline(ci_low, linestyle="--")
        ax.axvline(ci_high, linestyle="--")

        ax.axvline(null_r.mean(), linestyle=":", linewidth=2, label="Null mean")
        ax.axvline(actual_r.mean(), linestyle=":", linewidth=2, color="red", label="True mean")

        # Force same y-scale
        ymax = max(
            max(null_hist[0]),
            max(y_null_kde)
        ) #max(true_hist[0])
        ax.set_ylim(0, ymax * 1.1)

        ax.set_title(f"{hemi} Hemisphere — Pearson r")
        ax.set_xlabel("r")
        ax.set_ylabel("Density")
        ax.legend()


        # ======================================================
        # -------------------- R² ------------------------------
        # ======================================================

        ax = axes[1, h] if n_hemis > 1 else axes[1]

        null_r2 = r2_rand[:, :, h].mean(axis = 1).flatten()
        actual_r2 = rsquared[:, h]

        combined = np.concatenate([null_r2, actual_r2])
        bins = np.histogram_bin_edges(null_r2, bins=n_bins)

        null_hist = ax.hist(
            null_r2,
            bins=bins,
            density=True,
            alpha=0.3,
            label="Null (hist)"
        )

        kde_null = gaussian_kde(null_r2)
        x_vals = np.linspace(null_r2.min(), null_r2.max(), 500)
        y_null_kde = kde_null(x_vals)
        ax.plot(x_vals, y_null_kde, linewidth=2, color="blue", label="Null (KDE)")

        ci_low, ci_high = np.percentile(null_r2, [2.5, 97.5])
        ax.axvline(ci_low, linestyle="--")
        ax.axvline(ci_high, linestyle="--")

        ax.axvline(null_r2.mean(), linestyle=":", linewidth=2, label="Null mean")
        ax.axvline(actual_r2.mean(), linestyle=":", linewidth=2, color="red", label="True mean")

        ymax = max(
            max(null_hist[0]),
            max(y_null_kde)
        ) #            max(true_hist[0]),
        ax.set_ylim(0, ymax * 1.1)

        ax.set_title(f"{hemi} Hemisphere — R²")
        ax.set_xlabel("R²")
        ax.set_ylabel("Density")
        ax.legend()

    plt.show()


plot_null_vs_actual(
    rs, rsquared,
    r_rand, r2_rand,
    hemis
)

#====================
#Plot per group
#====================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_null_vs_actual_by_group_horizontal(
    rs, rsquared,
    r_rand, r2_rand,
    hemis,
    participants,
    groups=("EB", "NS"),
    n_bins=40
):

    n_hemis = len(hemis)
    n_groups = len(groups)

    fig, axes = plt.subplots(
        2, n_hemis * n_groups,
        figsize=(4 * n_hemis * n_groups, 8),
        constrained_layout=True
    )

    group_indices = {
        g: np.array([i for i, p in enumerate(participants) if g in p])
        for g in groups
    }

    null_colors = {
        "EB": "green",
        "NS": "blue"
    }

    for h, hemi in enumerate(hemis):

        # ==========================================================
        # Precompute axis limits per metric & hemisphere
        # ==========================================================

        # ---- Pearson r ----
        all_r_values = []
        for g in groups:
            idx = group_indices[g]
            null_vals = r_rand[:, idx, h].mean(axis=1)
            true_vals = rs[idx, h]
            all_r_values.extend(null_vals)
            all_r_values.extend(true_vals)

        r_min, r_max = min(all_r_values), max(all_r_values)

        # ---- R² ----
        all_r2_values = []
        for g in groups:
            idx = group_indices[g]
            null_vals = r2_rand[:, idx, h].mean(axis=1)
            true_vals = rsquared[idx, h]
            all_r2_values.extend(null_vals)
            all_r2_values.extend(true_vals)

        r2_min, r2_max = min(all_r2_values), max(all_r2_values)

        # ==========================================================
        # Plot per group
        # ==========================================================

        for g_idx, group in enumerate(groups):

            col_idx = h * n_groups + g_idx
            idx = group_indices[group]
            color = null_colors[group]

            # ==================================================
            # ---------------- Pearson r ----------------------
            # ==================================================
            ax = axes[0, col_idx]

            null_vals = r_rand[:, idx, h].mean(axis=1)
            true_vals = rs[idx, h]

            # Histogram
            ax.hist(
                null_vals,
                bins=n_bins,
                density=True,
                alpha=0.35,
                color=color,
                orientation="horizontal",
                label="Null histogram"
            )

            # KDE
            kde = gaussian_kde(null_vals)
            y_vals = np.linspace(r_min, r_max, 500)
            x_kde = kde(y_vals)

            ax.plot(
                x_kde, y_vals,
                linewidth=2,
                color=color,
                label="Null KDE"
            )

            # 95% CI
            ci_low, ci_high = np.percentile(null_vals, [2.5, 97.5])
            ax.axhline(ci_low, linestyle="--", color=color, label="Null 95% CI")
            ax.axhline(ci_high, linestyle="--", color=color)

            # Null mean
            ax.axhline(
                null_vals.mean(),
                linestyle=":",
                linewidth=2,
                color=color,
                label="Null mean"
            )

            # True group mean
            ax.axhline(
                true_vals.mean(),
                linewidth=3,
                color="red",
                label="True mean"
            )

            ax.set_ylim(r_min, r_max)
            ax.set_title(f"{hemi} — {group} (r)")
            ax.set_ylabel("Pearson r")
            ax.set_xlabel("Density")
            ax.legend()

            # ==================================================
            # ------------------- R² ---------------------------
            # ==================================================
            ax = axes[1, col_idx]

            null_vals = r2_rand[:, idx, h].mean(axis=1)
            true_vals = rsquared[idx, h]

            ax.hist(
                null_vals,
                bins=n_bins,
                density=True,
                alpha=0.35,
                color=color,
                orientation="horizontal",
                label="Null histogram"
            )

            kde = gaussian_kde(null_vals)
            y_vals = np.linspace(r2_min, r2_max, 500)
            x_kde = kde(y_vals)

            ax.plot(
                x_kde, y_vals,
                linewidth=2,
                color=color,
                label="Null KDE"
            )

            ci_low, ci_high = np.percentile(null_vals, [2.5, 97.5])
            ax.axhline(ci_low, linestyle="--", color=color, label="Null 95% CI")
            ax.axhline(ci_high, linestyle="--", color=color)

            ax.axhline(
                null_vals.mean(),
                linestyle=":",
                linewidth=2,
                color=color,
                label="Null mean"
            )

            ax.axhline(
                true_vals.mean(),
                linewidth=3,
                color="red",
                label="True mean"
            )

            ax.set_ylim(r2_min, r2_max)
            ax.set_title(f"{hemi} — {group} (R²)")
            ax.set_ylabel("R²")
            ax.set_xlabel("Density")
            ax.legend()

    plt.suptitle("Null vs True Distributions by Hemisphere and Group", fontsize=16)
    plt.show()

plot_null_vs_actual_by_group_horizontal(
    rs, rsquared,
    r_rand, r2_rand,
    hemis,
    participants,
    groups=("EB", "NS"),
    n_bins=40
)



#===================================================
# Plot only Pearson's R per group
#===================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

def plot_null_vs_actual_by_group_horizontal(rs, r_rand, hemis, participants, groups=("EB", "NS"), n_bins=40):

    n_hemis = len(hemis)
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_hemis * n_groups,figsize=(4 * n_hemis * n_groups, 6),constrained_layout=True)

    group_indices = {g: np.array([i for i, p in enumerate(participants) if g in p]) for g in groups}
    null_colors = {"EB": "green","NS": "blue"}

    for h, hemi in enumerate(hemis):

        # ==========================================================
        # Precompute axis limits per metric & hemisphere
        # ==========================================================

        # ---- Pearson r ----
        all_r_values = []
        for g in groups:
            idx = group_indices[g]
            null_vals = r_rand[:, idx, h].mean(axis=1)
            true_vals = rs[idx, h]
            all_r_values.extend(null_vals)
            all_r_values.extend(true_vals)

        r_min, r_max = min(all_r_values), max(all_r_values)

      
        # ==========================================================
        # Plot per group
        # ==========================================================

        for g_idx, group in enumerate(groups):

            col_idx = h * n_groups + g_idx
            idx = group_indices[group]
            color = null_colors[group]

            # ==================================================
            # ---------------- Pearson r ----------------------
            # ==================================================
            ax = axes[col_idx]

            null_vals = r_rand[:, idx, h].mean(axis=1)
            true_vals = rs[idx, h]
            true_mean = true_vals.mean()

            p_val = np.sum(null_vals >= true_mean) / len(null_vals)

            # Histogram
            ax.hist(null_vals,bins=n_bins,density=True,alpha=0.35,color=color,orientation="horizontal",label="Null histogram")

            # KDE
            kde = gaussian_kde(null_vals)
            y_vals = np.linspace(r_min, r_max, 500)
            x_kde = kde(y_vals)

            ax.plot(
                x_kde, y_vals,
                linewidth=2,
                color=color,
                label="Null KDE"
            )

            # 95% CI
            ci_low, ci_high = np.percentile(null_vals, [2.5, 97.5])
            ax.axhline(ci_low, linestyle="--", color=color, label="Null 95% CI")
            ax.axhline(ci_high, linestyle="--", color=color)

            # Null mean
            ax.axhline(null_vals.mean(),linestyle=":",linewidth=2,color=color,label="Null mean")

            # True group mean
            ax.axhline(true_mean,linewidth=3, color="red",label="True mean")


            # --- DISPLAY P-VALUE ON PLOT ---
            # Position it at the top of the KDE/Histogram
            # ax.text(
            #     x=8,                     # Horizontal position (check your x-scale)
            #     y=true_mean + 0.05,      # Just above the red line
            #     s=f"p = {p_val:.3f}",
            #     fontsize=16,
            #     fontweight='bold',
            #     color='red',
            #     ha='center'
            # )
            ax.set_ylim(r_min, r_max)
            ax.set_xlim(0, 16)
            sns.despine(ax=ax, top=True, right=True)
            ax.spines['left'].set_linewidth(2) #axis thickness
            ax.spines['bottom'].set_linewidth(2) #axis thickness
            ax.set_title(f"{hemi} — {group} (r)")
            plt.setp(ax.get_xticklabels(),fontsize=18,fontweight='bold')
            plt.setp(ax.get_yticklabels(),fontsize=18,fontweight='bold')
            ax.set_ylabel("Pearson's r", fontsize=18, fontweight = 'bold')
            ax.set_xlabel("Density",fontsize=18, fontweight = 'bold')
            
            if col_idx == 3:
                ax.legend()

    plt.suptitle("Null vs True Distributions by Hemisphere and Group", fontsize=16)
    saveDir = op.join(bids_path, 'analysis', 'plots')
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(op.join(saveDir, "participants_permutation_pearsonr_linearreg_combined_tracts.png"),
            dpi=300, bbox_inches='tight')
    plt.show()

plot_null_vs_actual_by_group_horizontal(
    rs, 
    r_rand, 
    hemis,
    participants,
    groups=("EB", "NS"),
    n_bins=40
)