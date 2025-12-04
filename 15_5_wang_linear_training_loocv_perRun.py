#Loic Daumail
#Started on 11/10/2025
#Trains a linear regression to predict functional activation based on tract end point densitiess

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import nibabel as nib
import os
import os.path as op

# ----------------------------
# Inputs preparation
# ----------------------------

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
density_dir = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps')
func_dir = op.join(bids_path, 'analysis', 'fMRI_data')
# -----------------------
#1 Generate endpoint densities array
#-------------------------

# ✅ Fixed tract order (keep consistent across subjects!)
tract_order = ['MTxLGN', 'MTxPT', 'MTxSTS1', 'MTxPU', 'MTxFEF','MTxhIP','MTxV1'] #
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
            matches = [f for f in os.listdir(subj_dir) if f"wang{tract}" in f and f"hemi-{hemi_fs}" in f and "fsaverage" in f and f.endswith("fsprojdensity0mm.mgh")]

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


#---------------------------
## Fit linear model to data
#---------------------------

verbose = True

hemis = ["L", "R"]
contrast = contrast_order[0]   # e.g. "motionXstationary"

# get n_subj, n_tracts
n_subj, n_tracts, _  = density_data["L"].shape
rs   = np.full((n_subj, n_tracts, len(hemis)), np.nan)
mses = np.full((n_subj, n_tracts, len(hemis)), np.nan)

for h, hemi in enumerate(hemis):

    _, _, n_vertices  = density_data[hemi].shape
    densities = density_data[hemi]        # (subj, tract, vertices)
    subj_contrasts = contrast_data[hemi]  # list: one dict per participant

    # ----------------------------
    # Load MT ROI
    # ----------------------------
    wang_hmt_path = op.join(
        '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
        f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang.mgh"
    )
    surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
    wang_hmt_vertices = np.where(surf_roi > 0)[0]
    print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")

    # Masking densities (subj, tract, masked_vertices)
    densities_masked = densities[:, :, wang_hmt_vertices]
    n_masked = densities_masked.shape[2]

    # Predicted and coef storage
    predicted = np.full((n_subj, n_tracts, 6, n_masked), np.nan)  # predicted maps per run
    trained_coefs = np.zeros((n_subj, n_tracts, 6))  # scalar summary per tract/run

    # -------------------------
    # main loop: subject -> tract -> run-LOOCV
    # -------------------------
    for s in range(n_subj):

        # get this subject's run maps for the chosen contrast
        subj_dict = subj_contrasts[s]
        if contrast not in subj_dict:
            print(f"Subject {s} missing contrast {contrast} for hemi {hemi}, skipping")
            continue

        C_full = subj_dict[contrast]              # (n_runs, n_vertices_fullspace)
        # mask ROI
        C = C_full[:, wang_hmt_vertices]          # (n_runs, n_masked)
        n_runs = C.shape[0]

        if verbose:
            print(f"\nSubject {s+1}: {n_runs} runs (hemi {hemi})")

        # For each tract, we will treat each run as one sample:
        # X_train shape -> (n_train_runs, n_masked)
        # y_train shape -> (n_train_runs, n_masked)
        # This is a multi-output regression: features = vertices, outputs = vertices

        for tract_idx in range(n_tracts):

            if verbose:
                print(f" Tract {tract_idx+1}/{n_tracts}")

            # anatomical vector for this subject and tract (length = n_masked)
            anat_vec = densities_masked[s, tract_idx, :]  # shape (n_masked,)

            # NOTE: anat_vec does NOT vary by run. We nonetheless use different runs as different samples
            # for multi-output regression (X rows repeated). This may be ill-conditioned if n_runs is small.

            for test_run in range(n_runs):

                if verbose:
                    print(f"  Test run {test_run+1}/{n_runs}")

                # training run indices
                train_runs = [r for r in range(n_runs) if r != test_run]
                if len(train_runs) < 1:
                    # cannot train with zero samples -> skip
                    continue

                # Build X_train: one row per train run, columns = vertices/features
                # Because anatomy does not vary across runs, this will be repeated rows of the same anat_vec.
                # Still construct correctly:
                # Build training data
                # Build training data
                X_train = densities_masked[s, tract_idx, :][None, :]
                X_train = np.repeat(X_train, len(train_runs), axis=0)   # (n_train_runs, V)

                # y_train: keep 2D, do NOT flatten
                y_train = np.squeeze(C[train_runs, :])                              # (n_train_runs, V)

                # Standardize X by vertices
                scalerX = StandardScaler().fit(X_train)
                Xtr = scalerX.transform(X_train)

                # Standardize y by vertices (multi-output)
                scaly = StandardScaler().fit(y_train)
                ytr = scaly.transform(y_train)

                # Train linear model (multi-output regression)
                linreg = LinearRegression()
                linreg.fit(Xtr, ytr)

                trained_coefs[s, tract_idx, test_run] = linreg.coef_.mean()

                # Predict left-out run
                X_test = densities_masked[s, tract_idx, :][None, :]
                X_test_s = scalerX.transform(X_test)

                y_pred_std = linreg.predict(X_test_s)      # (1, V)
                y_pred = scaly.inverse_transform(y_pred_std)[0]

                predicted[s, tract_idx, test_run, :] = y_pred


                # Evaluate this test_run if verbose
                if verbose:
                    y_run_true = np.squeeze(C[test_run, :])
                    r_run, p_run = pearsonr(np.squeeze(y_run_true), y_pred)
                    mse_run = mean_squared_error(np.squeeze(y_run_true), y_pred)
                    print(f"   run r={r_run:.4f}, MSE={mse_run:.4e}, p={p_run:.4e}")

    # ------------------------------------
    # Compute subject-level performance (concatenate runs)
    # ------------------------------------
    for s in range(n_subj):
        subj_dict = subj_contrasts[s]
        if contrast not in subj_dict:
            continue
        C_full = subj_dict[contrast]
        C = C_full[:, wang_hmt_vertices]
        n_runs = C.shape[0]

        for t in range(n_tracts):
            # flatten across runs
            y_true = C.reshape(-1)
            y_pred = predicted[s, t, :n_runs, :].reshape(-1)

            # if predictions are nan (e.g., missing), skip
            if np.isnan(y_pred).all():
                continue

            r_all, _ = pearsonr(y_true, y_pred)
            mse_all = mean_squared_error(y_true, y_pred)
            rs[s, t, h] = r_all
            mses[s, t, h] = mse_all

    print(f"\nFinished hemisphere {hemi}")




    # Optionally save predicted maps per subject using a reference image
# ----------------------------------------------------------
# Save predicted maps and true maps for each subject
# ----------------------------------------------------------
out_dir = op.join(bids_path, 'analysis', 'diff2func_model_fits', 'linearcv_predicted_maps')
os.makedirs(out_dir, exist_ok=True)

# Use Wang hMT map as reference for saving surface MGZ/MGH
ref_img_for_save = nib.load(wang_hmt_path)
ref_affine = ref_img_for_save.affine
ref_header = ref_img_for_save.header

for s in range(n_subj):

    participant = participants[s]
    subj_dir = op.join(out_dir, participant)
    os.makedirs(subj_dir, exist_ok=True)

    # Determine localizer type
    if "EB" in participant:
        task = "ptlocal"
    elif "NS" in participant:
        task = "mtlocal"
    else:
        task = "localizer"  # fallback if needed

    # --------------------------------------
    # Save true contrast map (masked/unmasked)
    # --------------------------------------
    true_map = true_full[s, :]        # (n_vertices)
    true_map = true_map.reshape((1, 1, n_vertices)).astype(np.float32)

    true_out = op.join(
        subj_dir,
        f"{participant}_task-{task}_hemi-{hemi}_space-fsaverage_desc-{contrast_order[0]}_tstat_wangmask.mgz"
    )

    nib.save(nib.MGHImage(true_map, ref_affine, ref_header), true_out)

    # --------------------------------------
    # Save predicted map for each tract
    # --------------------------------------
    for t, tract in enumerate(tract_order):

        pred_map = predicted_full[s, t, :]   # (n_vertices,)
        pred_map = pred_map.reshape((1, 1, n_vertices)).astype(np.float32)

        pred_out = op.join(
            subj_dir,
            f"{participant}_hemi-{hemi}_label-{tract}_desc-predicted_contrast.mgz"
        )

        nib.save(nib.MGHImage(pred_map, ref_affine, ref_header), pred_out)

print(f"Saved predicted & true maps to: {out_dir}")



#--------------------------------------------------------------

#Plotting results
import matplotlib.pyplot as plt
# ============================
# HEATMAP OF PEARSON r
# ============================

for h, hemi in enumerate(hemis):

    plt.figure(figsize=(10, 6))
    plt.imshow(rs[:, :, h], aspect='auto', interpolation='nearest')
    plt.colorbar(label='Pearson r')

    plt.xlabel("Tracts")
    plt.ylabel("Participants")
    plt.title(f"Correlation Between Predicted and True Maps\n Hemisphere: {hemi}")

    # Tract labels
    plt.xticks(np.arange(n_tracts), tract_order, rotation=45)

    # Participant labels
    plt.yticks(np.arange(n_subj), participants)

    plt.tight_layout()

    saveDir = op.join(bids_path, 'analysis', 'plots')
    os.makedirs(saveDir, exist_ok=True)

    plt.savefig(op.join(saveDir, f"hemi-{hemi}_pearsonrs_linearcv_perrun_heatmap.png"),
                dpi=300, bbox_inches='tight')
    plt.show()



# ==========================================
# GROUPED BAR PLOT (MEAN ± SD PER GROUP × HEMI)
# ==========================================

import pandas as pd

all_rows = []
for p, participant in enumerate(participants):
    gp = "EB" if "EB" in participant else "NS"
    for h, hemi in enumerate(hemis):
        for t, tract in enumerate(tract_order):
            all_rows.append({
                "participant": participant,
                "hemisphere": hemi,
                "tract": tract,
                "correlation": rs[p, t, h],
                "group": gp,
            })

df_pearson = pd.DataFrame(all_rows)

summary = (
    df_pearson.groupby(["group", "hemisphere"])["correlation"]
    .agg(["mean", "std"])
    .reset_index()
)

groups = ["EB", "NS"]
hemispheres = hemis
colors = {"EB": "#1f77b4", "NS": "#ff7f0e"}

means = np.zeros((len(groups), len(hemispheres)))
stds  = np.zeros((len(groups), len(hemispheres)))

for gi, g in enumerate(groups):
    for hi, h in enumerate(hemispheres):
        row = summary[(summary["group"] == g) & (summary["hemisphere"] == h)]
        means[gi, hi] = row["mean"].values[0]
        stds[gi, hi]  = row["std"].values[0]

x = np.arange(len(hemispheres))
width = 0.35

plt.figure(figsize=(10, 6))

for gi, g in enumerate(groups):
    plt.bar(
        x + gi * width - width/2,
        means[gi],
        yerr=stds[gi],
        width=width,
        capsize=5,
        color=colors[g],
        alpha=0.9,
        label=g,
    )

plt.xticks(x, hemispheres)
plt.xlabel("Hemisphere")
plt.ylabel("Pearson r (mean ± SD)")
plt.title("Prediction Accuracy \nGrouped by Hemisphere and Group")
plt.legend(title="Group")

plt.tight_layout()

saveDir = op.join(bids_path, 'analysis', 'plots')
os.makedirs(saveDir, exist_ok=True)

plt.savefig(op.join(saveDir, "pearsonrs_linearcv_grouped_barplot_wang.png"),
            dpi=300, bbox_inches='tight')
plt.show()



# =====================================
# SCATTER + JITTER PLOT BY GROUP × HEMI × TRACT
# =====================================

import matplotlib.patches as mpatches

all_rows = []
for p, participant in enumerate(participants):
    gp = "EB" if "EB" in participant else "NS"
    for t, tract in enumerate(tract_order):
        for h, hemi in enumerate(hemis):
            all_rows.append({
                "participant": participant,
                "hemisphere": hemi,
                "tract": tract,
                "correlation": rs[p, t, h],
                "group": gp,
            })

df_pearson = pd.DataFrame(all_rows)

summary = (
    df_pearson.groupby(["group", "hemisphere", "tract"])["correlation"]
    .agg(["mean", "std"])
    .reset_index()
)

hemispheres = hemis
groups = ["EB", "NS"]
tracts = tract_order
colors = {"EB": "#1f77b4", "NS": "#ff7f0e"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for i, hemi in enumerate(hemispheres):
    ax = axes[i]
    hemi_data = df_pearson[df_pearson["hemisphere"] == hemi]

    for g, group in enumerate(groups):
        group_data = hemi_data[hemi_data["group"] == group]

        for j, tract in enumerate(tracts):
            tract_data = group_data[group_data["tract"] == tract]

            # jitter
            x_jitter = np.random.normal(j + g * 0.25, 0.04, size=len(tract_data))

            ax.scatter(
                x_jitter,
                tract_data["correlation"],
                color=colors[group],
                alpha=0.6,
                s=40,
            )

            m = summary.query(
                "group == @group and hemisphere == @hemi and tract == @tract"
            )["mean"]
            s = summary.query(
                "group == @group and hemisphere == @hemi and tract == @tract"
            )["std"]

            if not m.empty:
                ax.errorbar(
                    j + g * 0.25,
                    m.values[0],
                    yerr=s.values[0],
                    fmt="o",
                    color="black",
                    capsize=5,
                    markersize=6,
                    zorder=5,
                )

        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    ax.set_title(f"{'Left' if hemi == 'L' else 'Right'} Hemisphere")
    ax.set_xticks(np.arange(len(tracts)) + 0.125)
    ax.set_xticklabels(tracts, rotation=25, ha="right")
    if i == 0:
        ax.set_ylabel("Pearson R")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Legend
legend_handles = [
    mpatches.Patch(color=colors["EB"], label="EB"),
    mpatches.Patch(color=colors["NS"], label="NS"),
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
