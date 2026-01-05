#Loic Daumail
#Started on 11/10/2025
#Trains a linear regression to predict functional activation based on tract end point densitiess

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import random
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
tract_order = ['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF'] #'MTxPT', 'MTxPU' 'MTxFEF','MTxhIP','MTxV1'
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


#---------------------------
## Fit linear model to data
#---------------------------

verbose = True

hemis = ["L", "R"]
contrast = contrast_order[0]   # e.g. "motionXstationary"

# get n_subj, n_tracts
n_subj, n_tracts, _  = density_data["L"].shape
rs   = np.full((3, n_subj, len(hemis)), np.nan)
mses = np.full((3, n_subj, len(hemis)), np.nan)
r_all = np.full(( n_subj, len(hemis)), np.nan)
rnd_run_idx = np.full((n_subj, 3), np.nan)
trained_coefs = np.zeros((3, n_tracts, n_subj, len(hemis)))  # scalar summary per tract/run
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
    predicted = np.full((n_subj, 3, n_masked), np.nan)  # predicted maps per run
    

    # -------------------------
    # main loop: subject -> tract -> run-LOOCV
    # -------------------------
    for s, participant in enumerate(participants):

        # get this subject's run maps for the chosen contrast
        subj_dict = subj_contrasts[s]
        if contrast not in subj_dict:
            print(f"Subject {s} missing contrast {contrast} for hemi {hemi}, skipping")
            continue

        if "NS" in participant:
            C_full = subj_dict[contrast]              # (n_runs, n_vertices_fullspace)
            rnd_run_idx[s,:] = [0, 1, 2]
        elif "EB" in participant: #need to randomize across runs selected for EB as they have 6 runs, and NS only has 3 runs
            r_idx = random.sample(range(6), 3)
            rnd_run_idx[s,:] = np.array(r_idx, dtype=int)
            C_full = subj_dict[contrast][r_idx,:]   
        # mask ROI
        C = np.squeeze(C_full[:, wang_hmt_vertices] )         # (n_runs, n_masked)
        #zscore the runs for a given participant
        n_runs = C.shape[0]
        zscored_C = np.array([(C[r_num,:] - np.mean(C[r_num,:]))/np.std(C[r_num,:]) for r_num in range(n_runs)])


        if verbose:
            print(f"\nSubject {s+1}: {n_runs} runs (hemi {hemi})")

        #Prepare zscored density maps for each tract
        zscored_densities = np.full((n_tracts, n_masked), np.nan)
        for tract_idx in range(n_tracts):

            if verbose:
                print(f" Tract {tract_idx+1}/{n_tracts}")

            # anatomical vector for this subject and tract (length = n_masked)
            # anat_vec = densities_masked[s, tract_idx, :]  # shape (n_masked,)
            #Zscore the density data of the tract of this participant
            if np.std(densities_masked[s, tract_idx, :]) == 0:
                zscored_densities[tract_idx,:] = 0
            else:
                zscored_densities[tract_idx,:] = (densities_masked[s, tract_idx, :] - np.mean(densities_masked[s, tract_idx, :]))/np.std(densities_masked[s, tract_idx, :])
            # np.sum(np.isnan(densities_masked))

        for test_idx in range(n_runs):

            if verbose:
                print(f" Left out run {test_idx+1}/{n_runs}")

            # training run indices
            train_idx = [r for r in range(n_runs) if r != test_idx]
            if len(train_idx) < 1:
                # cannot train with zero samples -> skip
                continue
            X_train = np.vstack([zscored_densities for _r in train_idx]).reshape(len(train_idx)*n_masked, n_tracts) # (n_train*n_masked, n_tracts)
            y_train = np.squeeze(zscored_C[train_idx, :]).reshape(-1, 1)  # (n_train*n_masked,)


            # Train linear model (multi-output regression)
            linreg = LinearRegression()
            linreg.fit(X_train, y_train)

            trained_coefs[test_idx,:, s, h] = linreg.coef_.copy() #[0:n_tracts-1]
            # linreg.intercept_

            X_test = zscored_densities.T
            y_pred_std = linreg.predict(X_test)
            y_pred = (y_pred_std*np.std(C[test_idx,:]) + np.mean(C[test_idx,:])).ravel()

            predicted[s, test_idx, :] = y_pred


            # Evaluate this test_run if verbose
            if verbose:
                y_run_true = np.squeeze(C[test_idx, :])
                r_run, p_run = pearsonr(np.squeeze(y_run_true), y_pred)
                mse_run = mean_squared_error(np.squeeze(y_run_true), y_pred)
                print(f"   run r={r_run:.4f}, MSE={mse_run:.4e}, p={p_run:.4e}")

        #------------------
        #Performance metrics
        #------------------
        #overall correlation across all runs (concatenated)
        r_all[s, h], _ = pearsonr(C.reshape(-1), predicted[s, :n_runs, :].reshape(-1))

        #Calculate correlation for a given run
        for r in range(n_runs):

            y_true = C[r, :].reshape(-1)
            y_pred = predicted[s, r, :].reshape(-1)

            # Skip if prediction missing
            if np.isnan(y_pred).all():
                continue

            r_r, _ = pearsonr(y_true, y_pred)
            mse_r = mean_squared_error(y_true, y_pred)

            rs[r, s, h] = r_r
            mses[r, s, h] = mse_r


    print(f"\nFinished hemisphere {hemi}")




    # Optionally save predicted maps per subject using a reference image
# ----------------------------------------------------------
# Save predicted maps 
# ----------------------------------------------------------
# ----------------------------------------------------------
# Save predicted maps (all 6 runs, with NaN-check)
# ----------------------------------------------------------

N_RUNS_SAVED = 6

predicted_full = np.full((n_subj, n_tracts, N_RUNS_SAVED, n_vertices), np.nan)
predicted_full[:, :, :, wang_hmt_vertices] = predicted

out_dir = op.join(bids_path, 'analysis', 'diff2func_model_fits', 'linearcv_loro_predicted_maps')
os.makedirs(out_dir, exist_ok=True)

ref_img_for_save = nib.load(wang_hmt_path)
ref_affine = ref_img_for_save.affine
ref_header = ref_img_for_save.header

for s in range(n_subj):

    participant = participants[s]
    subj_dir = op.join(out_dir, participant)
    os.makedirs(subj_dir, exist_ok=True)

    if "EB" in participant:
        task = "ptlocal"
    elif "NS" in participant:
        task = "mtlocal"
    else:
        task = "localizer"

    # -----------------------------------------------------
    # Save predicted map for each tract and each run
    # -----------------------------------------------------
    for t, tract in enumerate(tract_order):

        for r in range(N_RUNS_SAVED):

            pred_map = predicted_full[s, t, r, :]      # (n_vertices,)
            pred_map = pred_map.reshape((1, 1, n_vertices)).astype(np.float32)

            # --- NEW CHECK: skip run if map is empty ---
            if np.isnan(pred_map).all():
                continue

            pred_out = op.join(
                subj_dir,
                f"{participant}_hemi-{hemi}_label-{tract}_run-{r+1}_desc-predicted_contrast.mgz"
            )

            nib.save(nib.MGHImage(pred_map, ref_affine, ref_header), pred_out)

print(f"Saved predicted maps to: {out_dir}")



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



# =====================================
# SCATTER + JITTER PLOT BY GROUP × HEMI × TRACT
# =====================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# --------------------------------------
# Build long-format dataframe
# r_all shape should be (n_subj, 2 hemis)
# --------------------------------------

hemi_labels = ["L", "R"]
rows = []

for h in range(2):     # left/right hemispheres
    for s, participant in enumerate(participants):
        gp = "EB" if "EB" in participant else "NS"
                # # Determine allowed runs based on group
        if "NS" in gp:        # NS → only 3 runs
            valid_runs = range(3)      # 0,1,2
        else:                          # EB → all 6 runs
            valid_runs = range(6)
        pearson = rs[0:len(valid_runs):, s, h]

        rows.append({
            "Subject": s,
            "Hemisphere": hemi_labels[h],
            "Group": gp,
            "Correlation": pearson.mean() #r_all[s, h] #
        })

df = pd.DataFrame(rows)

# ------------------------------------------------
# Compute SEM per Group × Hemisphere
# ------------------------------------------------
sem_df = (
    df.groupby(["Group", "Hemisphere"])["Correlation"]
      .agg(["mean", sem])
      .reset_index()
      .rename(columns={"mean": "Mean", "sem": "SEM"})
)

# ------------------------------------------------
# Color palette: EB = blue, NS = orange
# ------------------------------------------------
palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

# ------------------------------------------------
# Create 2 subplots — one per hemisphere
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, hemi in zip(axes, hemi_labels):

    df_h = df[df["Hemisphere"] == hemi]
    sem_h = sem_df[sem_df["Hemisphere"] == hemi]

    # Jittered dots
    sns.stripplot(
        data=df_h,
        x="Group",
        y="Correlation",
        hue="Group",
        dodge=False,
        jitter=0.15,
        alpha=0.7,
        palette=palette,
        ax=ax
    )

    # ---- Mean ± SEM ----
    for _, row in sem_h.iterrows():
        group = row["Group"]
        mean = row["Mean"]
        se = row["SEM"]

        x_loc = 0 if group == "EB" else 1

        ax.errorbar(
            x=x_loc,
            y=mean,
            yerr=se,
            fmt="o",
            color="black",
            markersize=8,
            capsize=3,
            linewidth=2
        )

    # Formatting
    ax.set_title(f"{hemi}-Hemisphere", fontsize=16)
    ax.set_xlabel("Group", fontsize=14)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticklabels(["EB", "NS"], fontsize=13)

axes[0].set_ylabel("Pearson's r", fontsize=14)
# axes[1].get_legend().remove()   # remove duplicated legend
sns.despine()
plt.tight_layout()

# Saving
saveDir = op.join(bids_path, "analysis", "plots")
os.makedirs(saveDir, exist_ok=True)
# plt.savefig(op.join(saveDir, "pearson_mean_linreg_loro_combined_tracts_LGN-STS1.png"), dpi=300, bbox_inches='tight')

plt.show()

#----------------------------
### Plot beta values
#----------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# --------------------------------------
# Convert trained_coefs into long format
# trained_coefs shape = (6 runs, n_tracts, n_subj, 2 hemis)
# Requires subject_group: list of "EB" or "NS"
# --------------------------------------

runs, n_tracts, n_subj, n_hemi = trained_coefs.shape
hemi_labels = ["L", "R"]

rows = []
for h in range(n_hemi):
    for t in range(n_tracts):
        for s, participant in enumerate(participants):
            gp = "EB" if "EB" in participant else "NS"
            betas = trained_coefs[:, t, s, h]   # shape (6,)
            if np.isnan(betas).all():
                continue
            
            rows.append({
                "Tract": tract_order[t],
                "Subject": s,
                "Hemisphere": hemi_labels[h],
                "Group": gp,    # EB or NS
                "MeanBeta": np.nanmean(betas)
            })

df = pd.DataFrame(rows)


# ------------------------------------------------
# Compute SEM per tract × hemisphere × group (EB/NS)
# ------------------------------------------------
sem_df = (
    df.groupby(["Group", "Tract", "Hemisphere"])["MeanBeta"]
      .agg(["mean", sem])
      .reset_index()
      .rename(columns={"mean": "Mean", "sem": "SEM"})
)

# ------------------------------------------
# Create 2 subplots — one for each hemisphere
# ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
group_labels = ["EB", "NS"]
group_offset = {"EB": -0.2, "NS": 0.2}   # shift inside each tract

for ax, hemi in zip(axes, hemi_labels):

    # Filter for hemisphere
    df_h = df[df["Hemisphere"] == hemi]
    sem_h = sem_df[sem_df["Hemisphere"] == hemi]

    # Jitter dots per group (EB vs NS)
    sns.stripplot(
        data=df_h,
        x="Tract",
        y="MeanBeta",
        hue="Group",
        dodge=True,
        jitter=0.15,
        alpha=0.7,
        ax=ax
    )

    # ---------------------------------------
    # Plot Mean ± SEM for EB and NS separately
    # ---------------------------------------
    for _, row in sem_h.iterrows():

        tract = row["Tract"]
        mean  = row["Mean"]
        se    = row["SEM"]
        group = row["Group"]

        # shift within tract index to match stripplot's dodge layout
        x_loc = tract_order.index(tract)  + group_offset[group]

        # Mean point
        ax.plot(x_loc, mean, "o", color="black", markersize=7)

        # SEM bar
        ax.errorbar(
            x=x_loc,
            y=mean,
            yerr=se,
            color="black",
            capsize=3,
            linewidth=2
        )

    # ------------------ Ax formatting ------------------
    ax.set_title(f"{hemi}-Hemisphere β-coefficients (Mean ± SEM within EB / NS)",fontsize=16)
    ax.set_xlabel("Tract",fontsize=14)
    ax.tick_params(axis="x", rotation=90)
    ax.set_xticks(np.arange(len(tract_order)))
    ax.set_xticklabels(tract_order, rotation=30, ha="right",fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
axes[0].set_ylabel("Mean Beta", fontsize=14)
axes[1].legend(title="Group", labels=group_labels)
sns.despine()
plt.tight_layout()
saveDir = op.join(bids_path, 'analysis', 'plots')
os.makedirs(saveDir, exist_ok=True)

plt.savefig(op.join(saveDir, "betas_linreg_loro_combined_tracts_LGN-STS1-PT.png"),
            dpi=300, bbox_inches='tight')
plt.show()
