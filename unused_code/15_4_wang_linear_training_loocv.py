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
tract_order = ['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF'] #
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
            matches = [f for f in os.listdir(subj_dir) if f"wang{tract}" in f and f"hemi-{hemi_fs}" in f and "fsaverage" in f and f.endswith("fsprojdensity0mm2.mgh")] #f.endswith("fsprojdensity0mm.mgh")

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

contrast_order = ["motionXstationary", "motionXsilent"] # 
# Initialize storage dictionary
contrast_data = {hemi: [] for hemi in hemis}

for participant in participants:
    if not participant.startswith("sub-"):
        continue
    print(f"\n🔹 Participant: {participant}")
    contrasts_dir = op.join(func_dir, participant, 'glm', 'contrasts')
    # Loop by hemisphere
    for hemi in hemis:
        print(f"   🧩 Hemisphere: {hemi}")
        hemi_fs = "lh" if hemi == "L" else "rh"
        subj_contrasts = []

        for contrast in contrast_order:
            # Find file matching this tract and hemisphere
            if "EB" in participant:
                matches = [f for f in os.listdir(contrasts_dir) if f"hemi-{hemi}" in f and "fsaverage" in f and "ptlocal" in f and contrast in f and "tstat" in f]
            elif "NS" in participant: 
                matches = [f for f in os.listdir(contrasts_dir) if f"hemi-{hemi}" in f and "fsaverage" in f and "mtlocal" in f and contrast in f and "tstat" in f]
            # Load the file
            contrast_file = op.join(contrasts_dir, matches[0])
            img = nib.load(contrast_file)
            data = img.get_fdata().astype(np.float32)
            subj_contrasts.append(data)

        # Stack into one array: shape (n_tracts, n_vertices)
        subj_contrasts = np.stack(subj_contrasts, axis=0)  # (7, n_vertices)
        contrast_data[hemi].append(subj_contrasts)
        # for i, arr in enumerate(contrast_data[hemi]):
        #     print(f"{hemi} element {i}: shape = {arr.shape}")

# Convert to numpy arrays
for hemi in hemis:
    contrast_data[hemi] = np.squeeze(np.stack(contrast_data[hemi], axis=0))  # (n_subjects, n_contrasts, n_vertices)
    print(f"✅ {hemi}-hemisphere shape: {contrast_data[hemi].shape}")

#----------------
### Analyze data
#----------------

verbose = True

n_subj, n_tracts, n_vertices  = density_data["L"].shape
rs   = np.full((n_subj, n_tracts, len(hemis)), np.nan)
mses = np.full((n_subj, n_tracts, len(hemis)), np.nan)
trained_coefs = np.full((n_subj, n_tracts,len(hemis)), np.nan)
hemis = ["L", "R"]

for h, hemi in enumerate(hemis):

    densities = density_data[hemi]
    contrasts = contrast_data[hemi][:, 0, :]

    # ----------------------------
    # Load MT ROI
    # ----------------------------
    wang_hmt_path = op.join(
        '/Users','ldaumail3','Documents','research','brain_atlases',
        'Wang_2015','hmtplus',
        f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang.mgh"
    )
    surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
    wang_hmt_vertices = np.where(surf_roi > 0)[0]
    print(f"{len(wang_hmt_vertices)} vertices in ROI")

    # Masking
    densities_masked = densities[:, :, wang_hmt_vertices]  # (subj, tracts, vertices)
    contrasts_masked = contrasts[:, wang_hmt_vertices]     # (subj, vertices)

    n_masked = densities_masked.shape[2]

    # Storage per hemisphere
    predicted    = np.zeros_like(densities_masked)

    # --------------------------------
    # Train tract-specific linear model
    # --------------------------------
    for tract_idx in range(n_tracts):

        for test_idx in range(n_subj):
            if verbose:
                print(f"\nLOOCV: leaving out subject {test_idx+1}/{n_subj} — tract {tract_idx}")

            train_idx = [i for i in range(n_subj) if i != test_idx]

            # Build training data
            X_train = np.vstack([densities_masked[s, tract_idx] for s in train_idx]).reshape(-1, 1)
            y_train = np.hstack([contrasts_masked[s]            for s in train_idx])

            # Standardize
            scalerX = StandardScaler().fit(X_train)
            scaly   = StandardScaler().fit(y_train.reshape(-1, 1))

            Xtr = scalerX.transform(X_train)
            ytr = scaly.transform(y_train.reshape(-1, 1)).ravel()

            # --------------------------
            # SIMPLE LINEAR REGRESSION
            # --------------------------
            linreg = LinearRegression()
            linreg.fit(Xtr, ytr)

            # Save coefficient (one per tract)
            trained_coefs[test_idx, tract_idx, h] = linreg.coef_[0]

            # Predict left-out subject
            X_test = densities_masked[test_idx, tract_idx, :].reshape(-1, 1)
            X_test_s = scalerX.transform(X_test)

            y_pred_std = linreg.predict(X_test_s)
            y_pred = scaly.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()

            predicted[test_idx, tract_idx, :] = y_pred

            # Evaluate
            if verbose:
                y_true = contrasts_masked[test_idx]
                r, p = pearsonr(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                print(f"  r={r:.4f}, MSE={mse:.4e}, p={p:.4e}")

    # ------------------------------------
    # Save results
    # ------------------------------------
    predicted_full = np.full((n_subj, n_tracts, n_vertices), np.nan)
    predicted_full[:, :, wang_hmt_vertices] = predicted

    true_full = np.full((n_subj, n_vertices), np.nan)
    true_full[:, wang_hmt_vertices] = contrasts_masked

    # Store performance per tract × subject
    for s in range(n_subj):
        y_true = contrasts_masked[s]

        for t in range(n_tracts):
            y_pred = predicted[s, t, :]

            r, _ = pearsonr(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)

            rs[s, t, h] = r
            mses[s, t, h] = mse

    print("\nFinished hemisphere", hemi)




    # Optionally save predicted maps per subject using a reference image
    out_dir = op.join(bids_path, 'analysis', 'ridgecv_predicted_maps')
    os.makedirs(out_dir, exist_ok=True)
    ref_img_for_save_path = wang_hmt_path 
    ref_img_for_save = nib.load(ref_img_for_save_path)
    for s in range(n_subj):
        participant = participants[s]
        os.makedirs(op.join(out_dir, participant), exist_ok=True)

        if "EB" in participant:
            task = "ptlocal"
        elif "NS" in participant: 
            task = "mtlocal"
        true_map_path = os.path.join(out_dir, participant, f"{participant}_task-{task}_hemi-{hemi}_space-fsaverage_desc-{contrast_order[0]}_tstat_wangmask.mgz")
        true_map = true_full[s, :].reshape((1, 1, n_vertices)).astype(np.float32)
        nib.save(nib.MGHImage(true_map, ref_img_for_save.affine, ref_img_for_save.header), true_map_path)

        for t, tract in enumerate(tract_order):

            outpath = os.path.join(out_dir, participant, f"{participant}_hemi-{hemi}_label-{tract}_desc-predicted_contrast.mgz")
            # If ref_img is MGH and data must be shape (1,1,n_vertices) or so, adapt:
            data_to_save = predicted_full[s,t, :].reshape((1, 1, n_vertices)).astype(np.float32)
            # For surface mgz you may want a shape matching original — adapt as required
            nib.save(nib.MGHImage(data_to_save, ref_img_for_save.affine, ref_img_for_save.header), outpath)



#--------------------------------------------------------------

#Plotting results

#1. Heat map
import matplotlib.pyplot as plt
import numpy as np

# rs has shape (n_subj, n_tracts)
# rows = subjects
# columns = tracts

plt.figure(figsize=(10, 6))
plt.imshow(rs[:,:, h], aspect='auto', interpolation='nearest')
plt.colorbar(label='Pearson r')

plt.xlabel("Tracts")
plt.ylabel("Participants")
plt.title("Correlation Between Predicted and True Maps\n Right hemisphere")

# Tract labels
plt.xticks(np.arange(n_tracts), [f"{t}" for t in tract_order], rotation=45)

# Subject labels (optional)
plt.yticks(np.arange(n_subj), [f"{s}" for s in participants])

plt.tight_layout()
# saveDir = op.join(bids_path, 'analysis', 'plots')
# plt.savefig(op.join(saveDir,f"hemi-{hemi}_pearsonrs_linearcv_heatmap.png"), dpi=300, bbox_inches='tight')

plt.show()

## Bar plot

import matplotlib.pyplot as plt
import numpy as np
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
                "correlation": rs[p,t, h],
                "group": gp,
            })

# ==== STORE ALL RESULTS ====
df_pearson = pd.DataFrame(all_rows)
# rs shape: (n_subj, n_tracts)

# Compute mean and std per group × hemisphere × tract
summary = (
    df_pearson.groupby(["group", "hemisphere", "tract"])["correlation"]
    .agg(["mean", "std"])
    .reset_index()
)


groups = ["EB", "NS"]
hemispheres = hemis  # ["L", "R"]
colors = {"EB": "#1f77b4", "NS": "#ff7f0e"}

# ---- Build arrays for plotting ----
means = np.zeros((len(groups), len(hemispheres)))
stds  = np.zeros((len(groups), len(hemispheres)))

for gi, g in enumerate(groups):
    for hi, h in enumerate(hemispheres):
        row = summary[(summary["group"] == g) & (summary["hemisphere"] == h)]
        means[gi, hi] = row["mean"].values[0]
        stds[gi, hi]  = row["std"].values[0]

# ---- Plot grouped bar chart ----
x = np.arange(len(hemispheres))              # bar positions (L, R)
width = 0.35                                 # bar width

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

# ---- Labels ----
plt.xticks(x, hemispheres)
plt.xlabel("Hemisphere")
plt.ylabel("Pearson r (mean ± SD)")
plt.title("Prediction Accuracy \n Grouped by Hemisphere and Group")
plt.legend(title="Group")

plt.tight_layout()

# ---- Save ----
saveDir = op.join(bids_path, 'analysis', 'plots')
os.makedirs(saveDir, exist_ok=True)

plt.savefig(
    op.join(saveDir, "pearsonrs_linearcv_grouped_barplot_wang.png"),
    dpi=300, bbox_inches="tight"
)
plt.show()




## Plot with jitter/scatter points, devided by group

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
                "correlation": rs[p,t,h],
                "group": gp,
            })

# ==== STORE ALL RESULTS ====
df_pearson = pd.DataFrame(all_rows)

# Compute mean and std per group × hemisphere × tract
summary = (
    df_pearson.groupby(["group", "hemisphere", "tract"])["correlation"]
    .agg(["mean", "std"])
    .reset_index()
)

hemispheres = ["L", "R"]
groups = ["EB", "NS"]
tracts = df_pearson["tract"].unique()

# Define consistent colors per group
colors = {"EB": "#1f77b4", "NS": "#ff7f0e"}  # blue and orange

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, hemi in enumerate(hemispheres):
    ax = axes[i]
    hemi_data = df_pearson[df_pearson["hemisphere"] == hemi]

    for g, group in enumerate(groups):
        group_data = hemi_data[hemi_data["group"] == group]

        for j, tract in enumerate(tracts):
            tract_data = group_data[group_data["tract"] == tract]
            
            # Jitter x positions to avoid overlap
            x_jitter = np.random.normal(j + g * 0.25, 0.04, size=len(tract_data))
            
            # Scatter individual participant data
            ax.scatter(
                x_jitter,
                tract_data["correlation"],
                color=colors[group],
                alpha=0.6,
                s=40,
                label=group if (j == 0 and i == 0) else ""  # legend only once
            )
            
            # Plot mean ± std
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
                    lw=1.2,
                    zorder=5,
                )
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    ax.set_title(f"{'Left' if hemi == 'L' else 'Right'} Hemisphere")
    ax.set_xticks(np.arange(len(tracts)) + 0.125)
    ax.set_xticklabels(tracts, rotation=30, ha="right")
    ax.set_ylabel("Pearson R" if i == 0 else "")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# After plotting everything, before plt.show():
legend_handles = [
    mpatches.Patch(color=colors["EB"], label="EB"),
    mpatches.Patch(color=colors["NS"], label="NS")
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.95])
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

n_subj,n_tracts, n_hemi = trained_coefs.shape
hemi_labels = ["L", "R"]

rows = []
for h in range(n_hemi):
    for t in range(n_tracts):
        for s, participant in enumerate(participants):
            gp = "EB" if "EB" in participant else "NS"
            betas = trained_coefs[s,t, h]   # shape (6,)
            if np.isnan(betas).all():
                continue
            
            rows.append({
                "Tract": tract_order[t],
                "Subject": s,
                "Hemisphere": hemi_labels[h],
                "Group": gp,    # EB or NS
                "MeanBeta": betas #np.nanmean(betas)
            })

df = pd.DataFrame(rows)


# ------------------------------------------------
# Compute STD per tract × hemisphere × group (EB/NS)
# ------------------------------------------------
std_df = (
    df.groupby(["Group", "Tract", "Hemisphere"])["MeanBeta"]
      .agg(["mean", "std"])
      .reset_index()
      .rename(columns={"mean": "Mean", "std": "STD"})
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
    std_h = std_df[std_df["Hemisphere"] == hemi]

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
    # Plot Mean ± STD for EB and NS separately
    # ---------------------------------------
    for _, row in std_h.iterrows():

        tract = row["Tract"]
        mean  = row["Mean"]
        std   = row["STD"]
        group = row["Group"]

        # shift within tract index to match stripplot's dodge layout
        x_loc = tract_order.index(tract) + group_offset[group]

        # Mean point
        ax.plot(x_loc, mean, "o", color="black", markersize=7)

        # STD bar
        ax.errorbar(
            x=x_loc,
            y=mean,
            yerr=std,
            color="black",
            capsize=3,
            linewidth=2
        )

    # ------------------ Ax formatting ------------------
    ax.set_title(f"{hemi}-Hemisphere β-coefficients (Mean ± STD within EB / NS)",
                 fontsize=16)
    ax.set_xlabel("Tract", fontsize=14)
    ax.tick_params(axis="x", rotation=90)
    ax.set_xticks(np.arange(len(tract_order)))
    ax.set_xticklabels(tract_order, rotation=30, ha="right", fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

axes[0].set_ylabel("Mean Beta", fontsize=14)
axes[1].legend(title="Group", labels=group_labels)
sns.despine()
plt.tight_layout()

saveDir = op.join(bids_path, 'analysis', 'plots')
os.makedirs(saveDir, exist_ok=True)

plt.savefig(op.join(saveDir, "betas_linearcv_loso_jitter_wang_STD.png"),
            dpi=300, bbox_inches='tight')
plt.show()
