#In this script, the goal is to assess any difference in overlap of endpoints with MT 
# ROI across the different tracts

import os
import os.path as op
import numpy as np
import nibabel as nib
from nibabel.freesurfer import read_geometry, read_label
import pandas as pd
import matplotlib.pyplot as plt

# ==== PATHS ====
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
analysis_path = op.join(bids_path, 'analysis')
proj_density_path = op.join(analysis_path, 'tdi_maps', 'dipy_wmgmi_tdi_maps')
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')

subjects_file = op.join(bids_path,'code','utils','study2_subjects_updated.txt')

# ==== SETTINGS ====
with open(subjects_file, 'r') as f:
    participants = [line.strip() for line in f if line.strip()]  # removes empty lines and \n

print(f"Loaded {len(participants)} participants:")
print(participants)

tract_list = ['MTxLGN', 'MTxPT', 'MTxSTS1', 'MTxPU']
hemispheres = ['L', 'R']

# ==== RESULTS STORAGE ====
all_rows = []

# ==== MAIN LOOP ====
for participant in participants:
    print(f"\n=== Processing {participant} ===")
    for hemi in hemispheres:
        hemi_fs = 'lh' if hemi == 'L' else 'rh'

        # Load MT ROI vertices
        label_file = op.join(analysis_path, 'functional_surf_roi', participant, f"{participant}_hemi-{hemi}_space-fsnative_label-MT_mask.label")
        label_vertices = read_label(label_file)

        # Loop over tracts
        for tract in tract_list:
            new_name = tract.replace("MT", "MTmask")
            tract_name = f"Left{new_name}" if hemi == 'L' else f"Right{new_name}"

            density_map_file = op.join(proj_density_path,participant,
                f"{participant}_hemi-{hemi}_space-fsnative_label-{tract_name}_desc-fsprojdensity0mm.mgh")

            density_map = nib.load(density_map_file).get_fdata().squeeze()

            # Extract density values at ROI vertices
            roi_values = density_map[label_vertices]
            
            # Compute stats
            nonzero_count = np.count_nonzero(roi_values)
            total_count = len(roi_values)
            proportion = nonzero_count / total_count if total_count > 0 else np.nan

            all_rows.append({
                "participant": participant,
                "hemisphere": hemi,
                "tract": tract_name,
                "proportion": proportion,
                "total_vertices": total_count,
                "nonzero_vertices": nonzero_count
            })

            print(f"{participant} {hemi_fs}-{tract_name}: {proportion:.2%} non-empty MT vertices")

# ==== STORE ALL RESULTS ====
df_results = pd.DataFrame(all_rows)
# save_path = op.join(analysis_path, "surface_density_summary_allsubs.csv")
# df_results.to_csv(save_path, index=False)
# print(f"\n✅ Saved results to: {save_path}")

# ==== PLOT RESULTS ====
# Compute group means and stds
df_results["group"] = df_results["participant"].apply(
    lambda x: "EB" if x.startswith("sub-EB") else ("NS" if x.startswith("sub-NS") else "Other")
)

# 2️⃣ Compute mean and std per hemisphere, tract, and group
summary = (
    df_results.groupby(["hemisphere", "tract", "group"])["proportion"]
    .agg(["mean", "std"])
    .reset_index()
)
#Figure

hemispheres = ["L", "R"]
groups = ["EB", "NS"]
colors = {"EB": "skyblue", "NS": "lightcoral"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, hemi in enumerate(hemispheres):
    hemi_data = summary[summary["hemisphere"] == hemi]
    tracts = hemi_data["tract"].unique()
    x = np.arange(len(tracts))
    width = 0.35  # space between EB and NS bars

    for j, group in enumerate(groups):
        grp_data = hemi_data[hemi_data["group"] == group]
        # Align by tract
        grp_means = [grp_data[grp_data["tract"] == t]["mean"].values[0] * 100 if t in grp_data["tract"].values else 0 for t in tracts]
        grp_stds = [grp_data[grp_data["tract"] == t]["std"].values[0] * 100 if t in grp_data["tract"].values else 0 for t in tracts]

        axes[i].bar(
            x + (j - 0.5) * width,  # offset for grouping
            grp_means,
            yerr=grp_stds,
            width=width,
            capsize=5,
            label=group,
            color=colors[group],
            edgecolor="black"
        )

    axes[i].set_title(f"{'Left' if hemi == 'L' else 'Right'} Hemisphere")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(tracts, rotation=30, ha="right")
    axes[i].set_ylabel("Non-empty MT vertices (%)" if i == 0 else "")
    axes[i].legend()

plt.tight_layout()
plt.show()