#In this script, the goal is to assess any difference in overlap of endpoints with MT 
# ROI across the different tracts

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

tract_list = ['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF'] # 
hemispheres = ['L', 'R']

# ==== RESULTS STORAGE ====
all_rows = []

# ==== MAIN LOOP ====
for hemi in hemispheres:
    hemi_fs = 'lh' if hemi == 'L' else 'rh'
    #Load Wang MT ROI vertices:
    wang_hmt_path = op.join('/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
    f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh")
    surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
    wang_hmt_vertices = np.where(surf_roi > 0)[0]
    for participant in participants:
        print(f"\n=== Processing {participant} ===")

        #Load func MT ROI vertices
        label_file = op.join(analysis_path, 'ROIs', 'func_roi','functional_surf_roi', participant, f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label")
        func_hmt_vertices = read_label(label_file)

        #Func_MT and Wang MT vertices overlap
        overlap_mask = np.isin(wang_hmt_vertices, func_hmt_vertices)
        overlap_vertices = wang_hmt_vertices[overlap_mask]


        # Loop over tracts
        for tract in tract_list:

            density_map_file = op.join(proj_density_path,participant, 'wang_MT',
                f"{participant}_hemi-{hemi_fs}_space-fsaverage_label-wang{tract}_desc-fsprojdensity0mm2.mgh")

            density_map = nib.load(density_map_file).get_fdata().squeeze()

            # Extract density values at ROI vertices
            roi_density_values = density_map[overlap_vertices]
            
            # Compute stats
            nonzero_count = np.count_nonzero(roi_density_values)
            total_count = len(roi_density_values)
            # proportion = nonzero_count / total_count if total_count > 0 else np.nan
             #Dice similarity coefficient (DSC) = 2x number of vertices with non zero density within func MT / (number of vertices with density values+ number of func MT vertices)
            density_thresh = 9
            DSC =  2*np.sum((roi_density_values) > density_thresh) / (np.sum(density_map > density_thresh)+ np.sum(overlap_vertices > 0))
            all_rows.append({
                "participant": participant,
                "hemisphere": hemi,
                "tract": tract,
                "dice": DSC,
                "total_vertices": total_count,
                "nonzero_vertices": nonzero_count
            })

            print(f"{participant} {hemi_fs}-{tract}: {DSC:.2%}")

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
    df_results.groupby(["hemisphere", "tract", "group"])["dice"]
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
    axes[i].set_ylabel("Dice similarity coefficient" if i == 0 else "")
    axes[i].legend()

plt.tight_layout()
plt.show()


## Same plot with jitter scatter points 

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# Assume df_results has columns: participant, hemisphere, tract, proportion (0–1)
# Add a group label based on participant name pattern
df_results["group"] = df_results["participant"].apply(
    lambda x: "EB" if "EB" in x else ("NS" if "NS" in x else "Other")
)

# Compute mean and std per group × hemisphere × tract
summary = (
    df_results.groupby(["group", "hemisphere", "tract"])["dice"]
    .agg(["mean", "std"])
    .reset_index()
)

hemispheres = ["L", "R"]
groups = ["EB", "NS"]
tracts = df_results["tract"].unique()

# Define consistent colors per group
colors = {"EB": "#1f77b4", "NS": "#ff7f0e"}  # blue and orange

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, hemi in enumerate(hemispheres):
    ax = axes[i]
    hemi_data = df_results[df_results["hemisphere"] == hemi]

    for g, group in enumerate(groups):
        group_data = hemi_data[hemi_data["group"] == group]

        for j, tract in enumerate(tracts):
            tract_data = group_data[group_data["tract"] == tract]
            
            # Jitter x positions to avoid overlap
            x_jitter = np.random.normal(j + g * 0.25, 0.04, size=len(tract_data))
            
            # Scatter individual participant data
            ax.scatter(
                x_jitter,
                tract_data["dice"],
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

    ax.set_title(f"{'Left' if hemi == 'L' else 'Right'} Hemisphere")
    ax.set_xticks(np.arange(len(tracts)) + 0.125)
    ax.set_xticklabels(tracts, rotation=30, ha="right")
    ax.set_ylabel("Dice similarity coefficient" if i == 0 else "")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# After plotting everything, before plt.show():
legend_handles = [
    mpatches.Patch(color=colors["EB"], label="EB"),
    mpatches.Patch(color=colors["NS"], label="NS")
]
fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.95])
saveDir = op.join(bids_path, "analysis", "plots")
plt.savefig(op.join(saveDir, "dice_tracts_thresh_9.png"), dpi=300, bbox_inches='tight')
plt.show()


## Perform Stats