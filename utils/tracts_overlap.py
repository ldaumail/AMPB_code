#The purpose of this script is to calculate the overlap of different tracts to assess their similarity
#Loic Daumail 12/02/2025
import os
import os.path as op
import nibabel as nib
import numpy as np
#1 Load density maps

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
density_dir = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps')
participants = sorted([p for p in os.listdir(density_dir) if p.startswith("sub-")])
density_maps1 = ["PT","STS1"]
density_maps2 = ["LGN", "PU"]

hemisphere = ["Left", "Right"]
#PT x STS1
n_subj = len(participants)
DSC   = np.full((n_subj, len(hemisphere)), np.nan)
for p, participant in enumerate(participants):
    for h, hemi in enumerate(hemisphere):
        MTxPT_map_path = op.join(density_dir, f"{participant}/wang_MT/{participant}_ses-concat_desc-wang{hemi}MTxPT_tdi_map.nii.gz")
        MTxSTS1_map_path = op.join(density_dir, f"{participant}/wang_MT/{participant}_ses-concat_desc-wang{hemi}MTxSTS1_tdi_map.nii.gz")
        
        #Load data 
        MTxPT_map = nib.load(MTxPT_map_path).get_fdata() 
        MTxSTS1_map = nib.load(MTxSTS1_map_path).get_fdata() 
        
        #Dice similarity coefficient (DSC)
        DSC[p,h] =  2*np.sum((MTxPT_map * MTxSTS1_map) > 0) / (np.sum(MTxPT_map >0)+ np.sum(MTxSTS1_map > 0))

# LGN x PU
DSC   = np.full((n_subj, len(hemisphere)), np.nan)
for p, participant in enumerate(participants):
    for h, hemi in enumerate(hemisphere):
        MTxLGN_map_path = op.join(density_dir, f"{participant}/wang_MT/{participant}_ses-concat_desc-wang{hemi}MTxLGN_tdi_map.nii.gz")
        MTxPU_map_path = op.join(density_dir, f"{participant}/wang_MT/{participant}_ses-concat_desc-wang{hemi}MTxPU_tdi_map.nii.gz")
        
        #Load data 
        MTxLGN_map = nib.load(MTxLGN_map_path).get_fdata() 
        MTxPU_map = nib.load(MTxPU_map_path).get_fdata() 
        
        #Dice similarity coefficient (DSC)
        DSC[p,h] =  2*np.sum((MTxLGN_map * MTxPU_map) > 0) / (np.sum(MTxLGN_map >0)+ np.sum(MTxPU_map > 0))


#-------------
## Plot results
#-------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


all_rows = []
for p, participant in enumerate(participants):
    gp = "EB" if "EB" in participant else "NS"
    # for t, tract in enumerate(tract_order):
    for h, hemi in enumerate(hemisphere):
        all_rows.append({
            "participant": participant,
            "hemisphere": hemi,
            # "tract": tract,
            "correlation": DSC[p,h],
            "group": gp,
        })

# ==== STORE ALL RESULTS ====
df_DSC = pd.DataFrame(all_rows)

# Compute mean and std per group × hemisphere × tract
summary = (
    df_DSC.groupby(["group", "hemisphere"])["correlation"]
    .agg(["mean", "std"])
    .reset_index()
)

groups = ["EB", "NS"]
colors = {"EB": "#1f77b4", "NS": "#ff7f0e"}  # consistent colors
x_positions = {"EB": -0.12, "NS": +0.12}    # keeps groups close

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

for i, hemi in enumerate(hemisphere):
    ax = axes[i]
    hemi_data = df_DSC[df_DSC["hemisphere"] == hemi]

    for group in groups:
        group_data = hemi_data[hemi_data["group"] == group]

        # jitter around the group x-position
        x_jitter = np.random.normal(loc=x_positions[group], scale=0.03, size=len(group_data))

        # scatter individuals
        ax.scatter(
            x_jitter,
            group_data["correlation"],
            color=colors[group],
            alpha=0.6,
            s=45,
        )

        # mean ± std
        m = summary.query("group == @group and hemisphere == @hemi")["mean"]
        s = summary.query("group == @group and hemisphere == @hemi")["std"]

        if not m.empty:
            ax.errorbar(
                x_positions[group],
                float(m),
                yerr=float(s),
                fmt="o",
                color="black",
                markersize=6,
                capsize=4,
                lw=1,
                zorder=10,
            )

    # cosmetic improvements
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{hemi} Hemisphere")
    ax.set_ylim(0, 1)

    # grouped xticks under EB / NS
    ax.set_xticks([x_positions[g] for g in groups])
    ax.set_xticklabels(groups)

    if i == 0:
        ax.set_ylabel("Dice Coefficient")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# # Legend
# legend_handles = [
#     mpatches.Patch(color=colors[g], label=g) for g in groups
# ]
# fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
# MAIN TITLE
fig.suptitle(
    "Dice Similarity Coefficient for MTxPT and MTxSTS1 tracts",
    fontsize=14,
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
