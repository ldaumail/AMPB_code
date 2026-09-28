#Plots functions

################## 1. Model Performance plots #################

#=======================================================
# Pearson's r in a bar plot
#=======================================================
import os
import os.path as op
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from scipy.stats import sem

def pearsonr_plot(rs, participants, bids_path, savefilename):

    # --------------------------------------
    # Organize Pearson's r in table
    # --------------------------------------

    hemi_labels = ["L", "R"]
    rows = []

    for h in range(2):     # left/right hemispheres
        for s, participant in enumerate(participants):
            gp = "EB" if "EB" in participant else "NS"
            # pearson = rs[:, s, h]

            rows.append({
                "Subject": s,
                "Hemisphere": hemi_labels[h],
                "Group": gp,
                "Correlation": rs[s,h] #r_all[s, h] #
            })

    df = pd.DataFrame(rows)
    # 1. Global styling for bold fonts
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})
    palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, hemi in zip(axes, hemi_labels):
        df_h = df[df["Hemisphere"] == hemi]

        # 1. Bar Plot (Save to variable 'b')
        b = sns.barplot(
            data=df_h, x="Group", y="Correlation", hue="Group",
            palette=palette, ax=ax, capsize=0.1, errorbar='se',
            alpha=0.6, edgecolor="black", linewidth=2, width=0.5
        )

        # 2. Stripplot (Remove 'hue' here to center the dots)
        sns.stripplot(
            data=df_h, x="Group", y="Correlation", palette=palette,
            ax=ax, size=7, jitter=0.1, alpha=0.9, edgecolor="black", linewidth=1
        )

        # 3. LEGEND RECONSTRUCTION (The fix for "invisible" legends)
        # We grab only the Bar handles (first two items)
        handles, labels = ax.get_legend_handles_labels()
        
        # We force the legend to draw even if it thinks it shouldn't
        # 3. MANUAL LEGEND (This cannot fail)
        # Create proxy artists for the legend
        patch_eb = mpatches.Patch(color=palette["EB"], label='EB', alpha=0.6)
        patch_ns = mpatches.Patch(color=palette["NS"], label='NS', alpha=0.6)
        
        # Place the legend
        leg = ax.legend(
            handles=[patch_eb, patch_ns], 
            title="Group", 
            loc='upper right', 
            frameon=False, 
            fontsize=14
        )
        
        # Force Bold
        if leg:
            plt.setp(leg.get_texts(), fontweight='bold')
            plt.setp(leg.get_title(), fontweight='bold')
        
        # Formatting and Bold Ticks
        ax.set_ylim(-0.2, 1)
        ax.set_title(f"{hemi}-Hemisphere", fontsize=22, fontweight='bold')
        ax.set_xlabel("Group", fontsize=18, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.tick_params(axis='both', which='major', labelsize=18, width=2.5)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    axes[0].set_ylabel("Pearson's r", fontsize=18, fontweight='bold')
    sns.despine()
    plt.tight_layout()
    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(
        op.join(saveDir,
        savefilename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# MSE plot
def mse_plot(mse, participants, bids_path, savefilename):
    # --------------------------------------
    # Organize Pearson's r in table
    # --------------------------------------

    hemi_labels = ["L", "R"]
    rows = []

    for h in range(2):     # left/right hemispheres
        for s, participant in enumerate(participants):
            gp = "EB" if "EB" in participant else "NS"
            # pearson = rs[:, s, h]

            rows.append({
                "Subject": s,
                "Hemisphere": hemi_labels[h],
                "Group": gp,
                "MSE": mse[s,h] #r_all[s, h] #
            })

    df = pd.DataFrame(rows)
    # 1. Global styling for bold fonts
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold'})
    palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, hemi in zip(axes, hemi_labels):
        df_h = df[df["Hemisphere"] == hemi]

        # 1. Bar Plot (Save to variable 'b')
        b = sns.barplot(
            data=df_h, x="Group", y="MSE", hue="Group",
            palette=palette, ax=ax, capsize=0.1, errorbar='se',
            alpha=0.6, edgecolor="black", linewidth=2, width=0.5
        )

        # 2. Stripplot (Remove 'hue' here to center the dots)
        sns.stripplot(
            data=df_h, x="Group", y="MSE", palette=palette,
            ax=ax, size=7, jitter=0.1, alpha=0.9, edgecolor="black", linewidth=1
        )

        # 3. LEGEND RECONSTRUCTION (The fix for "invisible" legends)
        # We grab only the Bar handles (first two items)
        handles, labels = ax.get_legend_handles_labels()
        
        # We force the legend to draw even if it thinks it shouldn't
        # 3. MANUAL LEGEND (This cannot fail)
        # Create proxy artists for the legend
        patch_eb = mpatches.Patch(color=palette["EB"], label='EB', alpha=0.6)
        patch_ns = mpatches.Patch(color=palette["NS"], label='NS', alpha=0.6)
        
        # Place the legend
        leg = ax.legend(
            handles=[patch_eb, patch_ns], 
            title="Group", 
            loc='upper right', 
            frameon=False, 
            fontsize=14
        )
        
        # Force Bold
        if leg:
            plt.setp(leg.get_texts(), fontweight='bold')
            plt.setp(leg.get_title(), fontweight='bold')
        
        # Formatting and Bold Ticks
        ax.set_ylim(-0.2, 1)
        ax.set_title(f"{hemi}-Hemisphere", fontsize=22, fontweight='bold')
        ax.set_xlabel("Group", fontsize=18, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.tick_params(axis='both', which='major', labelsize=18, width=2.5)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

    axes[0].set_ylabel("MSE", fontsize=18, fontweight='bold')
    sns.despine()
    plt.tight_layout()
    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(
        op.join(saveDir,
        savefilename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def beta_plot(betas, participants, tract_order, bids_path, savefilename):

    tract_labels = [
        t.replace("InferiorLongitudinal", "ILF")
        .replace("InferiorFrontooccipital", "IFOF")
        .replace("AnteriorVerticalOccipital", "aVOF")
        .replace("PosteriorVerticalOccipital", "pVOF")
        .replace("PosteriorArcuate", "pArc")
        .replace("Arcuate", "AF")
        .replace("SuperiorLongitudinalI", "SLFI")
        .replace("SuperiorLongitudinalII", "SLFII")
        .replace("SuperiorLongitudinalI", "SLFIII")
        .replace("OpticRadiation", "OR")
        for t in tract_order
    ]

    # --------------------------------------
    # Organize MSE in table
    # --------------------------------------
    n_tracts = len(tract_order)
    hemi_labels = ["L", "R"]
    rows = []

    for h in range(2):     # left/right hemispheres
        for t in range(n_tracts):
            for s, participant in enumerate(participants):
                gp = "EB" if "EB" in participant else "NS"
                # pearson = rs[:, s, h]

                rows.append({
                    "Subject": s,
                    "Tract": tract_order[t],
                    "Hemisphere": hemi_labels[h],
                    "Group": gp,
                    "Beta": betas[t,s,h] #r_all[s, h] #
                })

    df = pd.DataFrame(rows)

    # ------------------------------------------------
    # Compute SEM per Group × Hemisphere
    # ------------------------------------------------
    sem_df = (
        df.groupby(["Group", "Tract", "Hemisphere"])["Beta"]
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

    # text_plotted = False  # Flag to ensure we only plot the label once

    #  plt.rcParams.update({
    #     'font.weight': 'bold',
    #     'axes.labelweight': 'bold'
    # })

    palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

    fig, axes = plt.subplots(
        2, 1,
        figsize=(18,18),
        sharey=True
    )

    group_offset = {"EB": -0.2, "NS": 0.2}

    for ax, hemi in zip(axes, hemi_labels):

        df_h = df[df["Hemisphere"] == hemi]
        sem_h = sem_df[sem_df["Hemisphere"] == hemi]


        # ---------------------------------------------------
        # Bar plot (mean ± SE)
        # ---------------------------------------------------

        sns.barplot(
            data=df_h,
            x="Tract",
            y="Beta",
            hue="Group",
            palette=palette,
            hue_order=["EB","NS"],
            errorbar="se",
            capsize=0.08,
            alpha=0.65,
            edgecolor="black",
            linewidth=2,
            ax=ax
        )

        # ---------------------------------------------------
        # Participant dots
        # ---------------------------------------------------

        sns.stripplot(
            data=df_h,
            x="Tract",
            y="Beta",
            hue="Group",
            hue_order=["EB","NS"],
            dodge=True,
            jitter=0.12,
            size=6,
            palette=palette,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            legend=False,
            ax=ax
        )

        # ---------------------------------------------------
        # Formatting
        # ---------------------------------------------------

        ax.set_ylim(-0.1,0.1)

        ax.set_title(
            f"{hemi}-Hemisphere",
            fontsize=22,
            fontweight="bold",
            pad=15
        )

        ax.set_xlabel(
            "Tract",
            fontsize=18,
            fontweight="bold"
        )

        ax.axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1.5
        )

        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=16,
            width=2.5
        )

        ax.set_xticks(np.arange(len(tract_order)))
        ax.set_xticklabels(
            tract_labels,
            rotation=35,
            ha="right",
            fontsize=16,
            fontweight="bold"
        )

        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        # ---------------------------------------------------
        # Manual legend
        # ---------------------------------------------------

        patch_eb = Patch(
            color=palette["EB"],
            alpha=0.65,
            label="EB"
        )

        patch_ns = Patch(
            color=palette["NS"],
            alpha=0.65,
            label="NS"
        )


        leg = ax.legend(
            handles=[patch_eb, patch_ns],
            title="Group",
            loc="upper right",
            frameon=False,
            fontsize=14
        )

        plt.setp(leg.get_texts(), fontweight="bold")
        plt.setp(leg.get_title(), fontweight="bold")


    axes[0].set_ylabel(
        "Beta",
        fontsize=18,
        fontweight="bold"
    )

    axes[1].set_ylabel(
        "Beta",
        fontsize=18,
        fontweight="bold"
    )

    sns.despine()
    plt.tight_layout()

    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)

    plt.savefig(
        op.join(saveDir,
        savefilename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
