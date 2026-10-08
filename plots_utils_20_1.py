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
from scipy.stats import sem

def pearsonr_plot(rs, reliability, participants, tract_order, bids_path, savefilename):
    #First compute noise ceiling 95% CI
    hemi_labels = ["L", "R"]
    n_tracts = len(tract_order)
    nc_rows = []

    for h in range(2):     # left/right hemispheres
        for t in range(n_tracts):
            for s, participant in enumerate(participants):
                gp = "EB" if "EB" in participant else "NS"

                nc_rows.append({
                    "Tract": tract_order[t],
                    "Subject": s,
                    "Hemisphere": hemi_labels[h],
                    "Group": gp,
                    "Correlation": reliability[s,t,h] #r_all[s, h] #
                })

    nc_df = pd.DataFrame(nc_rows)
    nc_sem_df = (
        nc_df.groupby(["Group","Tract", "Hemisphere"])["Correlation"]
        .agg(["mean", sem])
        .reset_index()
        .rename(columns={"mean": "Mean", "sem": "SEM"})
    )

    tract_labels = [
        t.replace("InferiorLongitudinal", "ILF")
        .replace("InferiorFrontooccipital", "IFOF")
        .replace("AnteriorVerticalOccipital", "aVOF")
        .replace("VerticalOccipital", "VOF")
        .replace("PosteriorVerticalOccipital", "pVOF")
        .replace("PosteriorArcuate", "pArc")
        .replace("Arcuate", "AF")
        .replace("SuperiorLongitudinal", "SLF")
        .replace("SuperiorLongitudinalI", "SLFI")
        .replace("SuperiorLongitudinalII", "SLFII")
        .replace("SuperiorLongitudinalI", "SLFIII")
        .replace("OpticRadiation", "OR")
        .replace("AnteriorThalamic", "ATR")
        .replace("CingulumCingulate", "CC")
        .replace("Corticospinal", "CST")
        for t in tract_order
    ]

    nc_sem_df["CI95_upper"] = nc_sem_df["Mean"] + 1.96 * nc_sem_df["SEM"]
    nc_sem_df["CI95_lower"] = nc_sem_df["Mean"] - 1.96 * nc_sem_df["SEM"]

    # --------------------------------------
    # Organize Pearson's r in table
    # --------------------------------------

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
                    "Correlation": rs[s,t,h] #r_all[s, h] #
                })

    df = pd.DataFrame(rows)

    # ------------------------------------------------
    # Compute SEM per Group × Hemisphere
    # ------------------------------------------------
    sem_df = (
        df.groupby(["Group", "Tract", "Hemisphere"])["Correlation"]
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
        nc_h = nc_sem_df[nc_sem_df["Hemisphere"] == hemi]

        # ---------------------------------------------------
        # Bar plot (mean ± SE)
        # ---------------------------------------------------

        sns.barplot(
            data=df_h,
            x="Tract",
            y="Correlation",
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
            y="Correlation",
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
        # Noise ceiling
        # ---------------------------------------------------

        for _, row in nc_h.iterrows():

            x_center = (
                tract_order.index(row["Tract"])
                + group_offset[row["Group"]]
            )

            ax.fill_between(
                [x_center-0.18, x_center+0.18],
                row["CI95_lower"],
                row["CI95_upper"],
                color="lightgray",
                alpha=0.35,
                zorder=0
            )

            ax.hlines(
                y=[row["CI95_lower"],
                row["CI95_upper"],
                row["Mean"]],
                xmin=x_center-0.18,
                xmax=x_center+0.18,
                colors="gray",
                linestyles=["--","--",":"],
                linewidth=2
            )

        # ---------------------------------------------------
        # Formatting
        # ---------------------------------------------------

        ax.set_ylim(-0.2,1)

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

        noise_patch = Patch(
            facecolor="lightgray",
            alpha=0.35,
            label="95% noise ceiling CI"
        )

        leg = ax.legend(
            handles=[patch_eb, patch_ns, noise_patch],
            title="Group",
            loc="upper right",
            frameon=False,
            fontsize=14
        )

        plt.setp(leg.get_texts(), fontweight="bold")
        plt.setp(leg.get_title(), fontweight="bold")


    axes[0].set_ylabel(
        "Pearson's r",
        fontsize=18,
        fontweight="bold"
    )

    axes[1].set_ylabel(
        "Pearson's r",
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


def mse_plot(mses, participants, tract_order, bids_path, savefilename):

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
                    "MSE": mses[s,t,h] #r_all[s, h] #
                })

    df = pd.DataFrame(rows)

    # ------------------------------------------------
    # Compute SEM per Group × Hemisphere
    # ------------------------------------------------
    sem_df = (
        df.groupby(["Group", "Tract", "Hemisphere"])["MSE"]
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
            y="MSE",
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
            y="MSE",
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

        ax.set_ylim(-0.2,1)

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
        "MSE",
        fontsize=18,
        fontweight="bold"
    )

    axes[1].set_ylabel(
        "MSE",
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
                    "Beta": betas[s,t,h] #r_all[s, h] #
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


def intercept_plot(intercepts, participants, tract_order, bids_path, savefilename):

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
                    "Intercept": intercepts[s,t,h] #r_all[s, h] #
                })

    df = pd.DataFrame(rows)

    # ------------------------------------------------
    # Compute SEM per Group × Hemisphere
    # ------------------------------------------------
    sem_df = (
        df.groupby(["Group", "Tract", "Hemisphere"])["Intercept"]
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
            y="Intercept",
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
            y="Intercept",
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

        ax.set_ylim(-0.00000001,0.00000001)

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
        "Intercept",
        fontsize=18,
        fontweight="bold"
    )

    axes[1].set_ylabel(
        "Intercept",
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

def rsquared_plot(rsquared, participants, tract_order, bids_path, savefilename):

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
                    "Rsquared": rsquared[s,t,h] #r_all[s, h] #
                })

    df = pd.DataFrame(rows)

    # ------------------------------------------------
    # Compute SEM per Group × Hemisphere
    # ------------------------------------------------
    sem_df = (
        df.groupby(["Group", "Tract", "Hemisphere"])["Rsquared"]
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
            y="Rsquared",
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
            y="Rsquared",
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
        "R^2",
        fontsize=18,
        fontweight="bold"
    )

    axes[1].set_ylabel(
        "R^2",
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



################################# 2. Data Plots ##########################################
from nilearn import plotting
import nibabel as nib
from nibabel.freesurfer import read_label
import matplotlib.pyplot as plt
#--------------------------
#Save Plot of predicted contrast map for each subject
#--------------------------
def pred_func_plot(predicted_maps, participants, tract_order, bids_path, img_out_dir):
    hemis = ["L", "R"]
    fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    for h, hemi in enumerate(hemis):
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
        n_vertices  = len(surf_roi)
        wang_hmt_vertices = np.where(surf_roi > 0)[0]

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

            for t, tract in enumerate(tract_order):
                # ----------------------------
                # Predicted Map visualization
                # ----------------------------
                # img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "wb_participants_loso", "surface_pngs", participant)
                os.makedirs(img_out_dir, exist_ok=True)

                vmin, vmax = -1.0, 1.0

                # Build full-surface vector 
                surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
                surf_map[wang_hmt_vertices] = predicted_maps[hemi][s, :, t]

                # Output filename (no run index)
                out_png = op.join(
                    img_out_dir,
                    f"{participant}_hemi-{hemi}_label-{tract}_desc-pred-motionXstationary_inflated.png"
                )

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

#========================================
#Save plot of average predicted contrast map for each group
#========================================
def avg_pred_func_plot(predicted_maps, participants, tract_order, bids_path, img_out_dir):
    fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    hemis = ["L", "R"]
    groups = ["EB", "NS"]
    for h, hemi in enumerate(hemis):
        hemi_fs = "lh" if hemi == "L" else "rh"
        for g, group in enumerate(groups):
            # ----------------------------
            # Load Wang MT ROI 
            # ----------------------------
            wang_hmt_path = op.join(
                '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
                f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
            )
            surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
            n_vertices  = len(surf_roi)
            wang_hmt_vertices = np.where(surf_roi > 0)[0]
            print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")
        
            #-----------------------------
            # Load func MT ROI
            #-----------------------------
            label_file = op.join(
                bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi',
                'group_averages',
                f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
            )

            func_mt_vertices = read_label(label_file)

            func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
            func_mt_roi[func_mt_vertices] = 1

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

            # -------------------------------------------------
            # Compute average functional map across runs FIRST
            # -------------------------------------------------

            for t, tract in enumerate(tract_order):
                # ----------------------------
                # Functional Contrast Map visualization
                # ----------------------------
                # img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "wb_participants_linearreg",  "surface_pngs", "mean_predicted_func")
                os.makedirs(img_out_dir, exist_ok=True)

                vmin, vmax = -.05, 0.05

                # mean_vals = np.nanmean(sub_mean_maps, axis=0)
                hemi_maps = predicted_maps[hemi][:, :, t]
                mean_vals = np.nanmean(hemi_maps[[group in p for p in participants],:], axis = 0)
                # Build full-surface vector once
                surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
                surf_map[wang_hmt_vertices] = mean_vals

                # Output filename (no run index)
                out_png = op.join(
                    img_out_dir,
                    f"{group}-mean_hemi-{hemi}_label-{tract}_desc-pred-motionXstationary_mean_inflated.png"
                )

                # -------------------------------------------------
                # Plot average participant map
                # -------------------------------------------------
                infl_surf = op.join(fs_path, "fsaverage", "surf", f"{'lh' if hemi == 'L' else 'rh'}.inflated")
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


#========================================
#Save plot of average density map for each group
#========================================
def avg_density_plot(density_maps, participants, tract_order, bids_path, img_out_dir, ylim):
    fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    hemis = ["L", "R"]
    groups = ["EB", "NS"]
    for h, hemi in enumerate(hemis):
        hemi_fs = "lh" if hemi == "L" else "rh"
        for g, group in enumerate(groups):
            # ----------------------------
            # Load Wang MT ROI 
            # ----------------------------
            wang_hmt_path = op.join(
                '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
                f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
            )
            surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
            n_vertices  = len(surf_roi)
            wang_hmt_vertices = np.where(surf_roi > 0)[0]
            print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")
        
            #-----------------------------
            # Load func MT ROI
            #-----------------------------
            label_file = op.join(
                bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi',
                'group_averages',
                f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
            )

            func_mt_vertices = read_label(label_file)

            func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
            func_mt_roi[func_mt_vertices] = 1

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

            for t, tract in enumerate(tract_order):
                # ----------------------------
                # Density Map visualization
                # ----------------------------
                # img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "wb_participants_linearreg",  "surface_pngs", "mean_predicted_func")
                os.makedirs(img_out_dir, exist_ok=True)

                [vmin, vmax] = ylim 

                # mean_vals = np.nanmean(sub_mean_maps, axis=0)
                hemi_maps = np.array([density_maps[hemi][sub][t] for sub in density_maps[hemi]]) #density_maps[hemi][:][t]
                mean_vals = np.nanmean(hemi_maps[[group in p for p in participants],:], axis = 0)
                # Build full-surface vector once
                surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
                surf_map[wang_hmt_vertices] = mean_vals

                # Output filename (no run index)
                out_png = op.join(
                    img_out_dir,
                    f"{group}-mean_hemi-{hemi}_label-{tract}_desc-density_mean_inflated.png"
                )

                # -------------------------------------------------
                # Plot average participant map
                # -------------------------------------------------
                infl_surf = op.join(fs_path, "fsaverage", "surf", f"{'lh' if hemi == 'L' else 'rh'}.inflated")
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
#========================================
#Save plot of average functional map for each group
#========================================
def avg_func_plot(func_maps, participants, bids_path, img_out_dir):
    fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    hemis = ["L", "R"]
    groups = ["EB", "NS"]
    for h, hemi in enumerate(hemis):
        hemi_fs = "lh" if hemi == "L" else "rh"
        for g, group in enumerate(groups):
            # ----------------------------
            # Load Wang MT ROI 
            # ----------------------------
            wang_hmt_path = op.join(
                '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
                f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
            )
            surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
            n_vertices  = len(surf_roi)
            wang_hmt_vertices = np.where(surf_roi > 0)[0]
            print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")
        
            #-----------------------------
            # Load func MT ROI
            #-----------------------------
            label_file = op.join(
                bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi',
                'group_averages',
                f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
            )

            func_mt_vertices = read_label(label_file)

            func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
            func_mt_roi[func_mt_vertices] = 1

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
            # Density Map visualization
            # ----------------------------
            # img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "wb_participants_linearreg",  "surface_pngs", "mean_predicted_func")
            os.makedirs(img_out_dir, exist_ok=True)

            vmin, vmax = -1, 1 

            # mean_vals = np.nanmean(sub_mean_maps, axis=0)
            hemi_maps = np.array([func_maps[hemi][sub,:] for sub in range(len(func_maps[hemi]))]) #density_maps[hemi][:][t]
            mean_vals = np.nanmean(hemi_maps[[group in p for p in participants],:], axis = 0)
            # Build full-surface vector once 
            surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
            surf_map[wang_hmt_vertices] = mean_vals

            # Output filename (no run index)
            out_png = op.join(
                img_out_dir,
                f"{group}-mean_hemi-{hemi}_desc-functional_mean_inflated.png"
            )

            # -------------------------------------------------
            # Plot average participant map
            # -------------------------------------------------
            infl_surf = op.join(fs_path, "fsaverage", "surf", f"{'lh' if hemi == 'L' else 'rh'}.inflated")
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


# #-----------------------------------------------
# # Plot average true functional map
# #-----------------------------------------------
# from nilearn import plotting
# from nibabel.freesurfer import read_label
# import matplotlib.pyplot as plt

# fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
# groups = ["EB", "NS"]
# hemis = ["L", "R"]
# for h, hemi in enumerate(hemis):
#     hemi_fs = "lh" if hemi == "L" else "rh"
#     infl_surf = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.inflated")
#     # ----------------------------
#     # Load curvature map (sulci/gyri)
#     # ----------------------------
#     curv_file = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv")
#     curv = nib.freesurfer.read_morph_data(curv_file)

#     # normalize curvature for nicer background display
#     curv_norm = (curv - np.percentile(curv, 5)) / (
#         np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8
#     )
#     curv_norm = np.clip(curv_norm, 0, 1)

#         # ----------------------------
#     # Load Wang MT ROI
#     # ----------------------------
#     wang_hmt_path = op.join(
#         '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
#         f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
#     )
#     surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
#     wang_hmt_vertices = np.where(surf_roi > 0)[0]

#     #---------------------
#     #Prepare func maps
#     #---------------------
#     n_target_runs = 3
#     all_C = []              # will store (n_subj, 3, n_vertices)
#     rnd_run_idx = np.full((n_subj, n_target_runs, len(hemis)), np.nan)
#     for s, participant in enumerate(participants):

#         C_full = norm_contrast_data[hemi][s][contrast]   # (n_runs_available, n_vertices)
#         n_runs_available = C_full.shape[0]

#         # ---- choose runs ----
#         if "NS" in participant:
#             run_idx = np.arange(3)
#         elif "EB" in participant:
#             run_idx = np.random.choice(
#                 n_runs_available, size=3, replace=False
#             )

#         rnd_run_idx[s, :, h] = run_idx

#         # ---- extract runs ----
#         C_sel = C_full[run_idx, :]     # (3, n_vertices)
#         all_C.append(C_sel)
#     all_C = np.stack(all_C, axis=0)
#     C_mean = all_C.mean(axis=1)


#     for g, group in enumerate(groups):
#         infl_surf = op.join(fs_path, "fsaverage", "surf", f"{'lh' if hemi == 'L' else 'rh'}.inflated")
#         _, _, n_vertices  = density_data[hemi].shape
#         # #-----------------------------
#         # # Load func MT ROI
#         # #-----------------------------
#         # label_file = op.join(
#         #     bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi',
#         #     'group_averages',
#         #     f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
#         # )

#         # func_mt_vertices = read_label(label_file)

#         # func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
#         # func_mt_roi[func_mt_vertices] = 1
#         # ----------------------------
#         # Load Wang MT ROI 
#         # ----------------------------
#         wang_hmt_path = op.join(
#             '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
#             f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
#         )
#         surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
#         wang_hmt_vertices = np.where(surf_roi > 0)[0]
#         print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")
#         # ----------------------------
#         # Load curvature map (sulci/gyri)
#         # ----------------------------
#         hemi_fs = "lh" if hemi == "L" else "rh"
#         curv_file = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv")
#         curv = nib.freesurfer.read_morph_data(curv_file)

#         # normalize curvature for nicer background display
#         curv_norm = (curv - np.percentile(curv, 5)) / (
#             np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8
#         )
#         curv_norm = np.clip(curv_norm, 0, 1)

#         # ----------------------------
#         # Functional Contrast Map visualization
#         # ----------------------------
#         img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "wb_participants_linearreg",  "surface_pngs", "mean")
#         os.makedirs(img_out_dir, exist_ok=True)

#         vmin, vmax = -0.5, 0.5
#         # -------------------------------------------------
#         # Compute average functional map across runs FIRST
#         # -------------------------------------------------
#         hemi_maps = C_mean #predicted_maps[hemi]
#         mean_vals = np.nanmean(hemi_maps[[group in p for p in participants],:], axis = 0)
#         # mean_vals = np.nanmean(sub_mean_maps, axis=0)

#         # Build full-surface vector once
#         surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
#         surf_map[wang_hmt_vertices] = mean_vals

#         # Output filename (no run index)
#         out_png = op.join(
#             img_out_dir,
#             f"{group}-mean_hemi-{hemi}_desc-true-motionXstationary_mean_inflated.png"
#         )

#         # -------------------------------------------------
#         # Plot average participant map
#         # -------------------------------------------------

#         display = plotting.plot_surf_stat_map(
#             surf_mesh=infl_surf,
#             stat_map=surf_map,
#             hemi="left" if hemi == "L" else "right",
#             view="lateral",
#             cmap="plasma",
#             colorbar=True,
#             vmin=vmin,
#             vmax=vmax,
#             threshold=None,
#             bg_map=curv_norm,
#             bg_on_data=True,
#             darkness=0.6,
#         )
#         # ---- MT boundary overlay ----
#         # plotting.plot_surf_contours(
#         #     surf_mesh=infl_surf,
#         #     roi_map=func_mt_roi,
#         #     levels=[1],
#         #     colors=["lightgray"],
#         #     linewidths=2.0,
#         #     figure=display.figure,
#         #     axes=display.axes[0]
#         # )

#         # ---- save + close ----
#         display.savefig(out_png, dpi=300)
#         plt.close(display.figure)

