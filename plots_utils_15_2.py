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
                    "Subject": s,
                    "Hemisphere": hemi_labels[h],
                    "Group": gp,
                    "Correlation": reliability[s, h] #r_all[s, h] #
                })

    nc_df = pd.DataFrame(nc_rows)
    nc_sem_df = (
        nc_df.groupby(["Group", "Hemisphere"])["Correlation"]
        .agg(["mean", "sem"])
        .reset_index()
        .rename(columns={"mean": "Mean", "sem": "SEM"})
    )


    nc_sem_df["CI95_upper"] = nc_sem_df["Mean"] + 1.96 * nc_sem_df["SEM"]
    nc_sem_df["CI95_lower"] = nc_sem_df["Mean"] - 1.96 * nc_sem_df["SEM"]

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

    # text_plotted = False  # Flag to ensure we only plot the label once

    #  plt.rcParams.update({
    #     'font.weight': 'bold',
    #     'axes.labelweight': 'bold'
    # })

    palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

    fig, axes = plt.subplots(
        1, 2,
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
            x="Group",
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
            x="Group",
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

        # Seaborn categorical positions:
        # EB = 0
        # NS = 1
        group_positions = {"EB": 0, "NS": 1}

        for _, row in nc_h.iterrows():

            x_center = group_positions[row["Group"]]
            half_width = 0.35

            # 95% CI shaded region
            ax.fill_between(
                [x_center - half_width, x_center + half_width],
                row["CI95_lower"],
                row["CI95_upper"],
                color="lightgray",
                alpha=0.35,
                zorder=0
            )

            # Lower CI
            ax.hlines(
                row["CI95_lower"],
                xmin=x_center - half_width,
                xmax=x_center + half_width,
                colors="gray",
                linestyles="--",
                linewidth=2,
                zorder=1
            )

            # Upper CI
            ax.hlines(
                row["CI95_upper"],
                xmin=x_center - half_width,
                xmax=x_center + half_width,
                colors="gray",
                linestyles="--",
                linewidth=2,
                zorder=1
            )

            # Mean noise ceiling
            ax.hlines(
                row["Mean"],
                xmin=x_center - half_width,
                xmax=x_center + half_width,
                colors="gray",
                linestyles=":",
                linewidth=2,
                zorder=1
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
            "Group",
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

        ax.set_xticks(np.arange(len(["EB", "NS"])))
        ax.set_xticklabels(
            ["EB", "NS"],
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


def dmse_plot(mses, participants, tract_order, bids_path, savefilename, ylim):

    tract_labels = [
        t.replace("MTxLGNxPU", "Thalamocortical")
        .replace("MTxPTxSTS1", "Temporal")
        .replace("MTxFEF", "Frontal")
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
                    "dMSE": mses[t, s, h] #r_all[s, h] #
                })

    df = pd.DataFrame(rows)

    # ------------------------------------------------
    # Compute SEM per Group × Hemisphere
    # ------------------------------------------------
    sem_df = (
        df.groupby(["Group", "Tract", "Hemisphere"])["dMSE"]
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
            y="dMSE",
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
            y="dMSE",
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

        ax.set_ylim(ylim)

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
        "dMSE",
        fontsize=18,
        fontweight="bold"
    )

    axes[1].set_ylabel(
        "dMSE",
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
        t.replace("MTxLGNxPU", "Thalamocortical")
        .replace("MTxPTxSTS1", "Temporal")
        .replace("MTxFEF", "Frontal")
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

        ax.set_ylim(-0.4,0.4)

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


# def intercept_plot(intercepts, participants, tract_order, bids_path, savefilename):

#     tract_labels = [
#         t.replace("InferiorLongitudinal", "ILF")
#         .replace("InferiorFrontooccipital", "IFOF")
#         .replace("AnteriorVerticalOccipital", "aVOF")
#         .replace("PosteriorVerticalOccipital", "pVOF")
#         .replace("PosteriorArcuate", "pArc")
#         .replace("Arcuate", "AF")
#         .replace("SuperiorLongitudinalI", "SLFI")
#         .replace("SuperiorLongitudinalII", "SLFII")
#         .replace("SuperiorLongitudinalI", "SLFIII")
#         .replace("OpticRadiation", "OR")
#         for t in tract_order
#     ]

#     # --------------------------------------
#     # Organize MSE in table
#     # --------------------------------------
#     n_tracts = len(tract_order)
#     hemi_labels = ["L", "R"]
#     rows = []

#     for h in range(2):     # left/right hemispheres
#         for t in range(n_tracts):
#             for s, participant in enumerate(participants):
#                 gp = "EB" if "EB" in participant else "NS"
#                 # pearson = rs[:, s, h]

#                 rows.append({
#                     "Subject": s,
#                     "Tract": tract_order[t],
#                     "Hemisphere": hemi_labels[h],
#                     "Group": gp,
#                     "Intercept": intercepts[s,t,h] #r_all[s, h] #
#                 })

#     df = pd.DataFrame(rows)

#     # ------------------------------------------------
#     # Compute SEM per Group × Hemisphere
#     # ------------------------------------------------
#     sem_df = (
#         df.groupby(["Group", "Tract", "Hemisphere"])["Intercept"]
#         .agg(["mean", sem])
#         .reset_index()
#         .rename(columns={"mean": "Mean", "sem": "SEM"})
#     )

#     # ------------------------------------------------
#     # Color palette: EB = blue, NS = orange
#     # ------------------------------------------------
#     palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

#     # ------------------------------------------------
#     # Create 2 subplots — one per hemisphere
#     # ------------------------------------------------

#     # text_plotted = False  # Flag to ensure we only plot the label once

#     #  plt.rcParams.update({
#     #     'font.weight': 'bold',
#     #     'axes.labelweight': 'bold'
#     # })

#     palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

#     fig, axes = plt.subplots(
#         2, 1,
#         figsize=(18,18),
#         sharey=True
#     )

#     group_offset = {"EB": -0.2, "NS": 0.2}

#     for ax, hemi in zip(axes, hemi_labels):

#         df_h = df[df["Hemisphere"] == hemi]
#         sem_h = sem_df[sem_df["Hemisphere"] == hemi]


#         # ---------------------------------------------------
#         # Bar plot (mean ± SE)
#         # ---------------------------------------------------

#         sns.barplot(
#             data=df_h,
#             x="Tract",
#             y="Intercept",
#             hue="Group",
#             palette=palette,
#             hue_order=["EB","NS"],
#             errorbar="se",
#             capsize=0.08,
#             alpha=0.65,
#             edgecolor="black",
#             linewidth=2,
#             ax=ax
#         )

#         # ---------------------------------------------------
#         # Participant dots
#         # ---------------------------------------------------

#         sns.stripplot(
#             data=df_h,
#             x="Tract",
#             y="Intercept",
#             hue="Group",
#             hue_order=["EB","NS"],
#             dodge=True,
#             jitter=0.12,
#             size=6,
#             palette=palette,
#             edgecolor="black",
#             linewidth=0.8,
#             alpha=0.9,
#             legend=False,
#             ax=ax
#         )

#         # ---------------------------------------------------
#         # Formatting
#         # ---------------------------------------------------

#         ax.set_ylim(-0.00000001,0.00000001)

#         ax.set_title(
#             f"{hemi}-Hemisphere",
#             fontsize=22,
#             fontweight="bold",
#             pad=15
#         )

#         ax.set_xlabel(
#             "Tract",
#             fontsize=18,
#             fontweight="bold"
#         )

#         ax.axhline(
#             0,
#             color="gray",
#             linestyle="--",
#             linewidth=1.5
#         )

#         ax.spines["left"].set_linewidth(2.5)
#         ax.spines["bottom"].set_linewidth(2.5)

#         ax.tick_params(
#             axis='both',
#             which='major',
#             labelsize=16,
#             width=2.5
#         )

#         ax.set_xticks(np.arange(len(tract_order)))
#         ax.set_xticklabels(
#             tract_labels,
#             rotation=35,
#             ha="right",
#             fontsize=16,
#             fontweight="bold"
#         )

#         for label in ax.get_yticklabels():
#             label.set_fontweight("bold")

#         # ---------------------------------------------------
#         # Manual legend
#         # ---------------------------------------------------

#         patch_eb = Patch(
#             color=palette["EB"],
#             alpha=0.65,
#             label="EB"
#         )

#         patch_ns = Patch(
#             color=palette["NS"],
#             alpha=0.65,
#             label="NS"
#         )


#         leg = ax.legend(
#             handles=[patch_eb, patch_ns],
#             title="Group",
#             loc="upper right",
#             frameon=False,
#             fontsize=14
#         )

#         plt.setp(leg.get_texts(), fontweight="bold")
#         plt.setp(leg.get_title(), fontweight="bold")


#     axes[0].set_ylabel(
#         "Intercept",
#         fontsize=18,
#         fontweight="bold"
#     )

#     axes[1].set_ylabel(
#         "Intercept",
#         fontsize=18,
#         fontweight="bold"
#     )

#     sns.despine()
#     plt.tight_layout()

#     saveDir = op.join(bids_path, "analysis", "plots")
#     os.makedirs(saveDir, exist_ok=True)

#     plt.savefig(
#         op.join(saveDir,
#         savefilename),
#         dpi=300,
#         bbox_inches="tight"
#     )

#     plt.show()

# def rsquared_plot(rsquared, participants, tract_order, bids_path, savefilename):

#     tract_labels = [
#         t.replace("InferiorLongitudinal", "ILF")
#         .replace("InferiorFrontooccipital", "IFOF")
#         .replace("AnteriorVerticalOccipital", "aVOF")
#         .replace("PosteriorVerticalOccipital", "pVOF")
#         .replace("PosteriorArcuate", "pArc")
#         .replace("Arcuate", "AF")
#         .replace("SuperiorLongitudinalI", "SLFI")
#         .replace("SuperiorLongitudinalII", "SLFII")
#         .replace("SuperiorLongitudinalI", "SLFIII")
#         .replace("OpticRadiation", "OR")
#         for t in tract_order
#     ]

#     # --------------------------------------
#     # Organize MSE in table
#     # --------------------------------------
#     n_tracts = len(tract_order)
#     hemi_labels = ["L", "R"]
#     rows = []

#     for h in range(2):     # left/right hemispheres
#         for t in range(n_tracts):
#             for s, participant in enumerate(participants):
#                 gp = "EB" if "EB" in participant else "NS"
#                 # pearson = rs[:, s, h]

#                 rows.append({
#                     "Subject": s,
#                     "Tract": tract_order[t],
#                     "Hemisphere": hemi_labels[h],
#                     "Group": gp,
#                     "Rsquared": rsquared[s,t,h] #r_all[s, h] #
#                 })

#     df = pd.DataFrame(rows)

#     # ------------------------------------------------
#     # Compute SEM per Group × Hemisphere
#     # ------------------------------------------------
#     sem_df = (
#         df.groupby(["Group", "Tract", "Hemisphere"])["Rsquared"]
#         .agg(["mean", sem])
#         .reset_index()
#         .rename(columns={"mean": "Mean", "sem": "SEM"})
#     )

#     # ------------------------------------------------
#     # Color palette: EB = blue, NS = orange
#     # ------------------------------------------------
#     palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

#     # ------------------------------------------------
#     # Create 2 subplots — one per hemisphere
#     # ------------------------------------------------

#     # text_plotted = False  # Flag to ensure we only plot the label once

#     #  plt.rcParams.update({
#     #     'font.weight': 'bold',
#     #     'axes.labelweight': 'bold'
#     # })

#     palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

#     fig, axes = plt.subplots(
#         2, 1,
#         figsize=(18,18),
#         sharey=True
#     )

#     group_offset = {"EB": -0.2, "NS": 0.2}

#     for ax, hemi in zip(axes, hemi_labels):

#         df_h = df[df["Hemisphere"] == hemi]
#         sem_h = sem_df[sem_df["Hemisphere"] == hemi]


#         # ---------------------------------------------------
#         # Bar plot (mean ± SE)
#         # ---------------------------------------------------

#         sns.barplot(
#             data=df_h,
#             x="Tract",
#             y="Rsquared",
#             hue="Group",
#             palette=palette,
#             hue_order=["EB","NS"],
#             errorbar="se",
#             capsize=0.08,
#             alpha=0.65,
#             edgecolor="black",
#             linewidth=2,
#             ax=ax
#         )

#         # ---------------------------------------------------
#         # Participant dots
#         # ---------------------------------------------------

#         sns.stripplot(
#             data=df_h,
#             x="Tract",
#             y="Rsquared",
#             hue="Group",
#             hue_order=["EB","NS"],
#             dodge=True,
#             jitter=0.12,
#             size=6,
#             palette=palette,
#             edgecolor="black",
#             linewidth=0.8,
#             alpha=0.9,
#             legend=False,
#             ax=ax
#         )

#         # ---------------------------------------------------
#         # Formatting
#         # ---------------------------------------------------

#         ax.set_ylim(-0.1,0.1)

#         ax.set_title(
#             f"{hemi}-Hemisphere",
#             fontsize=22,
#             fontweight="bold",
#             pad=15
#         )

#         ax.set_xlabel(
#             "Tract",
#             fontsize=18,
#             fontweight="bold"
#         )

#         ax.axhline(
#             0,
#             color="gray",
#             linestyle="--",
#             linewidth=1.5
#         )

#         ax.spines["left"].set_linewidth(2.5)
#         ax.spines["bottom"].set_linewidth(2.5)

#         ax.tick_params(
#             axis='both',
#             which='major',
#             labelsize=16,
#             width=2.5
#         )

#         ax.set_xticks(np.arange(len(tract_order)))
#         ax.set_xticklabels(
#             tract_labels,
#             rotation=35,
#             ha="right",
#             fontsize=16,
#             fontweight="bold"
#         )

#         for label in ax.get_yticklabels():
#             label.set_fontweight("bold")

#         # ---------------------------------------------------
#         # Manual legend
#         # ---------------------------------------------------

#         patch_eb = Patch(
#             color=palette["EB"],
#             alpha=0.65,
#             label="EB"
#         )

#         patch_ns = Patch(
#             color=palette["NS"],
#             alpha=0.65,
#             label="NS"
#         )


#         leg = ax.legend(
#             handles=[patch_eb, patch_ns],
#             title="Group",
#             loc="upper right",
#             frameon=False,
#             fontsize=14
#         )

#         plt.setp(leg.get_texts(), fontweight="bold")
#         plt.setp(leg.get_title(), fontweight="bold")


#     axes[0].set_ylabel(
#         "R^2",
#         fontsize=18,
#         fontweight="bold"
#     )

#     axes[1].set_ylabel(
#         "R^2",
#         fontsize=18,
#         fontweight="bold"
#     )

#     sns.despine()
#     plt.tight_layout()

#     saveDir = op.join(bids_path, "analysis", "plots")
#     os.makedirs(saveDir, exist_ok=True)

#     plt.savefig(
#         op.join(saveDir,
#         savefilename),
#         dpi=300,
#         bbox_inches="tight"
#     )

#     plt.show()



################################# 2. Data Plots ##########################################
from nilearn import plotting
import nibabel as nib
from nibabel.freesurfer import read_label
import matplotlib.pyplot as plt
#--------------------------
#Save Plot of predicted contrast map for each subject
#--------------------------
# def pred_func_plot(predicted_maps, participants, tract_order, bids_path, img_out_dir):
#     hemis = ["L", "R"]
#     fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
#     for h, hemi in enumerate(hemis):
#         hemi_fs = "lh" if hemi == "L" else "rh"
#         infl_surf = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.inflated")
#         # ----------------------------
#         # Load curvature map (sulci/gyri)
#         # ----------------------------
#         curv_file = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv")
#         curv = nib.freesurfer.read_morph_data(curv_file)

#         # normalize curvature for nicer background display
#         curv_norm = (curv - np.percentile(curv, 5)) / (
#             np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8
#         )
#         curv_norm = np.clip(curv_norm, 0, 1)

#             # ----------------------------
#         # Load Wang MT ROI
#         # ----------------------------
#         wang_hmt_path = op.join(
#             '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
#             f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
#         )
#         surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
#         n_vertices  = len(surf_roi)
#         wang_hmt_vertices = np.where(surf_roi > 0)[0]

#         for s, participant in enumerate(participants):
#         # s =1
#         # participant= 'sub-EBxGxEYx1965'
#             # # ----------------------------
#             # Functional MT ROI (binary surface map)
#             # ----------------------------
#             label_file = op.join(bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi', participant,
#                 f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label")

#             func_mt_vertices = read_label(label_file)

#             func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
#             func_mt_roi[func_mt_vertices] = 1

#             for t, tract in enumerate(tract_order):
#                 # ----------------------------
#                 # Predicted Map visualization
#                 # ----------------------------
#                 # img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "wb_participants_loso", "surface_pngs", participant)
#                 os.makedirs(img_out_dir, exist_ok=True)

#                 vmin, vmax = -1.0, 1.0

#                 # Build full-surface vector 
#                 surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
#                 surf_map[wang_hmt_vertices] = predicted_maps[hemi][s, :, t]

#                 # Output filename (no run index)
#                 out_png = op.join(
#                     img_out_dir,
#                     f"{participant}_hemi-{hemi}_label-{tract}_desc-pred-motionXstationary_inflated.png"
#                 )

#                 # -------------------------------------------------
#                 # Plot only once
#                 # -------------------------------------------------

#                 display = plotting.plot_surf_stat_map(
#                     surf_mesh=infl_surf,
#                     stat_map=surf_map,
#                     hemi="left" if hemi == "L" else "right",
#                     view="lateral",
#                     cmap="plasma",
#                     colorbar=True,
#                     vmin=vmin,
#                     vmax=vmax,
#                     threshold=None,
#                     bg_map=curv_norm,
#                     bg_on_data=True,
#                     darkness=0.6,
#                 )

#                 # ---- MT boundary overlay ----
#                 plotting.plot_surf_contours(
#                     surf_mesh=infl_surf,
#                     roi_map=func_mt_roi,
#                     levels=[1],
#                     colors=["lightgray"],
#                     linewidths=2.0,
#                     figure=display.figure,
#                     axes=display.axes[0]
#                 )

#                 # ---- save + close ----
#                 display.savefig(out_png, dpi=300)
#                 plt.close(display.figure)

# #========================================
# #Save plot of average predicted contrast map for each group
# #========================================
# def avg_pred_func_plot(predicted_maps, participants, tract_order, bids_path, img_out_dir):
#     fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
#     hemis = ["L", "R"]
#     groups = ["EB", "NS"]
#     for h, hemi in enumerate(hemis):
#         hemi_fs = "lh" if hemi == "L" else "rh"
#         for g, group in enumerate(groups):
#             # ----------------------------
#             # Load Wang MT ROI 
#             # ----------------------------
#             wang_hmt_path = op.join(
#                 '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
#                 f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
#             )
#             surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
#             n_vertices  = len(surf_roi)
#             wang_hmt_vertices = np.where(surf_roi > 0)[0]
#             print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")
        
#             #-----------------------------
#             # Load func MT ROI
#             #-----------------------------
#             label_file = op.join(
#                 bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi',
#                 'group_averages',
#                 f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
#             )

#             func_mt_vertices = read_label(label_file)

#             func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
#             func_mt_roi[func_mt_vertices] = 1

#             # ----------------------------
#             # Load curvature map (sulci/gyri)
#             # ----------------------------
#             curv_file = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv")
#             curv = nib.freesurfer.read_morph_data(curv_file)

#             # normalize curvature for nicer background display
#             curv_norm = (curv - np.percentile(curv, 5)) / (
#                 np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8
#             )
#             curv_norm = np.clip(curv_norm, 0, 1)

#             # -------------------------------------------------
#             # Compute average functional map across runs FIRST
#             # -------------------------------------------------

#             for t, tract in enumerate(tract_order):
#                 # ----------------------------
#                 # Functional Contrast Map visualization
#                 # ----------------------------
#                 # img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "wb_participants_linearreg",  "surface_pngs", "mean_predicted_func")
#                 os.makedirs(img_out_dir, exist_ok=True)

#                 vmin, vmax = -.05, 0.05

#                 # mean_vals = np.nanmean(sub_mean_maps, axis=0)
#                 hemi_maps = predicted_maps[hemi][:, :, t]
#                 mean_vals = np.nanmean(hemi_maps[[group in p for p in participants],:], axis = 0)
#                 # Build full-surface vector once
#                 surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
#                 surf_map[wang_hmt_vertices] = mean_vals

#                 # Output filename (no run index)
#                 out_png = op.join(
#                     img_out_dir,
#                     f"{group}-mean_hemi-{hemi}_label-{tract}_desc-pred-motionXstationary_mean_inflated.png"
#                 )

#                 # -------------------------------------------------
#                 # Plot average participant map
#                 # -------------------------------------------------
#                 infl_surf = op.join(fs_path, "fsaverage", "surf", f"{'lh' if hemi == 'L' else 'rh'}.inflated")
#                 display = plotting.plot_surf_stat_map(
#                     surf_mesh=infl_surf,
#                     stat_map=surf_map,
#                     hemi="left" if hemi == "L" else "right",
#                     view="lateral",
#                     cmap="plasma",
#                     colorbar=True,
#                     vmin=vmin,
#                     vmax=vmax,
#                     threshold=None,
#                     bg_map=curv_norm,
#                     bg_on_data=True,
#                     darkness=0.6,
#                 )
#                 # ---- MT boundary overlay ----
#                 plotting.plot_surf_contours(
#                     surf_mesh=infl_surf,
#                     roi_map=func_mt_roi,
#                     levels=[1],
#                     colors=["lightgray"],
#                     linewidths=2.0,
#                     figure=display.figure,
#                     axes=display.axes[0]
#                 )

#                 # ---- save + close ----
#                 display.savefig(out_png, dpi=300)
#                 plt.close(display.figure)


#========================================
#Save plot of average density map for each group
#========================================
def avg_density_plot(density_maps, participants, tract_order, bids_path, img_out_dir):
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

                vmin, vmax = -2, 2 

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
# def avg_func_plot(func_maps, participants, bids_path, img_out_dir):
#     fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
#     hemis = ["L", "R"]
#     groups = ["EB", "NS"]
#     for h, hemi in enumerate(hemis):
#         hemi_fs = "lh" if hemi == "L" else "rh"
#         for g, group in enumerate(groups):
#             # ----------------------------
#             # Load Wang MT ROI 
#             # ----------------------------
#             wang_hmt_path = op.join(
#                 '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
#                 f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
#             )
#             surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
#             n_vertices  = len(surf_roi)
#             wang_hmt_vertices = np.where(surf_roi > 0)[0]
#             print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")
        
#             #-----------------------------
#             # Load func MT ROI
#             #-----------------------------
#             label_file = op.join(
#                 bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi',
#                 'group_averages',
#                 f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
#             )

#             func_mt_vertices = read_label(label_file)

#             func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
#             func_mt_roi[func_mt_vertices] = 1

#             # ----------------------------
#             # Load curvature map (sulci/gyri)
#             # ----------------------------
#             curv_file = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv")
#             curv = nib.freesurfer.read_morph_data(curv_file)

#             # normalize curvature for nicer background display
#             curv_norm = (curv - np.percentile(curv, 5)) / (
#                 np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8
#             )
#             curv_norm = np.clip(curv_norm, 0, 1)

#             # ----------------------------
#             # Density Map visualization
#             # ----------------------------
#             # img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "wb_participants_linearreg",  "surface_pngs", "mean_predicted_func")
#             os.makedirs(img_out_dir, exist_ok=True)

#             vmin, vmax = -1, 1 

#             # mean_vals = np.nanmean(sub_mean_maps, axis=0)
#             hemi_maps = np.array([func_maps[hemi][sub,:] for sub in range(len(func_maps[hemi]))]) #density_maps[hemi][:][t]
#             mean_vals = np.nanmean(hemi_maps[[group in p for p in participants],:], axis = 0)
#             # Build full-surface vector once 
#             surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
#             surf_map[wang_hmt_vertices] = mean_vals

#             # Output filename (no run index)
#             out_png = op.join(
#                 img_out_dir,
#                 f"{group}-mean_hemi-{hemi}_desc-functional_mean_inflated.png"
#             )

#             # -------------------------------------------------
#             # Plot average participant map
#             # -------------------------------------------------
#             infl_surf = op.join(fs_path, "fsaverage", "surf", f"{'lh' if hemi == 'L' else 'rh'}.inflated")
#             display = plotting.plot_surf_stat_map(
#                 surf_mesh=infl_surf,
#                 stat_map=surf_map,
#                 hemi="left" if hemi == "L" else "right",
#                 view="lateral",
#                 cmap="plasma",
#                 colorbar=True,
#                 vmin=vmin,
#                 vmax=vmax,
#                 threshold=None,
#                 bg_map=curv_norm,
#                 bg_on_data=True,
#                 darkness=0.6,
#             )
#             # ---- MT boundary overlay ----
#             plotting.plot_surf_contours(
#                 surf_mesh=infl_surf,
#                 roi_map=func_mt_roi,
#                 levels=[1],
#                 colors=["lightgray"],
#                 linewidths=2.0,
#                 figure=display.figure,
#                 axes=display.axes[0]
#             )

#             # ---- save + close ----
#             display.savefig(out_png, dpi=300)
#             plt.close(display.figure)

