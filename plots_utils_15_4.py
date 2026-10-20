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
    nc_rows = []

    for h in range(2):     # left/right hemispheres
        for t, tract in enumerate(tract_order):
            for s, participant in enumerate(participants):
                gp = "EB" if "EB" in participant else "NS"
                # EB --> ptlocal only
                # NS --> mtlocal + ptlocal
                if "EB" in participant:
                        localizers = ["ptlocal"]
                elif "NS" in participant:
                        localizers = ["mtlocal", "ptlocal"]
                for l, localizer in enumerate(localizers):

                    nc_rows.append({
                        "Tract": tract,
                        "Localizer": localizer,
                        "Subject": s,
                        "Hemisphere": hemi_labels[h],
                        "Group": gp,
                        "Correlation": reliability[s,t,l,h] #r_all[s, h] #
                    })

    nc_df = pd.DataFrame(nc_rows)
    nc_sem_df = (
        nc_df.groupby(["Group", "Tract", "Localizer", "Hemisphere"])["Correlation"]
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
        .replace("MTxLGNxPU", "Thalamocortical")
        .replace("MTxPTxSTS1", "Temporal")
        .replace("MTxFEF", "Frontal")
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
        for t, tract in enumerate(tract_order):
            for s, participant in enumerate(participants):
                gp = "EB" if "EB" in participant else "NS"
                # EB --> ptlocal only
                # NS --> mtlocal + ptlocal
                if "EB" in participant:
                        localizers = ["ptlocal"]
                elif "NS" in participant:
                        localizers = ["mtlocal", "ptlocal"]
                for l, localizer in enumerate(localizers):
                # pearson = rs[:, s, h]

                    rows.append({
                        "Subject": s,
                        "Tract": tract,
                        "Localizer": localizer,
                        "Hemisphere": hemi_labels[h],
                        "Group": gp,
                        "Correlation": rs[s,t,l,h] #r_all[s, h] #
                    })

    df = pd.DataFrame(rows)

    # ------------------------------------------------
    # Compute SEM per Group × Hemisphere
    # ------------------------------------------------
    sem_df = (
        df.groupby(["Group", "Tract","Localizer","Hemisphere"])["Correlation"]
        .agg(["mean", sem])
        .reset_index()
        .rename(columns={"mean": "Mean", "sem": "SEM"})
    )
    # ------------------------------------------------
    # Color palette: Group = color family,
    # Localizer = shade within the family
    # ------------------------------------------------

    palette = {
        ("EB", "ptlocal"): "#9ecae1",  # light blue 
        ("NS", "mtlocal"): "#c45100",  # dark orange 
        ("NS", "ptlocal"): "#fdae6b",  # light orange
    } #        ("EB", "mtlocal"): "#1f4e79",  # dark blue

    # Create a combined categorical variable for hue
    for data_df in [df, nc_df]:
        data_df["GroupLocalizer"] = (
            data_df["Group"] + "_" + data_df["Localizer"]
        )

    hue_order = [
        "EB_ptlocal",
        "NS_mtlocal",
        "NS_ptlocal",
    ] #        "EB_mtlocal",

    combined_palette = {
        f"{group}_{localizer}": color
        for (group, localizer), color in palette.items()
    }

    # ------------------------------------------------
    # Create 2 subplots — one per hemisphere
    # ------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(18, 18),
        sharey=True
    )

    for ax, hemi in zip(axes, hemi_labels):

        df_h = df[df["Hemisphere"] == hemi].copy()
        nc_h = nc_sem_df[nc_sem_df["Hemisphere"] == hemi].copy()

        df_h["GroupLocalizer"] = (
            df_h["Group"] + "_" + df_h["Localizer"]
        )

        # ------------------------------------------------
        # Bar plot: hue combines Group and Localizer
        # ------------------------------------------------

        sns.barplot(
            data=df_h,
            x="Tract",
            y="Correlation",
            hue="GroupLocalizer",
            hue_order=hue_order,
            palette=combined_palette,
            errorbar="se",
            capsize=0.06,
            alpha=0.75,
            edgecolor="black",
            linewidth=1.5,
            ax=ax
        )

        # ------------------------------------------------
        # Participant dots
        # ------------------------------------------------

        sns.stripplot(
            data=df_h,
            x="Tract",
            y="Correlation",
            hue="GroupLocalizer",
            hue_order=hue_order,
            palette=combined_palette,
            dodge=True,
            jitter=0.12,
            size=5,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            legend=False,
            ax=ax
        )

        # ------------------------------------------------
        # Noise ceiling: plot separately by Group × Localizer
        # ------------------------------------------------

        n_hue = len(hue_order)
        bar_width = 0.8 / n_hue

        for _, row in nc_h.iterrows():

            key = f'{row["Group"]}_{row["Localizer"]}'
            hue_idx = hue_order.index(key)

            x_center = (
                tract_order.index(row["Tract"])
                - 0.4
                + bar_width / 2
                + hue_idx * bar_width
            )

            ax.fill_between(
                [x_center - bar_width / 2, x_center + bar_width / 2],
                row["CI95_lower"],
                row["CI95_upper"],
                color=combined_palette[key],
                alpha=0.12,
                zorder=0
            )

            ax.hlines(
                [row["CI95_lower"], row["CI95_upper"]],
                xmin=x_center - bar_width / 2,
                xmax=x_center + bar_width / 2,
                colors=combined_palette[key],
                linestyles="--",
                linewidth=1.5
            )

            ax.hlines(
                row["Mean"],
                xmin=x_center - bar_width / 2,
                xmax=x_center + bar_width / 2,
                colors=combined_palette[key],
                linestyles=":",
                linewidth=2
            )

        # ------------------------------------------------
        # Formatting
        # ------------------------------------------------

        ax.set_ylim(-0.2, 1)

        ax.set_title(
            f"{hemi}-Hemisphere",
            fontsize=22,
            fontweight="bold",
            pad=15
        )

        ax.set_xlabel("Tract", fontsize=18, fontweight="bold")

        ax.axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1.5
        )

        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)

        ax.tick_params(
            axis="both",
            which="major",
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

        # ------------------------------------------------
        # Manual legend
        # ------------------------------------------------

        legend_handles = [
            Patch(
                facecolor=combined_palette[key],
                edgecolor="black",
                alpha=0.75,
                label=(
                    f"{key.split('_')[0]} — "
                    f"{key.split('_')[1]}"
                )
            )
            for key in hue_order
        ]

        legend_handles.append(
            Patch(
                facecolor="lightgray",
                alpha=0.35,
                label="95% noise ceiling CI"
            )
        )

        leg = ax.legend(
            handles=legend_handles,
            title="Group — Localizer",
            loc="upper right",
            frameon=False,
            fontsize=12
        )

        plt.setp(leg.get_texts(), fontweight="bold")
        plt.setp(leg.get_title(), fontweight="bold")


    axes[0].set_ylabel("Pearson's r", fontsize=18, fontweight="bold")
    axes[1].set_ylabel("Pearson's r", fontsize=18, fontweight="bold")

    sns.despine()
    plt.tight_layout()

    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)

    plt.savefig(
        op.join(saveDir, savefilename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def beta_plot(betas, participants, tract_order, bids_path, savefilename, ylim):

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
        .replace("MTxLGNxPU", "Thalamocortical")
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
                                # EB --> ptlocal only
                # NS --> mtlocal + ptlocal
                if "EB" in participant:
                        localizers = ["ptlocal"]
                elif "NS" in participant:
                        localizers = ["mtlocal", "ptlocal"]
                for l, localizer in enumerate(localizers):
                # pearson = rs[:, s, h]

                    rows.append({
                        "Subject": s,
                        "Localizer": localizer,
                        "Tract": tract_order[t],
                        "Hemisphere": hemi_labels[h],
                        "Group": gp,
                        "Beta": betas[s,t,l,h] #r_all[s, h] #
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
    # Color palette: Group = color family,
    # Localizer = shade within the family
    # ------------------------------------------------

    palette = {
        ("EB", "ptlocal"): "#9ecae1",  # light blue 
        ("NS", "mtlocal"): "#c45100",  # dark orange 
        ("NS", "ptlocal"): "#fdae6b",  # light orange
    } #        ("EB", "mtlocal"): "#1f4e79",  # dark blue

    # Create a combined categorical variable for hue
    for data_df in [df]:
        data_df["GroupLocalizer"] = (
            data_df["Group"] + "_" + data_df["Localizer"]
        )

    hue_order = [
        "EB_ptlocal",
        "NS_mtlocal",
        "NS_ptlocal",
    ] #        "EB_mtlocal",

    combined_palette = {
        f"{group}_{localizer}": color
        for (group, localizer), color in palette.items()
    }
    # ------------------------------------------------
    # Create 2 subplots — one per hemisphere
    # ------------------------------------------------

    # text_plotted = False  # Flag to ensure we only plot the label once

    #  plt.rcParams.update({
    #     'font.weight': 'bold',
    #     'axes.labelweight': 'bold'
    # })


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
            hue="GroupLocalizer",
            hue_order=hue_order,
            palette=combined_palette,
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
            hue="GroupLocalizer",
            hue_order=hue_order,
            palette=combined_palette,
            dodge=True,
            jitter=0.12,
            size=6,
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

        legend_handles = [
            Patch(
                facecolor=combined_palette[key],
                edgecolor="black",
                alpha=0.75,
                label=(
                    f"{key.split('_')[0]} — "
                    f"{key.split('_')[1]}"
                )
            )
            for key in hue_order
        ]


        leg = ax.legend(
            handles=legend_handles,
            title="Group — Localizer",
            loc="upper right",
            frameon=False,
            fontsize=12
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


def pearsonr_combined_plot(rs, reliability, participants, tract_order, bids_path, savefilename):
    #First compute noise ceiling 95% CI
    hemi_labels = ["L", "R"]
    nc_rows = []

    for h in range(2):     # left/right hemispheres
        for s, participant in enumerate(participants):
            gp = "EB" if "EB" in participant else "NS"
            # EB --> ptlocal only
            # NS --> mtlocal + ptlocal
            if "EB" in participant:
                    localizers = ["ptlocal"]
            elif "NS" in participant:
                    localizers = ["mtlocal", "ptlocal"]
            for l, localizer in enumerate(localizers):

                nc_rows.append({
                    "Localizer": localizer,
                    "Subject": s,
                    "Hemisphere": hemi_labels[h],
                    "Group": gp,
                    "Correlation": reliability[s,l,h] #r_all[s, h] #
                })

    nc_df = pd.DataFrame(nc_rows)
    nc_sem_df = (
        nc_df.groupby(["Group", "Localizer", "Hemisphere"])["Correlation"]
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
        .replace("MTxLGNxPU", "Thalamocortical")
        .replace("MTxPTxSTS1", "Temporal")
        .replace("MTxFEF", "Frontal")
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
        for s, participant in enumerate(participants):
            gp = "EB" if "EB" in participant else "NS"
            # EB --> ptlocal only
            # NS --> mtlocal + ptlocal
            if "EB" in participant:
                    localizers = ["ptlocal"]
            elif "NS" in participant:
                    localizers = ["mtlocal", "ptlocal"]
            for l, localizer in enumerate(localizers):
            # pearson = rs[:, s, h]

                rows.append({
                    "Subject": s,
                    "Localizer": localizer,
                    "Hemisphere": hemi_labels[h],
                    "Group": gp,
                    "Correlation": rs[s,l,h] #r_all[s, h] #
                })

    df = pd.DataFrame(rows)

    # ------------------------------------------------
    # Compute SEM per Group × Hemisphere
    # ------------------------------------------------
    sem_df = (
        df.groupby(["Group", "Localizer","Hemisphere"])["Correlation"]
        .agg(["mean", sem])
        .reset_index()
        .rename(columns={"mean": "Mean", "sem": "SEM"})
    )
    # ------------------------------------------------
    # Color palette: Group = color family,
    # Localizer = shade within the family
    # ------------------------------------------------

    palette = {
        ("EB", "ptlocal"): "#9ecae1",  # light blue 
        ("NS", "mtlocal"): "#c45100",  # dark orange 
        ("NS", "ptlocal"): "#fdae6b",  # light orange
    } #        ("EB", "mtlocal"): "#1f4e79",  # dark blue

    # Create a combined categorical variable for hue
    for data_df in [df, nc_df]:
        data_df["GroupLocalizer"] = (
            data_df["Group"] + "_" + data_df["Localizer"]
        )

    hue_order = [
        "EB_ptlocal",
        "NS_mtlocal",
        "NS_ptlocal",
    ] #        "EB_mtlocal",

    combined_palette = {
        f"{group}_{localizer}": color
        for (group, localizer), color in palette.items()
    }

    # ------------------------------------------------
    # Create 2 subplots — one per hemisphere
    # ------------------------------------------------

    # ------------------------------------------------
    # Create 2 subplots — one per hemisphere
    # ------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 16),
        sharey=True
    )

    localizer_order = ["mtlocal", "ptlocal"]
    hue_order = ["NS_mtlocal", "EB_ptlocal", "NS_ptlocal"]

    # Only combinations that actually exist
    combined_palette = {
        "EB_ptlocal": "#9ecae1",  # light blue
        "NS_mtlocal": "#c45100",  # dark orange
        "NS_ptlocal": "#fdae6b",  # light orange
    }

    # Add combined hue variable
    df = df.copy()
    df["GroupLocalizer"] = df["Group"] + "_" + df["Localizer"]

    nc_sem_df = nc_sem_df.copy()
    nc_sem_df["GroupLocalizer"] = (
        nc_sem_df["Group"] + "_" + nc_sem_df["Localizer"]
    )

    for ax, hemi in zip(axes, hemi_labels):

        df_h = df[df["Hemisphere"] == hemi].copy()
        nc_h = nc_sem_df[nc_sem_df["Hemisphere"] == hemi].copy()

        # ------------------------------------------------
        # Bar plot: x-axis = Localizer
        # ------------------------------------------------

        sns.barplot(
            data=df_h,
            x="Localizer",
            y="Correlation",
            order=localizer_order,
            hue="GroupLocalizer",
            hue_order=hue_order,
            palette=combined_palette,
            errorbar="se",
            capsize=0.06,
            alpha=0.75,
            edgecolor="black",
            linewidth=1.5,
            ax=ax
        )

        # ------------------------------------------------
        # Participant dots
        # ------------------------------------------------

        sns.stripplot(
            data=df_h,
            x="Localizer",
            y="Correlation",
            order=localizer_order,
            hue="GroupLocalizer",
            hue_order=hue_order,
            palette=combined_palette,
            dodge=True,
            jitter=0.12,
            size=5,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            legend=False,
            ax=ax
        )

        # ------------------------------------------------
        # Noise ceiling: align with each localizer category
        # ------------------------------------------------

        n_hue = len(hue_order)
        bar_width = 0.8 / n_hue

        for _, row in nc_h.iterrows():

            key = row["GroupLocalizer"]
            if key not in hue_order:
                continue

            localizer_idx = localizer_order.index(row["Localizer"])
            hue_idx = hue_order.index(key)

            x_center = (
                localizer_idx - 0.4
                + bar_width / 2
                + hue_idx * bar_width
            )

            ax.fill_between(
                [x_center - bar_width / 2,
                x_center + bar_width / 2],
                row["CI95_lower"],
                row["CI95_upper"],
                color=combined_palette[key],
                alpha=0.12,
                zorder=0
            )

            ax.hlines(
                [row["CI95_lower"], row["CI95_upper"]],
                xmin=x_center - bar_width / 2,
                xmax=x_center + bar_width / 2,
                colors=combined_palette[key],
                linestyles="--",
                linewidth=1.5
            )

            ax.hlines(
                row["Mean"],
                xmin=x_center - bar_width / 2,
                xmax=x_center + bar_width / 2,
                colors=combined_palette[key],
                linestyles=":",
                linewidth=2
            )

        # ------------------------------------------------
        # Formatting
        # ------------------------------------------------

        ax.set_ylim(-0.2, 1)

        ax.set_title(
            f"{hemi}-Hemisphere",
            fontsize=22,
            fontweight="bold",
            pad=15
        )

        ax.set_xlabel("Localizer", fontsize=18, fontweight="bold")
        ax.set_xticklabels(
            ["MT localizer", "PT localizer"],
            fontsize=16,
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
            axis="both",
            which="major",
            labelsize=16,
            width=2.5
        )

        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        # ------------------------------------------------
        # Manual legend
        # ------------------------------------------------

        legend_labels = {
            "EB_ptlocal": "EB — ptlocal",
            "NS_mtlocal": "NS — mtlocal",
            "NS_ptlocal": "NS — ptlocal",
        }

        legend_handles = [
            Patch(
                facecolor=combined_palette[key],
                edgecolor="black",
                alpha=0.75,
                label=legend_labels[key]
            )
            for key in hue_order
        ]

        legend_handles.append(
            Patch(
                facecolor="lightgray",
                alpha=0.35,
                label="95% noise ceiling CI"
            )
        )

        leg = ax.legend(
            handles=legend_handles,
            title="Group — Localizer",
            loc="upper right",
            frameon=False,
            fontsize=12
        )

        plt.setp(leg.get_texts(), fontweight="bold")
        plt.setp(leg.get_title(), fontweight="bold")


    axes[0].set_ylabel("Pearson's r", fontsize=18, fontweight="bold")
    axes[1].set_ylabel("Pearson's r", fontsize=18, fontweight="bold")

    sns.despine()
    plt.tight_layout()

    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)

    plt.savefig(
        op.join(saveDir, savefilename),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

#========================================
#Save plot of average density map for each group
#========================================
from nilearn import plotting
import nibabel as nib
from nibabel.freesurfer import read_label

def avg_density_plot(density_maps, participants, tract_order, bids_path, img_out_dir, projfrac):
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
                    f"{group}-mean_hemi-{hemi}_label-{tract}_desc-density_mean_inflated_{projfrac}frac.png"
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
            # EB --> ptlocal only
            # NS --> mtlocal + ptlocal
            if "EB" in group:
                    localizers = ["ptlocal"]
            elif "NS" in group:
                    localizers = ["mtlocal", "ptlocal"]
            for c, cont in enumerate(localizers):
                # ----------------------------
                # Density Map visualization
                # ----------------------------
                # img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "wb_participants_linearreg",  "surface_pngs", "mean_predicted_func")
                os.makedirs(img_out_dir, exist_ok=True)

                vmin, vmax = -2, 2 

                # mean_vals = np.nanmean(sub_mean_maps, axis=0)
                hemi_maps = np.array([np.nanmean(func_maps[hemi][sub][cont]["motionXstationary"], axis = 0) for sub in func_maps[hemi] if group in participants[sub] ]) #density_maps[hemi][:][t]
                mean_vals = np.nanmean(hemi_maps, axis = 0) #[[group in p for p in participants],:]
                # Build full-surface vector once
                surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
                surf_map[wang_hmt_vertices] = mean_vals

                # Output filename (no run index)
                out_png = op.join(
                    img_out_dir,
                    f"{group}-mean_hemi-{hemi}_label-{cont}_desc-motionXstationary_mean_inflated.png"
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
