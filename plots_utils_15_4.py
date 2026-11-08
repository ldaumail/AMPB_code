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
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"

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
    # ---- save + close ----
    for fmt in ("png", "svg"):
        plt.savefig(op.join(saveDir, savefilename, f".{fmt}"), dpi=600, bbox_inches="tight")

    plt.show()


def beta_plot(betas, participants, tract_order, bids_path, savefilename, ylim):

    # Keep text as real text (editable in Illustrator)
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

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
    # Organize Betas in table
    # --------------------------------------
    n_tracts = len(tract_order)
    hemi_labels = ["L", "R"]
    if isinstance(betas, np.ndarray):
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
                        rows.append({
                            "Subject": s,
                            "Localizer": localizer,
                            "Tract": tract_order[t],
                            "Hemisphere": hemi_labels[h],
                            "Group": gp,
                            "Beta": betas[s, t, l, h]
                        })

        df = pd.DataFrame(rows)
    else:
        df = betas.copy()

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
    # Color palette
    # ------------------------------------------------
    palette = {
        ("NS", "mtlocal"): "#c45100",  # dark orange
        ("EB", "ptlocal"): "#9ecae1",  # light blue
        ("NS", "ptlocal"): "#fdae6b",  # light orange
    }

    df["GroupLocalizer"] = df["Group"] + "_" + df["Localizer"]

    hue_order = [
        "NS_mtlocal",
        "EB_ptlocal",
        "NS_ptlocal",
    ]

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

    for i, (ax, hemi) in enumerate(zip(axes, hemi_labels)):
        is_bottom = (i == len(axes) - 1)

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

        # x label and tick labels only on the bottom subplot
        ax.set_xticks(np.arange(len(tract_order)))
        if is_bottom:
            ax.set_xlabel("Tract", fontsize=18, fontweight="bold")
            ax.set_xticklabels(
                tract_labels,
                rotation=35,
                ha="right",
                fontsize=16,
                fontweight="bold"
            )
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])   # keeps tick marks, removes text

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
                label=f"{key.split('_')[0]} — {key.split('_')[1]}"
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

    axes[0].set_ylabel("Beta", fontsize=18, fontweight="bold")
    axes[1].set_ylabel("Beta", fontsize=18, fontweight="bold")

    sns.despine()
    plt.tight_layout()

    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)

    # ---- save + close ----
    for fmt in ("eps", "svg"):
        plt.savefig(op.join(saveDir, f"{savefilename}.{fmt}"),
                    dpi=600, bbox_inches="tight")

    plt.show()

# def pearsonr_combined_plot(rs, reliability, participants, tract_order, bids_path,
#                            savefilename, ylim, figsize=(7, 12), bar_width=0.25):
#     #First compute noise ceiling 95% CI
#     hemi_labels = ["L", "R"]
#     if isinstance(reliability, np.ndarray):
#         nc_rows = []

#         for h in range(2):     # left/right hemispheres
#             for s, participant in enumerate(participants):
#                 gp = "EB" if "EB" in participant else "NS"
#                 # EB --> ptlocal only
#                 # NS --> mtlocal + ptlocal
#                 if "EB" in participant:
#                         localizers = ["ptlocal"]
#                 elif "NS" in participant:
#                         localizers = ["mtlocal", "ptlocal"]
#                 for l, localizer in enumerate(localizers):

#                     nc_rows.append({
#                         "Localizer": localizer,
#                         "Subject": s,
#                         "Hemisphere": hemi_labels[h],
#                         "Group": gp,
#                         "Correlation": reliability[s,l,h] #r_all[s, h] #
#                     })
#         nc_df = pd.DataFrame(nc_rows)
#         nc_sem_df = (
#         nc_df.groupby(["Group", "Localizer", "Hemisphere"])["Correlation"]
#         .agg(["mean", sem])
#         .reset_index()
#         .rename(columns={"mean": "Mean", "sem": "SEM"})
#     )
#     else:
#         nc_df = reliability.copy()

#         nc_sem_df = (
#             nc_df.groupby(["Group", "Localizer", "Hemisphere"])["NC"]
#             .agg(["mean", sem])
#             .reset_index()
#             .rename(columns={"mean": "Mean", "sem": "SEM"})
#         )

#     tract_labels = [
#         t.replace("InferiorLongitudinal", "ILF")
#         .replace("InferiorFrontooccipital", "IFOF")
#         .replace("AnteriorVerticalOccipital", "aVOF")
#         .replace("VerticalOccipital", "VOF")
#         .replace("PosteriorVerticalOccipital", "pVOF")
#         .replace("PosteriorArcuate", "pArc")
#         .replace("Arcuate", "AF")
#         .replace("SuperiorLongitudinal", "SLF")
#         .replace("SuperiorLongitudinalI", "SLFI")
#         .replace("SuperiorLongitudinalII", "SLFII")
#         .replace("SuperiorLongitudinalI", "SLFIII")
#         .replace("OpticRadiation", "OR")
#         .replace("AnteriorThalamic", "ATR")
#         .replace("CingulumCingulate", "CC")
#         .replace("Corticospinal", "CST")
#         .replace("MTxLGNxPU", "Thalamocortical")
#         .replace("MTxPTxSTS1", "Temporal")
#         .replace("MTxFEF", "Frontal")
#         for t in tract_order
#     ]

#     nc_sem_df["CI95_upper"] = nc_sem_df["Mean"] + 1.96 * nc_sem_df["SEM"]
#     nc_sem_df["CI95_lower"] = nc_sem_df["Mean"] - 1.96 * nc_sem_df["SEM"]

#     # --------------------------------------
#     # Organize Pearson's r in table
#     # --------------------------------------

#     hemi_labels = ["L", "R"]
#     if isinstance(rs, np.ndarray):
#         rows = []

#         for h in range(2):     # left/right hemispheres
#             for s, participant in enumerate(participants):
#                 gp = "EB" if "EB" in participant else "NS"
#                 # EB --> ptlocal only
#                 # NS --> mtlocal + ptlocal
#                 if "EB" in participant:
#                         localizers = ["ptlocal"]
#                 elif "NS" in participant:
#                         localizers = ["mtlocal", "ptlocal"]
#                 for l, localizer in enumerate(localizers):
#                 # pearson = rs[:, s, h]

#                     rows.append({
#                         "Subject": s,
#                         "Localizer": localizer,
#                         "Hemisphere": hemi_labels[h],
#                         "Group": gp,
#                         "Correlation": rs[s,l,h] #r_all[s, h] #
#                     })

#         df = pd.DataFrame(rows)
#     else:
#          df = rs.copy()

#     # ------------------------------------------------
#     # Compute SEM per Group × Hemisphere
#     # ------------------------------------------------
#     sem_df = (
#         df.groupby(["Group", "Localizer","Hemisphere"])["Correlation"]
#         .agg(["mean", sem])
#         .reset_index()
#         .rename(columns={"mean": "Mean", "sem": "SEM"})
#     )
#     # ------------------------------------------------
#     # Color palette: Group = color family,
#     # Localizer = shade within the family
#     # ------------------------------------------------

#     palette = {
#         ("EB", "ptlocal"): "#9ecae1",  # light blue 
#         ("NS", "mtlocal"): "#c45100",  # dark orange 
#         ("NS", "ptlocal"): "#fdae6b",  # light orange
#     } #        ("EB", "mtlocal"): "#1f4e79",  # dark blue

#     # Create a combined categorical variable for hue
#     for data_df in [df, nc_df]:
#         data_df["GroupLocalizer"] = (
#             data_df["Group"] + "_" + data_df["Localizer"]
#         )

#     hue_order = [
#         "EB_ptlocal",
#         "NS_mtlocal",
#         "NS_ptlocal",
#     ] #        "EB_mtlocal",

#     combined_palette = {
#         f"{group}_{localizer}": color
#         for (group, localizer), color in palette.items()
#     }

#     # ------------------------------------------------
#     # Create 2 subplots — one per hemisphere
#     # ------------------------------------------------

#     fig, axes = plt.subplots(
#         2, 1,
#         figsize=figsize,
#         sharey=True
#     )

#     localizer_order = ["mtlocal", "ptlocal"]
#     hue_order = ["NS_mtlocal", "EB_ptlocal", "NS_ptlocal"]

#     # Only combinations that actually exist
#     combined_palette = {
#         "EB_ptlocal": "#9ecae1",  # light blue
#         "NS_mtlocal": "#c45100",  # dark orange
#         "NS_ptlocal": "#fdae6b",  # light orange
#     }

#     # Add combined hue variable
#     df = df.copy()
#     df["GroupLocalizer"] = df["Group"] + "_" + df["Localizer"]

#     nc_sem_df = nc_sem_df.copy()
#     nc_sem_df["GroupLocalizer"] = (
#         nc_sem_df["Group"] + "_" + nc_sem_df["Localizer"]
#     )

#     rng = np.random.default_rng(0)   # reproducible dot jitter

#     for ax, hemi in zip(axes, hemi_labels):

#         # Groups that actually exist at each localizer
#         groups_per_loc = {
#             "mtlocal": ["NS_mtlocal"],
#             "ptlocal": ["EB_ptlocal", "NS_ptlocal"],
#         }

#         df_h = df[df["Hemisphere"] == hemi].copy()
#         nc_h = nc_sem_df[nc_sem_df["Hemisphere"] == hemi].copy()

#         # ------------------------------------------------
#         # Bars + dots, one localizer at a time
#         # ------------------------------------------------
#         for loc, groups in groups_per_loc.items():
#             d = df_h[df_h["Localizer"] == loc]
#             if d.empty:
#                 continue

#             sns.barplot(
#                 data=d,
#                 x="Localizer",
#                 y="Correlation",
#                 order=localizer_order,        # keeps x positions 0 and 1
#                 hue="GroupLocalizer",
#                 hue_order=groups,
#                 palette=combined_palette,
#                 errorbar="se",
#                 capsize=0.06,
#                 alpha=0.75,
#                 edgecolor="black",
#                 linewidth=1.5,
#                 width=bar_width * len(groups),  # per-bar width = bar_width
#                 legend=False,
#                 ax=ax
#             )

#             # Participant dots, drawn manually so they stay centered on the
#             # (now thinner) bars. seaborn's stripplot dodges over a fixed
#             # width of 0.8 and would no longer line up with the bars.
#             loc_idx = localizer_order.index(loc)
#             for gi, key in enumerate(groups):
#                 vals = d.loc[d["GroupLocalizer"] == key, "Correlation"].to_numpy()
#                 xc = loc_idx + (gi - (len(groups) - 1) / 2) * bar_width
#                 xs = xc + rng.uniform(-bar_width * 0.25, bar_width * 0.25, len(vals))
#                 ax.scatter(xs, vals, s=25, color=combined_palette[key],
#                            edgecolor="black", linewidth=0.6, alpha=0.85, zorder=3)

#         # ------------------------------------------------
#         # Noise ceiling: aligned with the new bar positions
#         # ------------------------------------------------
#         for _, row in nc_h.iterrows():
#             key = row["GroupLocalizer"]
#             loc = row["Localizer"]
#             if key not in groups_per_loc.get(loc, []):
#                 continue

#             groups = groups_per_loc[loc]
#             loc_idx = localizer_order.index(loc)
#             x_center = loc_idx + (groups.index(key) - (len(groups) - 1) / 2) * bar_width
#             x0, x1 = x_center - bar_width / 2, x_center + bar_width / 2

#             ax.fill_between([x0, x1], row["CI95_lower"], row["CI95_upper"],
#                             color=combined_palette[key], alpha=0.12, zorder=0)
#             ax.hlines([row["CI95_lower"], row["CI95_upper"]], xmin=x0, xmax=x1,
#                       colors=combined_palette[key], linestyles="--", linewidth=1.5)
#             ax.hlines(row["Mean"], xmin=x0, xmax=x1,
#                       colors=combined_palette[key], linestyles=":", linewidth=2)

#         # ------------------------------------------------
#         # Formatting
#         # ------------------------------------------------

#         ax.set_ylim(ylim)
#         ax.set_xlim(-0.5, 1.5)          # shorter x axis around the two groups of bars

#         ax.set_title(
#             f"{hemi}-Hemisphere",
#             fontsize=22,
#             fontweight="bold",
#             pad=15
#         )

#         ax.set_xlabel("Localizer", fontsize=18, fontweight="bold")
#         ax.set_xticks([0, 1])
#         ax.set_xticklabels(
#             ["MT localizer", "PT localizer"],
#             fontsize=16,
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
#             axis="both",
#             which="major",
#             labelsize=16,
#             width=2.5
#         )

#         for label in ax.get_yticklabels():
#             label.set_fontweight("bold")

#         # ------------------------------------------------
#         # Manual legend
#         # ------------------------------------------------

#         legend_labels = {
#             "EB_ptlocal": "EB — ptlocal",
#             "NS_mtlocal": "NS — mtlocal",
#             "NS_ptlocal": "NS — ptlocal",
#         }

#         legend_handles = [
#             Patch(
#                 facecolor=combined_palette[key],
#                 edgecolor="black",
#                 alpha=0.75,
#                 label=legend_labels[key]
#             )
#             for key in hue_order
#         ]

#         legend_handles.append(
#             Patch(
#                 facecolor="lightgray",
#                 alpha=0.35,
#                 label="95% noise ceiling CI"
#             )
#         )

#         leg = ax.legend(
#             handles=legend_handles,
#             title="Group — Localizer",
#             loc="upper right",
#             frameon=False,
#             fontsize=12
#         )

#         plt.setp(leg.get_texts(), fontweight="bold")
#         plt.setp(leg.get_title(), fontweight="bold")


#     axes[0].set_ylabel("Pearson's r", fontsize=18, fontweight="bold")
#     axes[1].set_ylabel("Pearson's r", fontsize=18, fontweight="bold")

#     sns.despine()
#     plt.tight_layout()

#     saveDir = op.join(bids_path, "analysis", "plots")
#     os.makedirs(saveDir, exist_ok=True)

#     # ---- save + close ----
#     for fmt in ("eps", "svg"):
#         plt.savefig(op.join(saveDir, f"{savefilename}.{fmt}"), dpi=600, bbox_inches="tight")

#     plt.show()
def pearsonr_combined_plot(rs, reliability, participants, bids_path,
                           savefilename, ylim, figsize=(7, 12), bar_width=0.25):
    #First compute noise ceiling 95% CI
    hemi_labels = ["L", "R"]
    if isinstance(reliability, np.ndarray):
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
                        "Correlation": reliability[s, l, h]
                    })
        nc_df = pd.DataFrame(nc_rows)
        nc_sem_df = (
            nc_df.groupby(["Group", "Localizer", "Hemisphere"])["Correlation"]
            .agg(["mean", sem])
            .reset_index()
            .rename(columns={"mean": "Mean", "sem": "SEM"})
        )
    else:
        nc_df = reliability.copy()

        nc_sem_df = (
            nc_df.groupby(["Group", "Localizer", "Hemisphere"])["NC"]
            .agg(["mean", sem])
            .reset_index()
            .rename(columns={"mean": "Mean", "sem": "SEM"})
        )

    nc_sem_df["CI95_upper"] = nc_sem_df["Mean"] + 1.96 * nc_sem_df["SEM"]
    nc_sem_df["CI95_lower"] = nc_sem_df["Mean"] - 1.96 * nc_sem_df["SEM"]

    # --------------------------------------
    # Organize Pearson's r in table
    # --------------------------------------
    if isinstance(rs, np.ndarray):
        rows = []

        for h in range(2):     # left/right hemispheres
            for s, participant in enumerate(participants):
                gp = "EB" if "EB" in participant else "NS"
                if "EB" in participant:
                    localizers = ["ptlocal"]
                elif "NS" in participant:
                    localizers = ["mtlocal", "ptlocal"]
                for l, localizer in enumerate(localizers):
                    rows.append({
                        "Subject": s,
                        "Localizer": localizer,
                        "Hemisphere": hemi_labels[h],
                        "Group": gp,
                        "Correlation": rs[s, l, h]
                    })

        df = pd.DataFrame(rows)
    else:
        df = rs.copy()

    # ------------------------------------------------
    # Colors, bar order and x positions
    # ------------------------------------------------
    combined_palette = {
        "EB_ptlocal": "#9ecae1",  # light blue
        "NS_mtlocal": "#c45100",  # dark orange
        "NS_ptlocal": "#fdae6b",  # light orange
    }
    hue_order = ["NS_mtlocal", "EB_ptlocal", "NS_ptlocal"]

    # Three bars touching each other: centers are exactly one bar_width apart
    x_pos = {key: i * bar_width for i, key in enumerate(hue_order)}
    # Tick positions: the mtlocal bar, and the middle of the two ptlocal bars
    tick_pos = [x_pos["NS_mtlocal"],
                (x_pos["EB_ptlocal"] + x_pos["NS_ptlocal"]) / 2]

    df = df.copy()
    df["GroupLocalizer"] = df["Group"] + "_" + df["Localizer"]
    nc_sem_df = nc_sem_df.copy()
    nc_sem_df["GroupLocalizer"] = nc_sem_df["Group"] + "_" + nc_sem_df["Localizer"]

    legend_labels = {
        "EB_ptlocal": "EB — ptlocal",
        "NS_mtlocal": "NS — mtlocal",
        "NS_ptlocal": "NS — ptlocal",
    }

    # ------------------------------------------------
    # Create 2 subplots — one per hemisphere
    # ------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharey=True)

    rng = np.random.default_rng(0)   # reproducible dot jitter

    for ax, hemi in zip(axes, hemi_labels):

        df_h = df[df["Hemisphere"] == hemi]
        nc_h = nc_sem_df[nc_sem_df["Hemisphere"] == hemi]

        # ------------------------------------------------
        # Bars, error bars (SE) and participant dots
        # ------------------------------------------------
        for key in hue_order:
            vals = df_h.loc[df_h["GroupLocalizer"] == key, "Correlation"].to_numpy()
            if len(vals) == 0:
                continue
            xc = x_pos[key]

            ax.bar(xc, vals.mean(), width=bar_width,
                   color=combined_palette[key], alpha=0.75,
                   edgecolor="black", linewidth=1.5, zorder=2)
            ax.errorbar(xc, vals.mean(), yerr=sem(vals), color="black",
                        capsize=4, linewidth=1.5, capthick=1.5, zorder=4)

            xs = xc + rng.uniform(-bar_width * 0.25, bar_width * 0.25, len(vals))
            ax.scatter(xs, vals, s=25, color=combined_palette[key],
                       edgecolor="black", linewidth=0.6, alpha=0.85, zorder=3)

        # ------------------------------------------------
        # Noise ceiling: same x positions as the bars
        # ------------------------------------------------
        for _, row in nc_h.iterrows():
            key = row["GroupLocalizer"]
            if key not in x_pos:
                continue
            x0 = x_pos[key] - bar_width / 2
            x1 = x_pos[key] + bar_width / 2

            ax.fill_between([x0, x1], row["CI95_lower"], row["CI95_upper"],
                            color=combined_palette[key], alpha=0.12, zorder=0)
            ax.hlines([row["CI95_lower"], row["CI95_upper"]], xmin=x0, xmax=x1,
                      colors=combined_palette[key], linestyles="--", linewidth=1.5)
            ax.hlines(row["Mean"], xmin=x0, xmax=x1,
                      colors=combined_palette[key], linestyles=":", linewidth=2)

        # ------------------------------------------------
        # Formatting
        # ------------------------------------------------
        ax.set_ylim(ylim)
        pad = 0.4 * bar_width    # small margin around the three bars
        ax.set_xlim(x_pos[hue_order[0]] - bar_width / 2 - pad,
                    x_pos[hue_order[-1]] + bar_width / 2 + pad)

        ax.set_title(f"{hemi}-Hemisphere", fontsize=22, fontweight="bold", pad=15)

        ax.set_xticks([])        # no x ticks or tick labels
        ax.set_xlabel("")

        ax.axhline(0, color="gray", linestyle="--", linewidth=1.5)

        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)

        ax.tick_params(axis="both", which="major", labelsize=16, width=2.5)

        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        # ------------------------------------------------
        # Manual legend
        # ------------------------------------------------
        legend_handles = [
            Patch(facecolor=combined_palette[key], edgecolor="black",
                  alpha=0.75, label=legend_labels[key])
            for key in hue_order
        ]
        legend_handles.append(
            Patch(facecolor="lightgray", alpha=0.35, label="95% noise ceiling CI")
        )

        leg = ax.legend(handles=legend_handles, title="Group — Localizer",
                        loc="upper right", frameon=False, fontsize=12)
        plt.setp(leg.get_texts(), fontweight="bold")
        plt.setp(leg.get_title(), fontweight="bold")

    axes[0].set_ylabel("Pearson's r", fontsize=18, fontweight="bold")
    axes[1].set_ylabel("Pearson's r", fontsize=18, fontweight="bold")

    sns.despine()
    plt.tight_layout()

    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)

    # ---- save + close ----
    for fmt in ("eps", "svg"):
        plt.savefig(op.join(saveDir, f"{savefilename}.{fmt}"), dpi=600, bbox_inches="tight")

    plt.show()

#========================================
#Save plot of maps for each group
#========================================
from nilearn import plotting
import nibabel as nib
from nibabel.freesurfer import read_label
import os
import os.path as op
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nilearn import plotting

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
                out_base = op.join(
                    img_out_dir,
                    f"{group}-mean_hemi-{hemi}_label-{tract}_desc-density_mean_inflated_{projfrac}frac"
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
                # before saving: rasterize the 3D surface only
                for ax in display.axes:
                    if ax.name == "3d":
                        for coll in ax.collections:
                            coll.set_rasterized(True)

                plt.rcParams["svg.fonttype"] = "none"   # keep text editable (set once, at the top)

                # ---- save + close ----
                for fmt in ("png", "svg"):
                    display.savefig(f"{out_base}.{fmt}", dpi=600)
                plt.close(display.figure)


def avg_density_cropped_plot(
    density_maps, participants, tract_order, bids_path,
    img_out_dir, projfrac, ylim, crop_radius=None, margin=15,
    cmap="plasma", figsize=(11, 5.5)
):
    """
    One figure per group and tract: left + right hemispheres side by side.

    Both hemispheres use the SAME crop radius and are drawn at the SAME scale.

    crop_radius=None:
        Automatically calculate the radius from the largest extent of the
        Wang hMT+ ROI and both groups' functional MT ROIs across hemispheres,
        then add margin (mm).

    crop_radius=<float>:
        Use the specified radius (mm) for both hemispheres. In this case,
        margin is ignored.
    """

    fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    hemis = ["L", "R"]
    groups = ["EB", "NS"]

    vmin, vmax = ylim

    os.makedirs(img_out_dir, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"

    # ------------------------------------------------
    # Per-hemisphere setup: ROI, curvature, geometry,
    # centroid, extent, and density maps
    # ------------------------------------------------
    hemi_info = {}

    for hemi in hemis:

        hemi_fs = "lh" if hemi == "L" else "rh"

        # --------------------------------------------
        # Load Wang hMT+ ROI
        # --------------------------------------------
        wang_hmt_path = op.join(
            '/Users', 'ldaumail3', 'Documents', 'research',
            'brain_atlases', 'Wang_2015', 'hmtplus',
            f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
        )

        surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
        n_vertices = len(surf_roi)
        wang_hmt_vertices = np.where(surf_roi > 0)[0]

        print(f"{len(wang_hmt_vertices)} Wang hMT+ vertices ({hemi})")

        # --------------------------------------------
        # Load curvature background
        # --------------------------------------------
        curv_file = op.join(
            fs_path, "fsaverage", "surf", f"{hemi_fs}.curv"
        )

        curv = nib.freesurfer.read_morph_data(curv_file)

        curv_norm = (curv - np.percentile(curv, 5)) / (
            np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8
        )
        curv_norm = np.clip(curv_norm, 0, 1)

        # --------------------------------------------
        # Load inflated geometry
        # --------------------------------------------
        infl_surf = op.join(
            fs_path, "fsaverage", "surf", f"{hemi_fs}.inflated"
        )

        coords, faces = nib.freesurfer.read_geometry(infl_surf)

        # Wang hMT+ centroid
        center = coords[wang_hmt_vertices].mean(axis=0)

        # --------------------------------------------
        # Calculate ROI extent
        # Include Wang hMT+ and both groups' functional MT ROIs
        # --------------------------------------------
        roi_vertices = [wang_hmt_vertices]

        for group in groups:

            label_file = op.join(
                bids_path, 'analysis', 'ROIs', 'func_roi',
                'functional_surf_roi', 'group_averages',
                f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
            )

            if op.exists(label_file):
                roi_vertices.append(
                    np.asarray(read_label(label_file))
                )

        roi_vertices = np.unique(np.concatenate(roi_vertices))

        extent = np.linalg.norm(
            coords[roi_vertices] - center, axis=1
        ).max()

        # --------------------------------------------
        # Prepare density maps
        # Shape per tract: (n_participants, n_masked_vertices)
        # --------------------------------------------
        hemi_maps = {
            t: np.array([
                density_maps[hemi][sub][t]
                for sub in density_maps[hemi]
            ])
            for t in range(len(tract_order))
        }

        # Store hemisphere information
        hemi_info[hemi] = dict(
            n_vertices=n_vertices,
            wang_hmt_vertices=wang_hmt_vertices,
            curv_norm=curv_norm,
            coords=coords,
            faces=faces,
            center=center,
            extent=extent,
            hemi_maps=hemi_maps,
        )

    # ------------------------------------------------
    # Determine common crop radius
    # ------------------------------------------------
    if crop_radius is None:

        crop_radius = max(
            info["extent"] for info in hemi_info.values()
        ) + margin

    print(
        f"\nCrop radius used for both hemispheres: "
        f"{crop_radius:.2f} mm"
    )

    # ------------------------------------------------
    # Crop meshes
    # ------------------------------------------------
    for hemi, info in hemi_info.items():

        coords = info["coords"]
        faces = info["faces"]

        keep = np.linalg.norm(
            coords - info["center"], axis=1
        ) <= crop_radius

        idx = np.where(keep)[0]

        new_index = -np.ones(info["n_vertices"], dtype=int)
        new_index[idx] = np.arange(len(idx))

        face_mask = keep[faces].all(axis=1)

        info["idx"] = idx

        info["mesh_crop"] = (
            coords[idx],
            new_index[faces[face_mask]]
        )

        info["curv_crop"] = info["curv_norm"][idx]

    # ------------------------------------------------
    # Print crop diagnostics
    # ------------------------------------------------
    for hemi, info in hemi_info.items():

        print(
            f"{hemi}: "
            f"extent={info['extent']:.2f} mm | "
            f"radius={crop_radius:.2f} mm | "
            f"vertices retained={len(info['idx'])}/"
            f"{info['n_vertices']} "
            f"({100 * len(info['idx']) / info['n_vertices']:.1f}%)"
        )

    # ------------------------------------------------
    # One figure per group and tract
    # ------------------------------------------------
    for group in groups:

        # Select participants belonging to this group
        group_mask = np.array([
            group in participants[sub]
            for sub in density_maps["L"]
        ])

        print(
            f"\n{group}: "
            f"{group_mask.sum()} participants"
        )

        for t, tract in enumerate(tract_order):

            fig = plt.figure(figsize=figsize)

            # ----------------------------------------
            # Plot both hemispheres
            # ----------------------------------------
            for h, hemi in enumerate(hemis):

                info = hemi_info[hemi]

                ax = fig.add_subplot(
                    1, 2, h + 1,
                    projection="3d"
                )

                # Average density across group participants
                mean_vals = np.nanmean(
                    info["hemi_maps"][t][group_mask, :],
                    axis=0
                )

                # Build full-surface vector
                surf_map = np.full(
                    (info["n_vertices"],),
                    np.nan,
                    dtype=np.float32
                )

                surf_map[info["wang_hmt_vertices"]] = mean_vals

                # ------------------------------------
                # Plot cropped density map
                # ------------------------------------
                plotting.plot_surf_stat_map(
                    surf_mesh=info["mesh_crop"],
                    stat_map=surf_map[info["idx"]],
                    hemi="left" if hemi == "L" else "right",
                    view="lateral",
                    cmap=cmap,
                    colorbar=False,
                    vmin=vmin,
                    vmax=vmax,
                    threshold=None,
                    bg_map=info["curv_crop"],
                    bg_on_data=True,
                    darkness=0.6,
                    axes=ax,
                    figure=fig,
                )

                # ------------------------------------
                # Overlay functional MT boundary
                # for the current group
                # ------------------------------------
                # label_file = op.join(
                #     bids_path, 'analysis', 'ROIs', 'func_roi',
                #     'functional_surf_roi', 'group_averages',
                #     f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
                # )

                # if op.exists(label_file):

                #     func_mt_vertices = np.asarray(
                #         read_label(label_file)
                #     )

                #     func_mt_roi = np.zeros(
                #         info["n_vertices"],
                #         dtype=np.float32
                #     )

                #     func_mt_roi[func_mt_vertices] = 1

                #     # Restrict contour map to cropped vertices
                #     func_mt_crop = func_mt_roi[info["idx"]]

                #     plotting.plot_surf_contours(
                #         surf_mesh=info["mesh_crop"],
                #         roi_map=func_mt_crop,
                #         levels=[1],
                #         colors=["lightgray"],
                #         linewidths=2.0,
                #         figure=fig,
                #         axes=ax,
                #     )

                # ------------------------------------
                # Identical scale for both hemispheres
                # ------------------------------------
                cx, cy, cz = info["center"]

                ax.set_xlim(
                    cx - crop_radius,
                    cx + crop_radius
                )
                ax.set_ylim(
                    cy - crop_radius,
                    cy + crop_radius
                )
                ax.set_zlim(
                    cz - crop_radius,
                    cz + crop_radius
                )

                ax.set_box_aspect((1, 1, 1))

                ax.set_title(
                    f"{hemi} hemisphere",
                    fontsize=18,
                    fontweight="bold"
                )

                # Rasterize surface only
                for coll in ax.collections:
                    coll.set_rasterized(True)

            # ----------------------------------------
            # Shared colorbar in a dedicated right margin
            # ----------------------------------------
            sm = ScalarMappable(
                norm=Normalize(vmin=vmin, vmax=vmax),
                cmap=cmap
            )

            # Reserve space on the right for the colorbar
            fig.subplots_adjust(
                left=0.02,
                right=0.86,
                bottom=0.05,
                top=0.88,
                wspace=0.08
            )

            # Dedicated colorbar axis
            cbar_ax = fig.add_axes([0.89, 0.20, 0.018, 0.60])

            cbar = fig.colorbar(sm, cax=cbar_ax)

            cbar.ax.tick_params(labelsize=14)

            for lab in cbar.ax.get_yticklabels():
                lab.set_fontweight("bold")

            # ----------------------------------------
            # Figure title
            # ----------------------------------------
            fig.suptitle(
                f"{group} mean - {tract} (density)",
                fontsize=20,
                fontweight="bold"
            )

            cbar.ax.tick_params(labelsize=14)

            for lab in cbar.ax.get_yticklabels():
                lab.set_fontweight("bold")

            # ----------------------------------------
            # Save
            # ----------------------------------------
            out_base = op.join(
                img_out_dir,
                f"{group}_hemi-LR_label-{tract}_desc-density_"
                f"inflated_{projfrac}frac_cropped"
            )

            for fmt in ("png", "svg"):
                fig.savefig(
                    f"{out_base}.{fmt}",
                    dpi=600,
                    bbox_inches="tight"
                )

            plt.close(fig)

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
                out_base = op.join(
                    img_out_dir,
                    f"{group}-mean_hemi-{hemi}_label-{cont}_desc-motionXstationary_mean_inflated"
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

                # before saving: rasterize the 3D surface only
                for ax in display.axes:
                    if ax.name == "3d":
                        for coll in ax.collections:
                            coll.set_rasterized(True)

                plt.rcParams["svg.fonttype"] = "none"   # keep text editable (set once, at the top)

                # ---- save + close ----
                for fmt in ("png", "svg"):
                    display.savefig(f"{out_base}.{fmt}", dpi=600)
                # display.savefig(out_png, dpi=300)
                plt.close(display.figure)


def avg_func_cropped_plot(func_maps, participants, bids_path, img_out_dir,
                  crop_radius=None, margin=15, vmin=-2, vmax=2, cmap="plasma",
                  figsize=(11, 5.5)):
    """One figure per (group, localizer), left + right hemisphere side by side.
    Both hemispheres are cropped with the SAME radius and drawn at the SAME scale.
    crop_radius=None -> radius = largest hMT+ extent over both hemispheres + margin (mm)."""
    fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    hemis = ["L", "R"]
    groups = ["EB", "NS"]
    os.makedirs(img_out_dir, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"   # keep text editable

    # ------------------------------------------------
    # Per-hemisphere setup (done once): ROI, curvature, geometry, centroid, extent
    # ------------------------------------------------
    hemi_info = {}
    for hemi in hemis:
        hemi_fs = "lh" if hemi == "L" else "rh"

        wang_hmt_path = op.join(
            '/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases',
            'Wang_2015', 'hmtplus',
            f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
        )
        surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
        n_vertices = len(surf_roi)
        wang_hmt_vertices = np.where(surf_roi > 0)[0]
        print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")

        # Curvature background
        curv = nib.freesurfer.read_morph_data(
            op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv"))
        curv_norm = (curv - np.percentile(curv, 5)) / (
            np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8)
        curv_norm = np.clip(curv_norm, 0, 1)

        # Inflated geometry and hMT+ centroid
        coords, faces = nib.freesurfer.read_geometry(
            op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.inflated"))
        center = coords[wang_hmt_vertices].mean(axis=0)

        # Extent of hMT+ (Wang ROI + functional MT labels of both groups)
        roi_vertices = [wang_hmt_vertices]
        for group in groups:
            lf = op.join(
                bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi',
                'group_averages',
                f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label")
            if op.exists(lf):
                roi_vertices.append(np.asarray(read_label(lf)))
        roi_vertices = np.unique(np.concatenate(roi_vertices))
        extent = np.linalg.norm(coords[roi_vertices] - center, axis=1).max()

        hemi_info[hemi] = dict(
            n_vertices=n_vertices, wang_hmt_vertices=wang_hmt_vertices,
            curv_norm=curv_norm, coords=coords, faces=faces,
            center=center, extent=extent,
        )

    # ------------------------------------------------
    # Common radius: the less-cropped (larger) hemisphere sets it for both
    # ------------------------------------------------
    if crop_radius is None:
        crop_radius = max(info["extent"] for info in hemi_info.values()) + margin
    print(f"Crop radius used for both hemispheres: {crop_radius:.1f} mm")

    # Cropped meshes with the common radius
    for hemi, info in hemi_info.items():
        coords, faces = info["coords"], info["faces"]
        keep = np.linalg.norm(coords - info["center"], axis=1) <= crop_radius
        idx = np.where(keep)[0]
        new_index = -np.ones(info["n_vertices"], dtype=int)
        new_index[idx] = np.arange(len(idx))
        face_mask = keep[faces].all(axis=1)
        info["idx"] = idx
        info["mesh_crop"] = (coords[idx], new_index[faces[face_mask]])
        info["curv_crop"] = info["curv_norm"][idx]
    for hemi, info in hemi_info.items():
        print(
            f"{hemi}: "
            f"extent={info['extent']:.2f} mm | "
            f"radius={crop_radius:.2f} mm | "
            f"vertices retained={len(info['idx'])}/{info['n_vertices']} "
            f"({100 * len(info['idx']) / info['n_vertices']:.1f}%)"
        )
    # ------------------------------------------------
    # One figure per group x localizer
    # ------------------------------------------------
    for group in groups:
        localizers = ["ptlocal"] if "EB" in group else ["mtlocal", "ptlocal"]

        for cont in localizers:
            fig = plt.figure(figsize=figsize)

            for h, hemi in enumerate(hemis):
                info = hemi_info[hemi]
                ax = fig.add_subplot(1, 2, h + 1, projection="3d")

                # Functional MT ROI (group average label) for the contour
                label_file = op.join(
                    bids_path, 'analysis', 'ROIs', 'func_roi',
                    'functional_surf_roi', 'group_averages',
                    f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
                )
                func_mt_roi = np.zeros(info["n_vertices"], dtype=np.float32)
                func_mt_roi[read_label(label_file)] = 1

                # Group-average functional map on the full surface, then crop
                hemi_maps = np.array([
                    np.nanmean(func_maps[hemi][sub][cont]["motionXstationary"], axis=0)
                    for sub in func_maps[hemi] if group in participants[sub]
                ])
                mean_vals = np.nanmean(hemi_maps, axis=0)
                surf_map = np.full((info["n_vertices"],), np.nan, dtype=np.float32)
                surf_map[info["wang_hmt_vertices"]] = mean_vals

                plotting.plot_surf_stat_map(
                    surf_mesh=info["mesh_crop"],
                    stat_map=surf_map[info["idx"]],
                    hemi="left" if hemi == "L" else "right",
                    view="lateral",
                    cmap=cmap,
                    colorbar=False,                 # one shared colorbar below
                    vmin=vmin, vmax=vmax,
                    threshold=None,
                    bg_map=info["curv_crop"],
                    bg_on_data=True,
                    darkness=0.6,
                    axes=ax, figure=fig,
                )

                # ---- functional MT boundary overlay ----
                plotting.plot_surf_contours(
                    surf_mesh=info["mesh_crop"],
                    roi_map=func_mt_roi[info["idx"]],
                    levels=[1],
                    colors=["lightgray"],
                    linewidths=2.0,
                    figure=fig, axes=ax,
                )

                # ---- identical scale for both hemispheres ----
                # nilearn autoscales each axis to its own mesh; override with the
                # same span, centered on each hemisphere's hMT+ centroid
                cx, cy, cz = info["center"]
                ax.set_xlim(cx - crop_radius, cx + crop_radius)
                ax.set_ylim(cy - crop_radius, cy + crop_radius)
                ax.set_zlim(cz - crop_radius, cz + crop_radius)
                ax.set_box_aspect((1, 1, 1))

                ax.set_title(f"{hemi} hemisphere", fontsize=18, fontweight="bold")

                # Rasterize the 3D surface only (keeps the SVG small)
                for coll in ax.collections:
                    coll.set_rasterized(True)

            # ---- one shared colorbar ----
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.02)
            cbar.ax.tick_params(labelsize=14)
            for lab in cbar.ax.get_yticklabels():
                lab.set_fontweight("bold")

            fig.suptitle(f"{group} - {cont} (motion > stationary)",
                         fontsize=20, fontweight="bold")

            # ---- save + close ----
            out_base = op.join(
                img_out_dir,
                f"{group}-mean_hemi-LR_label-{cont}_desc-motionXstationary_mean_inflated_cropped"
            )
            for fmt in ("png", "svg"):
                fig.savefig(f"{out_base}.{fmt}", dpi=600, bbox_inches="tight")
            plt.close(fig)


def participant_density_plot(density_maps, participant, pidx, tract_order, bids_path, img_out_dir, ylim, projfrac):
    fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    hemis = ["L", "R"]
    groups = ["EB", "NS"]
    for h, hemi in enumerate(hemis):
        hemi_fs = "lh" if hemi == "L" else "rh"

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
            participant,
            f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
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

            vmin, vmax = ylim 

            # mean_vals = np.nanmean(sub_mean_maps, axis=0)
            hemi_maps = np.array([density_maps[hemi][sub][t] for sub in density_maps[hemi]]) #density_maps[hemi][:][t]
            
            density_vals = hemi_maps[pidx,:]
            # Build full-surface vector once
            surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
            surf_map[wang_hmt_vertices] = density_vals

            # Output filename (no run index)
            out_base = op.join(
                img_out_dir,
                f"{participant}_hemi-{hemi}_label-{tract}_desc-density_inflated_{projfrac}frac"
            )

            # -------------------------------------------------
            # Plot participant map
            # -------------------------------------------------
            infl_surf = op.join(fs_path, "fsaverage", "surf", f"{'lh' if hemi == 'L' else 'rh'}.inflated")
            display = plotting.plot_surf_stat_map(
                surf_mesh=infl_surf,
                stat_map=surf_map,
                hemi="left" if hemi == "L" else "right",
                view="lateral",
                cmap="jet",
                colorbar=True,
                vmin=vmin,
                vmax=vmax,
                threshold=None,
                bg_map=curv_norm,
                bg_on_data=True,
                darkness=0.6,
            )
            # ---- MT boundary overlay ----
            # plotting.plot_surf_contours(
            #     surf_mesh=infl_surf,
            #     roi_map=func_mt_roi,
            #     levels=[1],
            #     colors=["lightgray"],
            #     linewidths=2.0,
            #     figure=display.figure,
            #     axes=display.axes[0]
            # )
            # before saving: rasterize the 3D surface only
            for ax in display.axes:
                if ax.name == "3d":
                    for coll in ax.collections:
                        coll.set_rasterized(True)

            plt.rcParams["svg.fonttype"] = "none"   # keep text editable (set once, at the top)

            # ---- save + close ----
            for fmt in ("png", "svg"):
                display.savefig(f"{out_base}.{fmt}", dpi=600)
            plt.close(display.figure)

def participant_density_cropped_plot(density_maps, participant, pidx, tract_order, bids_path,
                             img_out_dir, ylim, projfrac,
                             crop_radius=None, margin=15, cmap="RdYlGn_r",
                             figsize=(11, 5.5)):
    """One figure per tract: left + right hemisphere side by side for one participant.
    Both hemispheres use the SAME crop radius and are drawn at the SAME scale.
    crop_radius=None -> radius = largest hMT+ (Wang ROI) extent over both hemispheres + margin (mm)."""
    fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    hemis = ["L", "R"]
    groups = ["EB", "NS"]
    vmin, vmax = ylim
    os.makedirs(img_out_dir, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"   # keep text editable

    # ------------------------------------------------
    # Per-hemisphere setup (done once): ROI, curvature, geometry, centroid, extent
    # ------------------------------------------------
    hemi_info = {}
    for hemi in hemis:
        hemi_fs = "lh" if hemi == "L" else "rh"

        wang_hmt_path = op.join(
            '/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases',
            'Wang_2015', 'hmtplus',
            f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
        )
        surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
        n_vertices = len(surf_roi)
        wang_hmt_vertices = np.where(surf_roi > 0)[0]
        print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")

        # Curvature background
        curv = nib.freesurfer.read_morph_data(
            op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv"))
        curv_norm = (curv - np.percentile(curv, 5)) / (
            np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8)
        curv_norm = np.clip(curv_norm, 0, 1)

        # Inflated geometry and hMT+ centroid
        coords, faces = nib.freesurfer.read_geometry(
            op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.inflated")
        )

        center = coords[wang_hmt_vertices].mean(axis=0)

        # Extent of hMT+ (Wang ROI + functional MT labels of both groups)
        roi_vertices = [wang_hmt_vertices]

        for group in groups:
            lf = op.join(
                bids_path, 'analysis', 'ROIs', 'func_roi',
                'functional_surf_roi', 'group_averages',
                f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
            )

            if op.exists(lf):
                roi_vertices.append(np.asarray(read_label(lf)))

        roi_vertices = np.unique(np.concatenate(roi_vertices))

        extent = np.linalg.norm(
            coords[roi_vertices] - center, axis=1
        ).max()
        hemi_info[hemi] = dict(
            n_vertices=n_vertices, wang_hmt_vertices=wang_hmt_vertices,
            curv_norm=curv_norm, coords=coords, faces=faces,
            center=center, extent=extent,
        )
        # Participant density maps for this hemisphere: (n_participants, n_tracts, n_masked)
        hemi_maps = {t: np.array([density_maps[hemi][sub][t] for sub in density_maps[hemi]])
                     for t in range(len(tract_order))}

        hemi_info[hemi] = dict(
            n_vertices=n_vertices, wang_hmt_vertices=wang_hmt_vertices,
            curv_norm=curv_norm, coords=coords, faces=faces,
            center=center, extent=extent, hemi_maps=hemi_maps,
        )

    # Common radius
    if crop_radius is None:
        crop_radius = max(
            info["extent"] for info in hemi_info.values()
        ) + margin

    print(f"Crop radius used for both hemispheres: {crop_radius:.1f} mm")

    # Crop meshes
    for hemi, info in hemi_info.items():

        coords, faces = info["coords"], info["faces"]

        keep = np.linalg.norm(
            coords - info["center"], axis=1
        ) <= crop_radius

        idx = np.where(keep)[0]

        new_index = -np.ones(info["n_vertices"], dtype=int)
        new_index[idx] = np.arange(len(idx))

        face_mask = keep[faces].all(axis=1)

        info["idx"] = idx
        info["mesh_crop"] = (
            coords[idx],
            new_index[faces[face_mask]]
        )

        info["curv_crop"] = info["curv_norm"][idx]
    for hemi, info in hemi_info.items():
        print(
            f"{hemi}: "
            f"extent={info['extent']:.2f} mm | "
            f"radius={crop_radius:.2f} mm | "
            f"vertices retained={len(info['idx'])}/{info['n_vertices']} "
            f"({100 * len(info['idx']) / info['n_vertices']:.1f}%)"
        )
    # ------------------------------------------------
    # One figure per tract, both hemispheres side by side
    # ------------------------------------------------
    for t, tract in enumerate(tract_order):
        fig = plt.figure(figsize=figsize)

        for h, hemi in enumerate(hemis):
            info = hemi_info[hemi]
            ax = fig.add_subplot(1, 2, h + 1, projection="3d")

            # Participant density on the full surface, then crop
            density_vals = info["hemi_maps"][t][pidx, :]
            surf_map = np.full((info["n_vertices"],), np.nan, dtype=np.float32)
            surf_map[info["wang_hmt_vertices"]] = density_vals

            plotting.plot_surf_stat_map(
                surf_mesh=info["mesh_crop"],
                stat_map=surf_map[info["idx"]],
                hemi="left" if hemi == "L" else "right",
                view="lateral",
                cmap=cmap,
                colorbar=False,                 # one shared colorbar below
                vmin=vmin, vmax=vmax,
                threshold=None,
                bg_map=info["curv_crop"],
                bg_on_data=True,
                darkness=0.6,
                axes=ax, figure=fig,
            )

            # ---- identical scale for both hemispheres ----
            cx, cy, cz = info["center"]
            ax.set_xlim(cx - crop_radius, cx + crop_radius)
            ax.set_ylim(cy - crop_radius, cy + crop_radius)
            ax.set_zlim(cz - crop_radius, cz + crop_radius)
            ax.set_box_aspect((1, 1, 1))

            ax.set_title(f"{hemi} hemisphere", fontsize=18, fontweight="bold")

            # Rasterize the 3D surface only (keeps the SVG small)
            for coll in ax.collections:
                coll.set_rasterized(True)

        # ---- one shared colorbar ----
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=14)
        for lab in cbar.ax.get_yticklabels():
            lab.set_fontweight("bold")

        fig.suptitle(f"{participant} - {tract} (density)",
                     fontsize=20, fontweight="bold")

        # ---- save + close ----
        out_base = op.join(
            img_out_dir,
            f"{participant}_hemi-LR_label-{tract}_desc-density_inflated_{projfrac}frac_cropped"
        )
        for fmt in ("png", "svg"):
            fig.savefig(f"{out_base}.{fmt}", dpi=600, bbox_inches="tight")
        plt.close(fig)

def participant_pred_plot(func_maps, participant, pidx, bids_path, img_out_dir, ylim):
    fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    hemis = ["L", "R"]
    for h, hemi in enumerate(hemis):
        hemi_fs = "lh" if hemi == "L" else "rh"
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
            participant,
            f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
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
        if "EB" in participant:
                localizers = ["ptlocal"]
        elif "NS" in participant:
                localizers = ["mtlocal", "ptlocal"]
        for c, cont in enumerate(localizers):
            # ----------------------------
            # Density Map visualization
            # ----------------------------
            # img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "wb_participants_linearreg",  "surface_pngs", "mean_predicted_func")
            os.makedirs(img_out_dir, exist_ok=True)

            vmin, vmax = ylim 

            # mean_vals = np.nanmean(sub_mean_maps, axis=0)
            hemi_maps = np.array(np.nanmean(np.squeeze(func_maps[hemi][c][pidx]), axis = 0)) #density_maps[hemi][:][t]
            func_vals = hemi_maps #[[group in p for p in participants],:]
            # Build full-surface vector once
            surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
            surf_map[wang_hmt_vertices] = func_vals

            # Output filename (no run index)
            out_base = op.join(
                img_out_dir,
                f"{participant}_hemi-{hemi}_label-{cont}_desc-motionXstationary_inflated"
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
            # plotting.plot_surf_contours(
            #     surf_mesh=infl_surf,
            #     roi_map=func_mt_roi,
            #     levels=[1],
            #     colors=["lightgray"],
            #     linewidths=2.0,
            #     figure=display.figure,
            #     axes=display.axes[0]
            # )

            # before saving: rasterize the 3D surface only
            for ax in display.axes:
                if ax.name == "3d":
                    for coll in ax.collections:
                        coll.set_rasterized(True)

            plt.rcParams["svg.fonttype"] = "none"   # keep text editable (set once, at the top)

            # ---- save + close ----
            for fmt in ("png", "svg"):
                display.savefig(f"{out_base}.{fmt}", dpi=600)
            # display.savefig(out_png, dpi=300)
            plt.close(display.figure)


def participant_pred_cropped_plot(func_maps, participant, pidx, bids_path, img_out_dir, ylim,
                          crop_radius=None, margin=15, cmap="plasma",
                          figsize=(11, 5.5), show_func_mt_boundary=False):

    fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
    hemis = ["L", "R"]
    vmin, vmax = ylim

    os.makedirs(img_out_dir, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"

    # EB --> ptlocal only
    # NS --> mtlocal + ptlocal
    if "EB" in participant:
        localizers = ["ptlocal"]
    elif "NS" in participant:
        localizers = ["mtlocal", "ptlocal"]
    else:
        raise ValueError(f"Cannot determine localizers for {participant}")

    # ------------------------------------------------
    # Per-hemisphere setup: ROI, curvature, geometry,
    # centroid, extent, and participant functional maps
    # ------------------------------------------------
    hemi_info = {}

    for hemi in hemis:

        hemi_fs = "lh" if hemi == "L" else "rh"

        # ----------------------------
        # Load Wang hMT+ ROI
        # ----------------------------
        wang_hmt_path = op.join(
            '/Users', 'ldaumail3', 'Documents', 'research',
            'brain_atlases', 'Wang_2015', 'hmtplus',
            f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
        )

        surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
        n_vertices = len(surf_roi)

        wang_hmt_vertices = np.where(surf_roi > 0)[0]
        print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")

        # ----------------------------
        # Load functional MT ROI
        # ----------------------------
        label_file = op.join(
            bids_path, 'analysis', 'ROIs', 'func_roi',
            'functional_surf_roi', participant,
            f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
        )

        func_mt_vertices = read_label(label_file)

        func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
        func_mt_roi[func_mt_vertices] = 1

        # ----------------------------
        # Curvature background
        # ----------------------------
        curv = nib.freesurfer.read_morph_data(
            op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv")
        )

        curv_norm = (curv - np.percentile(curv, 5)) / (
            np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8
        )
        curv_norm = np.clip(curv_norm, 0, 1)

        # ----------------------------
        # Inflated surface geometry
        # ----------------------------
        coords, faces = nib.freesurfer.read_geometry(
            op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.inflated")
        )

        center = coords[wang_hmt_vertices].mean(axis=0)

        # Extent of hMT+ (Wang ROI + functional MT labels of both groups)
        groups = ["EB", "NS"]
        roi_vertices = [wang_hmt_vertices]

        for group in groups:
            lf = op.join(
                bids_path, 'analysis', 'ROIs', 'func_roi',
                'functional_surf_roi', 'group_averages',
                f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
            )

            if op.exists(lf):
                roi_vertices.append(np.asarray(read_label(lf)))

        roi_vertices = np.unique(np.concatenate(roi_vertices))

        extent = np.linalg.norm(
            coords[roi_vertices] - center, axis=1
        ).max()

        # ----------------------------
        # Participant functional maps
        # One map per localizer
        # ----------------------------
        hemi_maps = {}

        for c, cont in enumerate(localizers):

            hemi_maps[cont] = np.nanmean(
                np.squeeze(func_maps[hemi][c][pidx]),
                axis=0
            )

        hemi_info[hemi] = dict(
            n_vertices=n_vertices,
            wang_hmt_vertices=wang_hmt_vertices,
            func_mt_roi=func_mt_roi,
            curv_norm=curv_norm,
            coords=coords,
            faces=faces,
            center=center,
            extent=extent,
            hemi_maps=hemi_maps
        )

    # ------------------------------------------------
    # Common crop radius: largest hMT+ extent sets
    # the radius for BOTH hemispheres
    # ------------------------------------------------
    if crop_radius is None:
        crop_radius = max(
            info["extent"] for info in hemi_info.values()
        ) + margin

    print(f"Crop radius used for both hemispheres: {crop_radius:.1f} mm")

    # ------------------------------------------------
    # Crop both meshes using the same radius
    # ------------------------------------------------
    for hemi, info in hemi_info.items():

        coords = info["coords"]
        faces = info["faces"]

        keep = np.linalg.norm(
            coords - info["center"], axis=1
        ) <= crop_radius

        idx = np.where(keep)[0]

        new_index = -np.ones(info["n_vertices"], dtype=int)
        new_index[idx] = np.arange(len(idx))

        face_mask = keep[faces].all(axis=1)

        info["idx"] = idx
        info["mesh_crop"] = (
            coords[idx],
            new_index[faces[face_mask]]
        )

        info["curv_crop"] = info["curv_norm"][idx]

        # Cropped functional MT ROI for optional boundary overlay
        info["func_mt_roi_crop"] = info["func_mt_roi"][idx]

    # ------------------------------------------------
    # One figure per localizer:
    # left + right hemispheres side by side
    # ------------------------------------------------
    for cont in localizers:

        fig = plt.figure(figsize=figsize)

        for h, hemi in enumerate(hemis):

            info = hemi_info[hemi]

            ax = fig.add_subplot(
                1, 2, h + 1, projection="3d"
            )

            # ----------------------------
            # Functional map on full surface
            # then crop
            # ----------------------------
            func_vals = info["hemi_maps"][cont]

            surf_map = np.full(
                (info["n_vertices"],),
                np.nan,
                dtype=np.float32
            )

            surf_map[info["wang_hmt_vertices"]] = func_vals

            # ----------------------------
            # Plot cropped surface
            # ----------------------------
            plotting.plot_surf_stat_map(
                surf_mesh=info["mesh_crop"],
                stat_map=surf_map[info["idx"]],
                hemi="left" if hemi == "L" else "right",
                view="lateral",
                cmap=cmap,
                colorbar=False,
                vmin=vmin,
                vmax=vmax,
                threshold=None,
                bg_map=info["curv_crop"],
                bg_on_data=True,
                darkness=0.6,
                axes=ax,
                figure=fig
            )

            # ----------------------------
            # Optional functional MT boundary
            # ----------------------------
            if show_func_mt_boundary:

                plotting.plot_surf_contours(
                    surf_mesh=info["mesh_crop"],
                    roi_map=info["func_mt_roi_crop"],
                    levels=[0.5],
                    colors=["lightgray"],
                    linewidths=2.0,
                    figure=fig,
                    axes=ax
                )

            # ----------------------------
            # Identical scale for both hemispheres
            # ----------------------------
            cx, cy, cz = info["center"]

            ax.set_xlim(cx - crop_radius, cx + crop_radius)
            ax.set_ylim(cy - crop_radius, cy + crop_radius)
            ax.set_zlim(cz - crop_radius, cz + crop_radius)

            ax.set_box_aspect((1, 1, 1))

            ax.set_title(
                f"{hemi} hemisphere",
                fontsize=18,
                fontweight="bold"
            )

            # Rasterize surface only
            for coll in ax.collections:
                coll.set_rasterized(True)

        # ------------------------------------------------
        # Shared colorbar
        # ------------------------------------------------
        sm = ScalarMappable(
            norm=Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap
        )

        cbar = fig.colorbar(
            sm,
            ax=fig.axes,
            shrink=0.6,
            pad=0.02
        )

        cbar.ax.tick_params(labelsize=14)

        for lab in cbar.ax.get_yticklabels():
            lab.set_fontweight("bold")

        # ------------------------------------------------
        # Figure title
        # ------------------------------------------------
        fig.suptitle(
            f"{participant} - {cont} (motionXstationary)",
            fontsize=20,
            fontweight="bold"
        )

        # ------------------------------------------------
        # Save PNG + SVG
        # ------------------------------------------------
        out_base = op.join(
            img_out_dir,
            f"{participant}_hemi-LR_label-{cont}_"
            f"desc-motionXstationary_inflated_cropped"
        )

        for fmt in ("png", "svg"):
            fig.savefig(
                f"{out_base}.{fmt}",
                dpi=600,
                bbox_inches="tight"
            )

        plt.close(fig)
