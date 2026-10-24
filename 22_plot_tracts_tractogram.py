
#======================================================
# Save in vector format
#======================================================

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dipy.io.streamline import load_tractogram


def load_streamlines(path):
    """Load a .trx/.trk/.tck file into RASMM streamlines."""
    sft = load_tractogram(path, reference="same")
    sft.to_rasmm()
    return sft.streamlines


def hemi_prefix(tract, hemi):
    """PTR tracts use 'L'/'R'; all others use 'Left'/'Right' (as in your code)."""
    if "PTR" in tract:
        return hemi
    return "Left" if hemi == "L" else "Right"


def project_lateral(streamlines, hemi, step=1):
    """Project RASMM streamlines to 2D for a lateral view.
    Horizontal = y (flipped for L so anterior is on the left), vertical = z."""
    flip = -1 if hemi == "L" else 1
    return [np.column_stack([flip * np.asarray(s)[::step, 1],
                             np.asarray(s)[::step, 2]])
            for s in streamlines]


def plot_tracts_svg(participant, tract_names, afq_TR_path,
                    whole_brain_path=None, n_background=20000,
                    hemis=("L", "R"), cmap="tab10", line_width=1.0,
                    bg_opacity=0.15, bg_color="0.5", bg_line_width=0.3,
                    figsize=(8, 6), step=2, hemi_only_background=True,
                    save_path=None, fmt="svg", seed=0, show=False):
    """Save one vector figure per (hemisphere, tract).
    step: keep every Nth point of each streamline (reduces file size)."""

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Keep text as editable text in SVG (no text drawn here, but harmless)
    plt.rcParams["svg.fonttype"] = "none"

    # One fixed color per tract, shared by both hemispheres
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(len(tract_names), 2)))
    color_map = {t: tuple(c[:3]) for t, c in zip(tract_names, colors)}

    # Whole-brain tractogram: load and subsample once
    wb_sub, wb_meanx, center = None, None, None
    if whole_brain_path is not None:
        wb = load_streamlines(whole_brain_path)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(wb), size=min(n_background, len(wb)),
                         replace=False)
        wb_sub = [wb[i] for i in idx]
        wb_meanx = np.array([np.asarray(s)[:, 0].mean() for s in wb_sub])
        pts = np.vstack([np.asarray(s) for s in wb_sub])
        center = (pts.min(0) + pts.max(0)) / 2

    for hemi in hemis:
        # Background for this view (optionally only the near hemisphere)
        bg = wb_sub
        if wb_sub is not None and hemi_only_background:
            keep = (wb_meanx < center[0]) if hemi == "L" else (wb_meanx > center[0])
            bg = [s for s, k in zip(wb_sub, keep) if k]
        bg_2d = project_lateral(bg, hemi, step) if bg else None

        # Shared axis limits so every tract figure has the same framing
        if bg_2d:
            allpts = np.vstack(bg_2d)
            xlim = (allpts[:, 0].min(), allpts[:, 0].max())
            ylim = (allpts[:, 1].min(), allpts[:, 1].max())
        else:
            xlim = ylim = None

        for tract in tract_names:
            f = op.join(
                afq_TR_path, participant, "bundles",
                f"{participant}_ses-concat_acq-HCPdir99_desc-"
                f"{hemi_prefix(tract, hemi)}{tract}_tractography.trx"
            )
            if not op.exists(f):
                print(f"Missing: {f}")
                continue
            sl_2d = project_lateral(load_streamlines(f), hemi, step)

            fig, ax = plt.subplots(figsize=figsize)

            if bg_2d:
                ax.add_collection(LineCollection(
                    bg_2d, colors=bg_color, linewidths=bg_line_width,
                    alpha=bg_opacity, rasterized=True))   # raster inside SVG
            ax.add_collection(LineCollection(
                sl_2d, colors=[color_map[tract]], linewidths=line_width))

            if xlim is not None:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            else:
                ax.autoscale()
            ax.set_aspect("equal")
            ax.axis("off")

            if save_path:
                out = op.join(save_path,
                              f"{participant}_hemi-{hemi}_label-{tract}.{fmt}")
                fig.savefig(out, format=fmt, bbox_inches="tight",
                            transparent=True, dpi=300)  # dpi sets raster background quality
            if show:
                plt.show()
            plt.close(fig)

    return color_map

#================================================================================================
# Whole Brain plots
#================================================================================================
tract_list = list([
'PTR', 
'InferiorLongitudinal',
'InferiorFrontooccipital',
'SuperiorLongitudinal',
'VerticalOccipital',
'Arcuate',
'PosteriorArcuate',
    ])    # or any list of tract names
participant = "sub-NSxLxATx1954"
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research',
                    'ampb_mt_tractometry_analysis', 'ampb')
afq_TR_path = op.join(bids_path, 'derivatives', 'pyAFQ', 'wmgmi_wb', 'afq3beta-wb_6rounds')

plot_tracts_svg(
    participant=participant,
    tract_names=tract_list,
    afq_TR_path=afq_TR_path,
    whole_brain_path=op.join(afq_TR_path, participant, 'tractography', f"{participant}_ses-concat_acq-HCPdir99_tractography.trx"),
    save_path=op.join(bids_path, "analysis", "plots", "tracts_svg"),
)

#================================================================================================
# Targetted plots
#================================================================================================

import re

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dipy.io.streamline import load_tractogram


def load_streamlines(path):
    """Load a .trx/.trk/.tck file into RASMM streamlines."""
    sft = load_tractogram(path, reference="same")
    sft.to_rasmm()
    return sft.streamlines


def hemi_prefix(tract, hemi):
    """PTR tracts use 'L'/'R'; all others use 'Left'/'Right' (as in your code)."""
    if "PTR" in tract:
        return hemi
    return "Left" if hemi == "L" else "Right"


def project_lateral(streamlines, hemi, step=1):
    """Project RASMM streamlines to 2D for a lateral view.
    Horizontal = y (flipped for L so anterior is on the left), vertical = z."""
    flip = -1 if hemi == "L" else 1
    return [np.column_stack([flip * np.asarray(s)[::step, 1],
                             np.asarray(s)[::step, 2]])
            for s in streamlines]

def plot_tracts_svg(participant, tract_names, afq_TR_path,
                    whole_brain_path, n_background=20000,
                    hemis=("L", "R"), cmap="tab10", line_width=1.0,
                    bg_opacity=0.15, bg_color="0.5", bg_line_width=0.3,
                    figsize=(8, 6), step=2, hemi_only_background=True,
                    save_path=None, fmt="svg", seed=0, show=False):
    """Save one vector figure per (hemisphere, tract).
    step: keep every Nth point of each streamline (reduces file size)."""

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Keep text as editable text in SVG (no text drawn here, but harmless)
    plt.rcParams["svg.fonttype"] = "none"

    # One fixed color per tract, shared by both hemispheres
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, max(len(tract_names), 2)))
    color_map = {t: tuple(c[:3]) for t, c in zip(tract_names, colors)}

    for hemi in hemis:
        for tract in tract_names:
            # Whole-brain tractogram: load and subsample once
            # whole_brain_path=op.join(afq_TR_path, f"afq-{hemi_prefix(tract, hemi)}{tract}", participant, 'tractography', f"{participant}_ses-concat_acq-HCPdir99_tractography.trx")
            wb_sub, wb_meanx, center = None, None, None

            wb = load_streamlines(whole_brain_path)
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(wb), size=min(n_background, len(wb)),
                                replace=False)
            wb_sub = [wb[i] for i in idx]
            wb_meanx = np.array([np.asarray(s)[:, 0].mean() for s in wb_sub])
            pts = np.vstack([np.asarray(s) for s in wb_sub])
            center = (pts.min(0) + pts.max(0)) / 2
            # Background for this view (optionally only the near hemisphere)
            bg = wb_sub
            if wb_sub is not None and hemi_only_background:
                keep = (wb_meanx < center[0]) if hemi == "L" else (wb_meanx > center[0])
                bg = [s for s, k in zip(wb_sub, keep) if k]
            bg_2d = project_lateral(bg, hemi, step) if bg else None

            # Shared axis limits so every tract figure has the same framing
            if bg_2d:
                allpts = np.vstack(bg_2d)
                xlim = (allpts[:, 0].min(), allpts[:, 0].max())
                ylim = (allpts[:, 1].min(), allpts[:, 1].max())
            else:
                xlim = ylim = None


            tract_mask_name = re.sub(r"^MTx", "MTmaskx", tract) 
            f = op.join(
                afq_TR_path, f"afq-{hemi_prefix(tract, hemi)}{tract}", participant, "bundles",
                f"{participant}_ses-concat_acq-HCPdir99_desc-"
                f"{hemi_prefix(tract, hemi)}{tract_mask_name}_tractography.trx"
            )
            if not op.exists(f):
                print(f"Missing: {f}")
                continue
            sl_2d = project_lateral(load_streamlines(f), hemi, step)

            fig, ax = plt.subplots(figsize=figsize)

            if bg_2d:
                ax.add_collection(LineCollection(
                    bg_2d, colors=bg_color, linewidths=bg_line_width,
                    alpha=bg_opacity, rasterized=True))   # raster inside SVG
            ax.add_collection(LineCollection(
                sl_2d, colors=[color_map[tract]], linewidths=line_width))

            if xlim is not None:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            else:
                ax.autoscale()
            ax.set_aspect("equal")
            ax.axis("off")

            if save_path:
                out = op.join(save_path,
                              f"{participant}_hemi-{hemi}_label-{tract}_wb.{fmt}")
                fig.savefig(out, format=fmt, bbox_inches="tight",
                            transparent=True, dpi=300)  # dpi sets raster background quality
            if show:
                plt.show()
            plt.close(fig)

    return color_map



tract_list = list(['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF'])    # or any list of tract names
participant = "sub-NSxLxATx1954"
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research',
                    'ampb_mt_tractometry_analysis', 'ampb')
afq_TR_path = op.join('/Volumes', 'cos-lab-wpark78','LoicDaumail', 'ampb', 'derivatives', 'pyafq', 'wmgmi_targ', 'pyAFQ3beta')
 
plot_tracts_svg(
    participant=participant,
    tract_names=tract_list,
    afq_TR_path=afq_TR_path,
    whole_brain_path=op.join(bids_path, 'derivatives', 'pyAFQ', 'wmgmi_wb', 'afq3beta-wb_6rounds', participant, 'tractography', f"{participant}_ses-concat_acq-HCPdir99_tractography.trx"),
    save_path=op.join(bids_path, "analysis", "plots", "tracts_svg"),
)