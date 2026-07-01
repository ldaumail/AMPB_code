#The goal of this script is to:
# 1. convert tracts streamlines into density maps
# 2. assess which tract overlaps with hMT+
import os
import os.path as op
import sys
import re
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path 
from scipy.stats import sem
current_dir = op.dirname(op.abspath(__file__))
project_dir = op.abspath(op.join(current_dir, '..'))  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.streamlines_utils import streamline2dipy_density

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research',
                    'ampb_mt_tractometry_analysis', 'ampb')
participants_file = op.join('/Users','ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb', 'code', 'utils', 'study2_subjects_updated.txt')
with open(participants_file, "r") as f:
    participants = [line.strip().lstrip("/") for line in f if line.strip()]
pyAFQ_path = op.join(bids_path, 'derivatives', 'pyafq', 'wmgmi_wb')
afq_wb_path = op.join(pyAFQ_path, 'afq33-wb_5rounds') ##op.join('/Volumes', 'cos-lab-wpark78', 'LoicDaumail', 'ampb', 'derivatives', 'pyafq', 'wmgmi_wang') #op.join(bids_path, 'derivatives', 'pyafq', 'wmgmi_wang')#
# afq_julich_path = op.join(bids_path, 'derivatives', 'pyAFQ', 'wmgmi_wb', 'afq-wb_julich_10rounds')

#filenames = glob.glob(os.path.join(pyAFQ_path, "afq-wb_TRs_10rounds", "sub-EBxGxCCx1986", "bundles","*.trx"))#

tract_names = {
    "PTR",
    "STR",
    "InferiorLongitudinal", 
    "InferiorFrontooccipital", 
    "SuperiorLongitudinalI",
    "SuperiorLongitudinalII",
    "SuperiorLongitudinalIII", 
    "AnteriorVerticalOccipital", 
    "PosteriorVerticalOccipital", 
    "CallosumOccipital"
    }
n_subj = len(participants)
n_tracts = len(tract_names)
for hemi in {"L", "R"}:
    hemi_afq = "Left" if hemi == "L" else "Right"
    for p, participant in enumerate(participants): 
            
        for t, tract in enumerate(tract_names):
            # participant = 'sub-EBxGxCCx1986'
            # tract_name = 'RightMTmaskxLGN'
            # pyAFQ_path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/pyAFQ/wmgmi/RightMTxLGN'
            tdi_path = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps', participant, 'pyAFQ33_wb')
            os.makedirs(tdi_path, exist_ok=True)

            if "CallosumOccipital" in tract:
                tract_path = Path(os.path.join(afq_wb_path, participant, 'bundles', f"{participant}_ses-concat_acq-HCPdir99_desc-{tract}_tractography.trx"))
            elif any(substring in tract for substring in ["TR"]):
                tract_path = Path(os.path.join(afq_wb_path, participant, 'bundles', f"{participant}_ses-concat_acq-HCPdir99_desc-{hemi}{tract}_tractography.trx"))
            else:
                tract_path = Path(os.path.join(afq_wb_path, participant, 'bundles', f"{participant}_ses-concat_acq-HCPdir99_desc-{hemi_afq}{tract}_tractography.trx"))
            
            if tract_path.exists():
                if "CallosumOccipital" in tract:
                    tract_tdi_map = os.path.join(tdi_path, participant+'_ses-concat_desc-'+f"{tract}"+'_tdi_map_5rounds.nii.gz')
                else:
                    tract_tdi_map = os.path.join(tdi_path, participant+'_ses-concat_desc-'+f"{hemi_afq}{tract}"+'_tdi_map_5rounds.nii.gz')
                template = op.join(afq_wb_path, participant, participant+'_ses-concat_acq-HCPdir99_b0ref.nii.gz')

                streamline2dipy_density(tract_path, template, tract_tdi_map)



##--------------------------------------------
# Quantify overlap between whole brain tracts and julich targetted tracts using endpoint density maps
##--------------------------------------------
import os
import os.path as op
import pandas as pd
import nibabel as nib
import numpy as np
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research',
                    'ampb_mt_tractometry_analysis', 'ampb')
density_dir = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps')
participants = sorted([p for p in os.listdir(density_dir) if p.startswith("sub-")])
# participants.remove("sub-NSxLxQUx1953")
rows = []
TR_tract_names = {"InferiorLongitudinal", "InferiorFrontooccipital", "SuperiorLongitudinalI","SuperiorLongitudinalII","SuperiorLongitudinalIII", "AnteriorVerticalOccipital", "PosteriorVerticalOccipital", "CallosumOccipital", "PTR", "STR"} #{"PTR", "STR"}
targ_tract_names = {"MTxLGNxPU", "MTxFEF", "MTxPTxSTS1"}

for hemi in {"L", "R"}:
    hemi_afq = "Left" if hemi == "L" else "Right"
    for p, participant in enumerate(participants): 
        for tr in TR_tract_names:
            for targ in targ_tract_names:
                if "CallosumOccipital" in tr:
                    TR_path = op.join(density_dir, participant, 'pyAFQ33_wb', participant+'_ses-concat_desc-' + f"{tr}" + '_tdi_map_5rounds.nii.gz')
                else:
                    TR_path = op.join(density_dir, participant, 'pyAFQ33_wb', participant+'_ses-concat_desc-' + f"{hemi_afq}{tr}" + '_tdi_map_5rounds.nii.gz')
                    
                targ_path = op.join(density_dir, participant, 'wang_MT', participant+'_ses-concat_desc-' + f"wang{hemi_afq}{targ}" + '_tdi_map2.nii.gz')
        
                TR_map = nib.load(TR_path).get_fdata() 
                targ_map = nib.load(targ_path).get_fdata() 

                overlap_val = np.sum((TR_map * targ_map) > 0)
                percent_TR = np.sum((TR_map * targ_map) > 0) * 100 / np.sum(TR_map > 0) 
                percent_targ = np.sum((TR_map * targ_map) > 0) * 100 / np.sum(targ_map > 0)

                rows.append({
                    "hemi": hemi,
                    "participant": participant,
                    "TR_tract": tr,
                    "targ_tract": targ,
                    "overlap": overlap_val,
                    "percent_TR": percent_TR,
                    "percent_Targ": percent_targ,

                })

overlap_df = pd.DataFrame(rows)


#Plot
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

for ax, hemi in zip(axes, ["L", "R"]):

    tmp = overlap_df[overlap_df["hemi"] == hemi]

    pivot = (
        tmp.groupby(["TR_tract", "targ_tract"])["percent_Targ"]
        .mean()
        .unstack()
    )

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        ax=ax
    )

    ax.set_title(f"{hemi} hemisphere")
    ax.set_xlabel("Target tract")
    ax.set_ylabel("Atlas tract")
saveDir = op.join(bids_path, 'analysis', 'plots')
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, f"pyafq3_5rounds_tracts_overlap_percentTarg.png"),
                dpi=300, bbox_inches='tight')
plt.tight_layout()

#Plot 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# choose metric to plot
metric = "percent_Targ"   # or "percent_Targ" or "overlap"

TR_order = ["PTR", "STR"]
targ_order = ["MTLGNxPU", "MTFEF", "MTPTxSTS1"]

for hemi in ["L", "R"]:

    hemi_df = overlap_df[overlap_df["hemi"] == hemi]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 8),
        sharex=True
    )

    for ax, tr in zip(axes, TR_order):

        df_tr = hemi_df[hemi_df["TR_tract"] == tr]

        # Mean ± SEM bars
        sns.barplot(
            data=df_tr,
            x="targ_tract",
            y=metric,
            order=targ_order,
            errorbar="se",
            color="lightgray",
            edgecolor="black",
            ax=ax
        )

        # Individual participants
        sns.stripplot(
            data=df_tr,
            x="targ_tract",
            y=metric,
            order=targ_order,
            color="black",
            size=6,
            jitter=0.15,
            alpha=0.7,
            ax=ax
        )

        ax.set_title(f"{hemi} hemisphere - {tr}")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_ylim((0,50))

    sns.despine()
    plt.tight_layout()

    saveDir = op.join(bids_path, 'analysis', 'plots')
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(op.join(saveDir, f"{hemi}_pyafq3_15rounds_tracts_overlap_with_thalamic_radiations_percentTarg.png"),
                 dpi=300, bbox_inches='tight')
    plt.show()


#--------------------------------------------------
# Quantify overlap between MT-PUxLGN and MT-PTxSTS1
#--------------------------------------------------

import os
import pandas as pd
density_dir = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps')
participants = sorted([p for p in os.listdir(density_dir) if p.startswith("sub-")])
rows = []
targ_tract_names = {"MTLGNxPU", "MTFEF", "MTPTxSTS1"}

for hemi in {"L", "R"}:
    for p, participant in enumerate(participants): 
        # for tr in TR_tract_names:
            # for targ in targ_tract_names:
            lgn_pu_path = op.join(density_dir, participant, 'pyAFQ_wb', participant+'_ses-concat_desc-' + f"{hemi}MTLGNxPU" + '_15rounds_tdi_map.nii.gz')
            pt_sts1_path = op.join(density_dir, participant, 'pyAFQ_wb', participant+'_ses-concat_desc-' + f"{hemi}MTPTxSTS1" + '_15rounds_tdi_map.nii.gz')
    
            lgn_pu_map = nib.load(lgn_pu_path).get_fdata() 
            pt_sts1_map = nib.load(pt_sts1_path).get_fdata() 

            overlap_val = np.sum((lgn_pu_map * pt_sts1_map) > 0)
            percent_lgn_pu = np.sum((lgn_pu_map * pt_sts1_map) > 0) * 100 / np.sum(lgn_pu_map > 0) 


            rows.append({
                "hemi": hemi,
                "participant": participant,
                # "TR_tract": tr,
                # "targ_tract": targ,
                "overlap": overlap_val,
                "percent_MT-LGNxPU": percent_lgn_pu,
                # "percent_Targ": percent_targ,

            })

overlap_df = pd.DataFrame(rows)

#Plot 

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

metric = "percent_MT-LGNxPU"

fig, axes = plt.subplots(
    1, 2,
    figsize=(8, 5),
    sharey=True
)

for ax, hemi in zip(axes, ["L", "R"]):

    hemi_df = overlap_df[overlap_df["hemi"] == hemi]

    # mean ± SEM bar
    sns.barplot(
        data=hemi_df,
        x=[""] * len(hemi_df),   # single category
        y=metric,
        errorbar="se",
        color="lightgray",
        edgecolor="black",
        ax=ax
    )

    # individual participants
    sns.stripplot(
        data=hemi_df,
        x=[""] * len(hemi_df),
        y=metric,
        color="black",
        size=6,
        jitter=0.15,
        alpha=0.7,
        ax=ax
    )

    ax.set_title(f"{hemi} hemisphere")
    ax.set_xlabel("")
    ax.set_ylabel("% overlap of MTLGNxPU with MTPTxSTS1")
    ax.set_ylim(0, 100)

sns.despine()
plt.tight_layout()

saveDir = op.join(bids_path, "analysis", "plots")
os.makedirs(saveDir, exist_ok=True)

plt.savefig(
    op.join(
        saveDir,
        "pyafq3_15rounds_overlap_MTLGNxPU_MTPTxSTS1_by_hemi.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()