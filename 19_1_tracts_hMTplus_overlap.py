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
filenames = glob.glob(os.path.join(pyAFQ_path, "afq-Left", "sub-EBxGxCCx1986", "bundles","*.trx"))#

# tract_names = [
#     re.search(r"desc-(.*?)_tractography", f).group(1)
#     for f in filenames
# ]

# tract_names = ['CallosumMotor',
#                'CallosumTemporal']
# tract_names = ['CallosumOrbital',
#  'RightInferiorFrontooccipital',
#  'RightArcuate',
#  'RightSuperiorLongitudinal',
#  'RightUncinate',
#  'RightInferiorLongitudinal',
#  'CallosumOccipital',
#  'LeftPosteriorArcuate',
#  'RightPosteriorArcuate',
#  'CallosumSuperiorFrontal',
#  'RightCingulumCingulate',
#  'CallosumAnteriorFrontal',
#  'CallosumPosteriorParietal',
#  'LeftVerticalOccipital',
#  'LeftUncinate',
#  'CallosumSuperiorParietal',
#  'LeftInferiorFrontooccipital',
#  'LeftInferiorLongitudinal',
#  'RightCorticospinal',
#  'RightAnteriorThalamic',
#  'RightVerticalOccipital',
#  'LeftCingulumCingulate',
#  'LeftArcuate',
#  'LeftCorticospinal',
#  'LeftAnteriorThalamic',
#  'LeftSuperiorLongitudinal']


tract_names =  {"CallosumAnteriorFrontal",
    "CallosumMotor",
    "CallosumOccipital",
    "CallosumOrbital",
    "CallosumPosteriorParietal",
    "CallosumSuperiorFrontal",
    "CallosumSuperiorParietal",
    "CallosumTemporal",
    "LeftAnteriorThalamic",
    "LeftArcuate",
    "LeftCingulumCingulate",
    "LeftCorticospinal",
    "LeftInferiorFrontooccipital",
    "LeftInferiorLongitudinal",
    "LeftPosteriorArcuate",
    "LeftSuperiorLongitudinal",
    "LeftUncinate",
    "LeftVerticalOccipital",
    "RightAnteriorThalamic",
    "RightArcuate",
    "RightCingulumCingulate",
    "RightCorticospinal",
    "RightInferiorFrontooccipital",
    "RightInferiorLongitudinal",
    "RightPosteriorArcuate",
    "RightSuperiorLongitudinal",
    "RightUncinate",
    "RightVerticalOccipital"}
n_subj = len(participants)
n_tracts = len(tract_names)
rows = []
# nnz_count = np.full((n_tracts), np.zeros)
nnz_count = {}
for p, participant in enumerate(participants): 

    hmt_path = op.join(bids_path, 'analysis', 'ROIs', 'wang_space-ACPC_rois',participant, participant+'_hemi-L_space-ACPC_label-MT_mask_dilated.nii.gz')
         
    for t, tract in enumerate(tract_names):
        # participant = 'sub-EBxGxCCx1986'
        # tract_name = 'RightMTmaskxLGN'
        # pyAFQ_path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/pyAFQ/wmgmi/RightMTxLGN'
        tdi_path = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps', participant, 'pyAFQ_default_atlas')
        os.makedirs(tdi_path, exist_ok=True)
    
        tract_path = Path(os.path.join(pyAFQ_path, f"afq-Left", participant, 'bundles', participant+'_ses-concat_acq-HCPdir99_desc-' + tract + '_tractography.trx'))
        
        if tract_path.exists():
            tract_tdi_map = os.path.join(tdi_path, participant+'_ses-concat_desc-' + tract + '_tdi_map.nii.gz')
            template = op.join(pyAFQ_path, f"afq-Left", participant, participant+'_ses-concat_acq-HCPdir99_b0ref.nii.gz')

            streamline2dipy_density(tract_path, template, tract_tdi_map)

            #Load data 
            MT_map = nib.load(hmt_path).get_fdata() 
            tract_map = nib.load(tract_tdi_map).get_fdata() 
            

            overlap_val = np.sum((MT_map * tract_map) > 0)

            if overlap_val > 0:
                nnz_count[t] = nnz_count.get(t, 0) +1

            # ✅ store structured row
            rows.append({
                "participant": participant,
                "tract": tract,
                "overlap": overlap_val
            })

# 👉 convert to DataFrame
overlap_df = pd.DataFrame(rows)

summary_df = (
    overlap_df.groupby(["tract"])["overlap"]
    .agg([
        "mean", 
        "std", 
        sem, 
        lambda x: (x > 0).sum()
    ])
    .reset_index()
    .rename(columns={
        "mean": "Mean", 
        "std": "SD", 
        "sem": "SEM", 
        "<lambda_0>": "N_NonZero"
    })
)