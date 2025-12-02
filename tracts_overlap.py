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

