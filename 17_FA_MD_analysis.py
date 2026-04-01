#Loic Daumail 04/01/2026
#FA analysis across tracts

import os.path as op
import os
import nibabel as nib
import numpy as np

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
pyAFQ_dir = op.join(bids_path, "derivatives", "pyAFQ", "wmgmi_wang")
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
# -----------------------
#1 Generate endpoint densities array
#-------------------------

# ✅ Fixed tract order (keep consistent across subjects!)
tract_order = ['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF'] #'MTxLGNxPU', 'MTxPTxSTS1', 
participants_list = op.join(bids_path, 'code', 'utils', 'study2_subjects_updated.txt')
with open(participants_list, 'r') as f: #read file and create file object that you can read line by line iteratively
    participants = sorted([line.strip() for line in f if line.strip()])  #line.strip() for line in f = reads line by line and removes white spaces "if line.strip()"" removes empty lines

hemis = ["L", "R"]

# Initialize storage dictionary
density_data = {hemi: [] for hemi in hemis}

for participant in participants:
    if not participant.startswith("sub-"):
        continue
    print(f"\n🔹 Participant: {participant}")
    # -----------------
    # Loop by hemisphere
    # -----------------
    for hemi in hemis:
        hemi_pyAFQ = "Left" if hemi == "L" else "Right"
        # for tract in ['MTmaskxLGN', 'MTmaskxPT', 'MTmaskxSTS1', 'MTmaskxPU', 'MTmaskxFEF', 'MTmaskxhIP', 'MTmaskxV1']:
        print(f"   🧩 Hemisphere: {hemi}")
        hemi_fs = "lh" if hemi == "L" else "rh"
        # subj_masks = []
        subj_fa = []

        # Loop through *tracts in fixed order*
        for tract in tract_order:
            fa_file_path = op.join(pyAFQ_dir,f"afq-{hemi_pyAFQ}{tract}", participant, 'models', f"{participant}_ses-concat_acq-HCPdir99_model-dki_param-fa_dwimap.nii.gz")
            density_file_path = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps', participant, 'wang_MT', f"{participant}_ses-concat_desc-wang{hemi_pyAFQ}{tract}_tdi_map2.nii.gz")
            # Load the file

            density_img = nib.load(density_file_path)
            density_data = density_img.get_fdata().astype(np.float32)
            tract_mask = density_data > 0
            # subj_masks.append(tract_mask)
            fa_img = nib.load(fa_file_path)
            fa_data = fa_img.get_fdata().astype(np.float32)
            subj_fa.append(fa_data)



        # Stack into one array: shape (n_tracts, n_vertices)
        subj_fa = np.stack(subj_fa, axis=0)  # (7, n_vertices)
        fa_data[hemi].append(subj_fa)
        # subj_densities = np.stack(subj_densities, axis=0)  # (7, n_vertices)
        # density_data[hemi].append(subj_densities)

# for i, arr in enumerate(density_data[hemi]):
#     print(f"{hemi} element {i}: shape = {arr.shape}")
    
# Convert to numpy arrays
for hemi in hemis:
    density_data[hemi] = np.squeeze(np.stack(density_data[hemi], axis=0))  # (n_subjects, n_tracts, n_vertices)
    fa_data[hemi] = np.squeeze(np.stack(fa_data[hemi], axis=0))  # (n_subjects, n_tracts, n_vertices)
    print(f"✅ {hemi}-hemisphere shape: {density_data[hemi].shape}")