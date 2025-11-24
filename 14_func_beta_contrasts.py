#Perform contrasts on the beta data

import nibabel as nib
import os.path as op
import os
import pandas as pd
import numpy as np
import subprocess

# -----------------
# 1. Load event files and compute contrasts per hemisphere
# -----------------
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research',
                    'ampb_mt_tractometry_analysis', 'ampb')
func_dir = op.join(bids_path, 'analysis', 'fMRI_data')
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')

# --- Define condition codes ---
conditions = {"motion": 1, "silent": 2, "stationary": 3}

# --- Define contrasts (positive - negative) ---
contrasts = {
    "motionXstationary": ("motion", "stationary")
} #"motionXsilent": ("motion", "silent")

designs = {"mtlocal", "ptlocal"}
data_type = {"pval", "tstat"}

# --- Loop through participants ---
for participant in sorted(os.listdir(func_dir)):
    if not participant.startswith("sub-"):
        continue
    #participant = 'sub-EBxGxCCx1986'
    print(f"\n🔹 Processing {participant}")
    glm_dir = op.join(func_dir, participant, "glm")

    # -----------------
    # Loop by hemisphere
    # -----------------
    for hemi in ["L", "R"]:
        print(f"   🧩 Hemisphere: {hemi}")
        hemi_fs = "lh" if hemi == "L" else "rh"
        # # -----------------
        # # 2. Compute contrasts for this hemisphere
        # # -----------------
        for contrast_name, (pos_cond, neg_cond) in contrasts.items():

            for design in designs: 
                for map in data_type:  
                    # -------------------------
                    #Save it in fsaverage space
                    # -------------------------
                    surf_map_dir = op.join(glm_dir, "contrasts")
                    os.makedirs(surf_map_dir, exist_ok=True)
                    source_fsnative_file = op.join(glm_dir, f"{participant}_task-{design}_hemi-{hemi}_space-fsnative_desc-{contrast_name}_{map}.mgz")
                    if not os.path.exists(source_fsnative_file): 
                        continue
                    out_fsaverage_file = op.join(surf_map_dir, f"{participant}_task-{design}_hemi-{hemi}_space-fsaverage_desc-{contrast_name}_{map}.mgz")
                    

                    cmd = ["mri_surf2surf",
                    "--srcsubject", participant, 
                    "--trgsubject", "fsaverage",
                    "--hemi", hemi_fs, 
                    "--sval", source_fsnative_file, 
                    "--tval", out_fsaverage_file ]

                    # Run the command
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})

print("\n✅ All contrasts computed and saved per hemisphere.")

