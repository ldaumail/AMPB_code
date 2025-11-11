#Loic Daumail
#11/11/2025
#Resample surface density maps from fsnative to fsaverage space
import subprocess
import os
import os.path as op
from pathlib import Path

bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
participants = sorted([p for p in os.listdir(bids_path) if p.startswith("sub-")])
fs_path = op.join(bids_path, 'derivatives', 'freesurfer') #op.join('/Applications', 'freesurfer', '8.0.0-beta', 'subjects')
tract_names = ["MTxLGN", "MTxPT", "MTxSTS1", "MTxPU", "MTxhIP", "MTxFEF", "MTxV1"]
for participant in participants: #participants_file:
    print(f"\n🔹 Processing {participant}")
    # participant = 'sub-EBxGxCCx1986'
    surface_density_path = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps', participant)
    hemisphere = ["L", "R"]
    for h, hemi in enumerate(hemisphere):
        # Example for left hemisphere:
        hemi_fs = "lh" if hemi == "L" else "rh"
        side = "Left" if hemi == "L" else "Right" 
        for tract in tract_names:
            old_name = tract.replace("MT", "MTmask")
            source_density_file = op.join(surface_density_path, f"{participant}_hemi-{hemi}_space-fsnative_label-{side}{old_name}_desc-fsprojdensity0mm.mgh")
            if not Path(source_density_file).is_file():
                print(f"   ⚠️ No density files found for hemi-{hemi}")
                continue
            out_density_file = op.join(surface_density_path, f"{participant}_hemi-{hemi_fs}_space-fsaverage_label-{tract}_desc-fsprojdensity0mm.mgh")


            cmd = ["mri_surf2surf",
            "--srcsubject", participant, 
            "--trgsubject", "fsaverage",
            "--hemi", hemi_fs, 
            "--sval", source_density_file, 
            "--tval", out_density_file ]

            # Run the command
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})
