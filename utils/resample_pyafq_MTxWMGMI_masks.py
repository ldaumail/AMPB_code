import ants
import os
import os.path as op
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = op.abspath(op.join(current_dir, '..'))   # main_script.py is inside project/
sys.path.append(project_dir)
from utils.resample_file import resample_file

hemis = ["L", "R"]
bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
participants_file = op.join(bids_path, 'code', 'utils', 'study2_subjects_updated.txt')
with open(participants_file, "r") as f:
    participants = [line.strip() for line in f if line.strip()] 

for participant in participants:
    target_file = op.join(
        bids_path,
        "derivatives",
        "qsiprep",
        participant,
        "anat",
        f"{participant}_space-ACPC_desc-preproc_T1w.nii.gz"
    )

    for hemi in hemis:
        func_mt_path = op.join(
            bids_path,
            "analysis",
            "ROIs",
            "func_roi",
            "functional_vol_roi",
            participant,
            f"{participant}_hemi-{hemi}_space-ACPC_label-MTxWMGMI_mask.nii.gz"
        )

        out_dir = op.join(
            bids_path,
            "analysis",
            "ROIs",
            "func_roi",
            "functional_vol_roi",
            participant
        )
        os.makedirs(out_dir, exist_ok=True)

        output_file = op.join(
            out_dir,
            f"{participant}_hemi-{hemi}_space-ACPC_label-MTxWMGMI_mask_T1wresampled.nii.gz"
        )

        resample_file(
            input_file=func_mt_path,
            target_file=target_file,
            output_file=output_file,
            interpolator="linear" #"nearestNeighbor"  # IMPORTANT for masks
        )

        print(f"Resampled {participant} {hemi}")
