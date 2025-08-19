import argparse
import os.path as op
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = current_dir  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.create_fs_wmmask import create_fs_wmmask

def main(participants_file, bids_path):
    for participant in participants_file:
        fs_dir = op.join(bids_path, 'derivatives', 'freesurfer', participant)
        target_brain = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'anat', participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
        target_sample = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'ses-concat', 'dwi', participant+'_ses-concat_acq-HCPdir99_space-ACPC_desc-brain_mask.nii.gz')
        output_fname = op.join(bids_path, 'analysis', 'fs_wm', participant, participant+'_space-T1w_rec-fs_label-WM_mask.nii.gz')
        os.makedirs(op.join(bids_path, 'analysis', 'fs_wm', participant), exist_ok=True)
        create_fs_wmmask(fs_dir, target_brain, target_sample, output_fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create streamline density maps")
    parser.add_argument(
        "--participants_file",
        type=str,
        required=True,
        help="Path to a text file containing participant IDs (one per line)."
    )
    parser.add_argument(
        "--bids_path",
        type=str,
        required=True,
        help="Name of the tract as written in bundle file name"
    )
    args = parser.parse_args()
    # Read participants from file
    with open(args.participants_file, "r") as f:
        participants = [line.strip() for line in f if line.strip()]

    main(participants, args.bids_path)