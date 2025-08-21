import os
import os.path as op
import sys
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = current_dir  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.concat_dwi import concat_sessions
from utils.resample_file import resample_file

def main(participants_file, bids_path):

    for participant in participants_file:
        # participant = 'sub-NSxGxBAx1970'
        qsiprep_dir =  op.join(bids_path, 'derivatives', 'qsiprep', participant)
        out_path = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'ses-concat', 'dwi')
        os.makedirs(out_path, exist_ok=True)
        out_prefix =  op.join(out_path, participant+'_ses-concat_acq-HCPdir99_space-ACPC_desc-preproc_dwi') 
        if os.path.exists(op.join(out_prefix+'.nii.gz')):
            print(f"File exists: {op.join(out_prefix+'.nii.gz')}")
        else:
            concat_sessions(qsiprep_dir, out_prefix)
            print(f"Source file not found, creating it now: {op.join(out_prefix+'.nii.gz')}")
        

        #Create anatomical brain mask with diffusion resolution = aligned to anatomical but with diffusion resolution
        in_brainmask =  op.join(qsiprep_dir, 'anat', participant+'_space-ACPC_desc-brain_mask.nii.gz')
        target_file =  op.join(qsiprep_dir, 'ses-04', 'dwi', participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-brain_mask.nii.gz')
        out_brainmask = os.path.join(out_path, participant+'_ses-concat_acq-HCPdir99_space-ACPC_desc-brain_mask.nii.gz')
        
        if os.path.exists(out_brainmask):
             print(f"Source file exists: {out_brainmask}")
        else: 
            # copy and rename brain mask
            resample_file(in_brainmask, target_file, out_brainmask, interpolator = "linear")
            print(f"Source file not found, creating it now: {out_brainmask}")

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
