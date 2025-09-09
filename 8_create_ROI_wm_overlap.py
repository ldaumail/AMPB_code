import argparse
import os.path as op
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = current_dir  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.overlap_masks import overlap_masks

def main(participants_file, bids_path):
    '''
    Intersects MT ROis with white matter mask
    example: python 8_create_ROI_wm_overlap.py --participants_file ./utils/study2_subjects_updated.txt --bids_path '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb'
    '''
    for participant in participants_file:
        input_file1 = op.join(bids_path, 'analysis', 'fs_wm', participant, participant+'_space-T1w_rec-fs_label-WM_mask.nii.gz')
        for mask in ['MT']: #, 'PT'
            for hemi in ['L', 'R']:
                input_file2 = op.join(bids_path, 'analysis', 'functional_vol_roi', participant, participant+'_hemi-'+hemi+'_space-ACPC_label-'+mask+'_mask_dilated.nii.gz')
                input_files = [input_file1, input_file2]
                output_file = op.join(bids_path, 'analysis', 'functional_vol_roi', participant, participant+'_hemi-'+hemi+'_space-ACPC_label-'+mask+'xWM_mask.nii.gz')
                overlap_masks(input_files, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create intersection of white matter and MT/PT masks into a new mask")
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
