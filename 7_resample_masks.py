#Resample ROI masks to dwi data resolution
import os
import os.path as op
import argparse
import sys
import argparse
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = current_dir  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.resample_file import resample_file


def main(participants_file, bids_path):

    for participant in participants_file:
        target_file =  op.join(bids_path, 'derivatives', 'qsiprep', participant, 'ses-concat', 'dwi', participant+'_ses-concat_acq-HCPdir99_space-ACPC_desc-brain_mask.nii.gz')
        
        mask_name = ['lhV1', 'lhPT', 'lhSTS1', 'lhLGN', 'lhPO', 'lhFEF', 'lhPU', 'lhhIP',
                    'rhV1', 'rhPT', 'rhSTS1', 'rhLGN', 'rhPO', 'rhFEF', 'rhPU', 'rhhIP']
        
        for mask in mask_name:
            hemi  = re.sub(".*(lh|rh).*", "\\1", mask)
            hemi  = "L" if hemi == "lh" else "R"
            roi = re.sub(".*(?:lh|rh)(.*)", "\\1", mask)
            input_file = op.join(bids_path, 'analysis', 'julich_space-ACPC_rois', participant, 'ses-concat', 'anat', participant+'_hemi-'+hemi+'_space-ACPC_desc-'+roi+'_mask.nii.gz')
            output_file = op.join(bids_path, 'analysis', 'julich_space-ACPC_rois', participant, 'ses-concat', 'anat', participant+'_hemi-'+hemi+'_space-ACPC_label-'+roi+'_mask.nii.gz')
            resample_file(input_file, target_file, output_file, interpolator = "linear")
        
        func_mask_name = ['MT', 'PT'] #
        for mask in func_mask_name:
            for hemi in ['L', 'R']:
                input_file = op.join(bids_path, 'analysis', 'functional_vol_roi', participant, participant+'_hemi-'+hemi+'_space-ACPC_label-'+mask+'_mask_dilated.nii.gz')
                output_file = op.join(bids_path, 'analysis', 'functional_vol_roi', participant, participant+'_hemi-'+hemi+'_space-ACPC_label-'+mask+'_mask_dilated.nii.gz')
                resample_file(input_file, target_file, output_file, interpolator = "linear")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample mask to target resolution for a list of participants.")
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

    main(participants_file = participants, bids_path = args.bids_path)



# brain_mask = nib.load('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-EBxGxCCx1986/ses-concat/dwi/sub-EBxGxCCx1986_ses-concat_acq-HCPdir99_space-ACPC_desc-preproc_dwi.nii.gz')
# print(brain_mask.header)
# import nibabel as nib
# mt_roi = nib.load('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/functional_vol_roi/sub-EBxGxCCx1986/sub-EBxGxCCx1986_hemi-L_space-ACPC_label-MT_mask_dilated.nii.gz')
# print(mt_roi.header)
# print((mt_roi.get_fdata() > 0).sum())
