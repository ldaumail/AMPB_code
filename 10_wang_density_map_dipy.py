## Dipy Approach
#Compute streamline density map with dipy
import os
import os.path as op
import argparse
import sys
import re
current_dir = op.dirname(op.abspath(__file__))
project_dir = op.abspath(op.join(current_dir, '..'))  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.streamlines_utils import streamline2dipy_density


def main(participants_file, tract_name, pyAFQ_path, density_out_path, out_folder):
    '''
    Ex usage: python 10_wang_density_map_dipy.py --participants_file ./utils/study2_subjects_updated.txt --tract_name LeftMTxLGNxPU --pyAFQ_path /Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/deriva
tives/pyAFQ/wmgmi_targ/pve_pyafq33 --density_out_path /Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/tdi_maps/dipy_wmgmi_tdi_maps --out_folder pve_pyAFQ33_targ
    '''
    #bids_path, --bids_path /Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb 
    for participant in participants_file:
        # participant = 'sub-EBxGxCCx1986'
        # tract_name = 'RightMTxFEF'
        # bids_path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb'
        # pyAFQ_path = '/Volumes/cos-lab-wpark78/LoicDaumail/ampb/derivatives/pyafq/wmgmi_wang/pyAFQ3.3'
        # density_out_path = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps', participant, 'pyAFQ33_targ')
        os.makedirs(op.join(density_out_path, participant, out_folder), exist_ok=True)
        
        new_name = tract_name.replace("MT", "MTmask")
        ## Captures (Hemisphere)(TargetROI)x(RestOfLabel)
        pattern = r"^(Left|Right)(MT)x(.+)$"

        # Reorders components and appends 'mask '
        new_name = re.sub(pattern, r"\1\3x\2mask", tract_name)
        print(repr(new_name))  # Output: 'LeftPTxSTS1xMTmask '

        tract_path = os.path.join(pyAFQ_path, f"afq-{tract_name}", participant, 'bundles', participant+'_ses-concat_acq-HCPdir99_desc-' + new_name + '_tractography.trx')
        tract_tdi_map = os.path.join(density_out_path, participant, out_folder, participant+'_ses-concat_desc-wang' + tract_name + '_tdi_map.nii.gz')
        template = op.join(pyAFQ_path, f"afq-{tract_name}", participant, participant+'_ses-concat_acq-HCPdir99_b0ref.nii.gz')

        # print(f"\n🔹 Trx file: {tract_path}")
        streamline2dipy_density(tract_path, template, tract_tdi_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create streamline density maps")
    parser.add_argument(
        "--participants_file",
        type=str,
        required=True,
        help="Path to a text file containing participant IDs (one per line)."
    )
    parser.add_argument(
        "--tract_name",
        type=str,
        required=True,
        help="Name of the tract as written in bundle file name"
    )
    # parser.add_argument(
    #     "--bids_path",
    #     type=str,
    #     required=True,
    #     help="Name of the tract as written in bundle file name"
    # )
    parser.add_argument(
        "--pyAFQ_path",
        type=str,
        required=True,
        help="Path to pyAFQ derivatives"
    )
    parser.add_argument(
        "--density_out_path",
        type=str,
        required=True,
        help="Path to density maps output"
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        required=True,
        help="Density output folder"
    )
    args = parser.parse_args()

    # Read participants from file
    with open(args.participants_file, "r") as f:
        participants = [line.strip() for line in f if line.strip()]

    main(participants, args.tract_name, args.pyAFQ_path, args.density_out_path, args.out_folder) #args.bids_path,

# Example
# participant = 'sub-NSxGxBAx1970'
# bids_path = op.join('/Users','ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
# tdi_path = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_tdi_maps', participant)
# os.makedirs(tdi_path, exist_ok=True)

# pyAFQ_path = op.join('/Volumes','cos-lab-wpark78','LoicDaumail','ampb','derivatives','pyafq', 'gpu-afq_MT-STS1_nseeds20_0mm_nowm_dist3',participant)
# bundle_path = op.join(pyAFQ_path,'bundles')

# for tract in ['STS1xMTL', 'STS1xMTR']: 
#     sts1_mtL_path = os.path.join(bundle_path, participant+'_ses-04_acq-HCPdir99_desc-'+tract+'_tractography.trx')
#     sts1_mtL_tdi_map = os.path.join(tdi_path, participant+'_ses-04_desc-'+tract+'_tdi_map.nii.gz')
#     trx_template = op.join(pyAFQ_path, participant+'_ses-04_acq-HCPdir99_b0ref.nii.gz')

#     streamline2dipy_density(sts1_mtL_path, trx_template, sts1_mtL_tdi_map)
