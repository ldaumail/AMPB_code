#Run pyAFQ in subject ACPC space
#04/14/2025

import os
import os.path as op
import AFQ.api.bundle_dict as abd
from AFQ.api.participant import ParticipantAFQ
from AFQ.definitions.image import ImageFile, RoiImage
import argparse


def main(participant_list, paths_local):
    '''
    To run this function, you need to provide 2 variables:
    1. The path to a text file containing the list of participant IDs
    2. Provide the path to the Bids directory
    '''
    for participant in participant_list:
        # participant = 'sub-NSxLxYKx1964'
        # directories
        # paths_local = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis', 'ampb')
        output_dir = op.join(paths_local, 'derivatives/wb-pyafq', participant)
        os.makedirs(output_dir, exist_ok=True)

        paths_wm = op.join(paths_local, 'analysis', 'acpc_wm', participant)
        paths_dwi = op.join(paths_local, 'derivatives', 'qsiprep', participant, 'ses-04', 'dwi') # dwi


        brain_mask_definition = ImageFile(
            path = op.join(paths_dwi, participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-brain_mask.nii.gz')
        )

        wm_mask_definition = ImageFile(path = op.join(paths_wm, participant+"_space-ACPC_label-WM_mask.nii.gz"))


        scalars = [
            "dki_fa", "dki_md", "dki_mk", "dki_awf", 
            "fwdti_fa", "fwdti_md", "fwdti_fwf"
        ]

        tracking_params = {
            "seed_mask": wm_mask_definition, 
            "n_seeds":25000000,
            "random_seeds":True,
            "rng_seed":2022,
        }

        myafq = ParticipantAFQ(
            dwi_data_file         = op.join(paths_dwi, participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-preproc_dwi.nii.gz'), 
            bval_file             = op.join(paths_dwi, participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-preproc_dwi.bval'), 
            bvec_file             = op.join(paths_dwi, participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-preproc_dwi.bvec'), 
            output_dir            = output_dir, 
            brain_mask_definition = brain_mask_definition, 
            scalars               = scalars, 
            tracking_params       = tracking_params,
        ) #   reg_template_spec     = op.join(paths_local, 'analysis','MNI152NLin2009cAsym','anat','MNI152NLin2009cAsym_res-08_rec-wsinc_T1w.nii.gz'),

        # myafq.cmd_outputs(cmd = "rm")
        bundle = myafq.export_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs pyAFQ for a list of participants.")
    parser.add_argument(
        "--participants_list",
        type=str,
        required=True,
        help="Comma-separated list of participant IDs, e.g. sub-001,sub-002,sub-003"
    )
    parser.add_argument(
        "--paths_local",
        type=str,
        required=True,
        help="Base path to the local project directory."
    )
    args = parser.parse_args()
    participants = args.participants_list.split(",")
    # # Read participants from file
    # with open(args.participants_file, "r") as f:
    #     participants = [line.strip() for line in f if line.strip()]

    main(participants, args.paths_local)
