#Run pyAFQ in subject ACPC space
#04/14/2025

import os
import os.path as op
import AFQ.api.bundle_dict as abd
from AFQ.api.participant import ParticipantAFQ
from AFQ.definitions.image import ImageFile, RoiImage
import argparse


def main(participant_file):

    for participant in participant_file:
        # participant = 'sub-NSxLxYKx1964'
        # directories
        # paths_server = op.join('/Volumes', 'cos-lab-wpark78', 'Loic_backup', 'tests', 'afq-functionalROI2', 'derivatives')
        paths_local = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis', 'ampb')
        output_dir = op.join(paths_local, 'derivatives/afq', participant)
        os.makedirs(output_dir, exist_ok=True)

        paths_MT_roi = op.join(paths_local, 'analysis', 'func_space-ACPC_rois', participant, 'ses-04', 'func') # roi
        paths_roi = op.join(paths_local, 'analysis', 'julich_space-ACPC_rois', participant,  'ses-04', 'anat')
        paths_dwi = op.join(paths_local, 'derivatives', 'qsiprep', participant, 'ses-04', 'dwi') # dwi


        # define ROIs 
        mt_left = op.join(paths_MT_roi, participant+'_hemi-L_space-ACPC_label-MT_mask_dilated.nii.gz')
        mt_right = op.join(paths_MT_roi, participant+'_hemi-R_space-ACPC_label-MT_mask_dilated.nii.gz')

        # mt_left = op.join(paths_roi, participant+'_ses-04_desc-lhMT03SyN_mask.nii.gz')
        # mt_right = op.join(paths_roi, participant+'_ses-04_desc-rhMT03SyN_mask.nii.gz')

        pt_left = op.join(paths_roi, participant+'_ses-04_desc-lhPT03SyN_mask.nii.gz')
        pt_right = op.join(paths_roi, participant+'_ses-04_desc-rhPT03SyN_mask.nii.gz')

        sts_left = op.join(paths_roi, participant+'_ses-04_desc-lhSTS103SyN_mask.nii.gz')
        sts_right = op.join(paths_roi, participant+'_ses-04_desc-rhSTS103SyN_mask.nii.gz')

        v1_left = op.join(paths_roi, participant+'_ses-04_desc-lhV103SyN_mask.nii.gz')
        v1_right = op.join(paths_roi, participant+'_ses-04_desc-rhV103SyN_mask.nii.gz')

        lgn_left = op.join(paths_roi, participant+'_ses-04_desc-lhLGN03SyN_mask.nii.gz')
        lgn_right = op.join(paths_roi, participant+'_ses-04_desc-rhLGN03SyN_mask.nii.gz')

        pu_left = op.join(paths_roi, participant+'_ses-04_desc-lhPU03SyN_mask.nii.gz')
        pu_right = op.join(paths_roi, participant+'_ses-04_desc-rhPU03SyN_mask.nii.gz')

        fef_left = op.join(paths_roi, participant+'_ses-04_desc-lhFEF03SyN_mask.nii.gz')
        fef_right = op.join(paths_roi, participant+'_ses-04_desc-rhFEF03SyN_mask.nii.gz')

        po_left = op.join(paths_roi, participant+'_ses-04_desc-lhPO03SyN_mask.nii.gz')
        po_right = op.join(paths_roi, participant+'_ses-04_desc-rhPO03SyN_mask.nii.gz')


        # setup
        bundle_dict = abd.BundleDict({})
        bundle_kwargs = {
            "cross_midline": False,
            "space": "subject"
        } #    
        #    

        bundle_dict = abd.BundleDict({
            "PTxMT_L": {
                "start": pt_left,
                "end": mt_left,
                **bundle_kwargs
            },
            "PTxMT_R": {
                "start": pt_right,
                "end": mt_right,
                **bundle_kwargs
            },
            "V1xMT_L": {
                "start": v1_left,
                "end": mt_left,
                **bundle_kwargs
            },
            "V1xMT_R": {
                "start": v1_right,
                "end": mt_right,
                **bundle_kwargs
            },
            "LGNxMT_L": {
                "start": lgn_left,
                "end": mt_left,
                **bundle_kwargs
            },
            "LGNxMT_R": {
                "start": lgn_right,
                "end": mt_right,
                **bundle_kwargs
            },
            "STS1xMT_L": {
                "start": sts_left,
                "end": mt_left,
                **bundle_kwargs
            },
            "STS1xMT_R": {
                "start": sts_right,
                "end": mt_right,
                **bundle_kwargs
            },
            "PUxMT_L": {
                "start": pu_left,
                "end": mt_left,
                **bundle_kwargs
            },
            "PUxMT_R": {
                "start": pu_right,
                "end": mt_right,
                **bundle_kwargs
            },
            "FEFxMT_L": {
                "start": fef_left,
                "end": mt_left,
                **bundle_kwargs
            },
            "FEFxMT_R": {
                "start": fef_right,
                "end": mt_right,
                **bundle_kwargs
            },
            "POxMT_L": {
                "start": po_left,
                "end": mt_left,
                **bundle_kwargs
            },
            "POxMT_R": {
                "start": po_right,
                "end": mt_right,
                **bundle_kwargs
            }
            },
            resample_subject_to=my_dwi_path)

        brain_mask_definition = ImageFile(
            path = op.join(paths_dwi, participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-brain_mask.nii.gz')
        )

        scalars = [
            "dki_fa", "dki_md", "dki_mk", "dki_awf", 
            "fwdti_fa", "fwdti_md", "fwdti_fwf"
        ]

        tracking_params = {
            "seed_mask": RoiImage(use_endpoints = True), 
            "n_seeds": 10, 
        }

        myafq = ParticipantAFQ(
            dwi_data_file         = op.join(paths_dwi, participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-preproc_dwi.nii.gz'), 
            bval_file             = op.join(paths_dwi, participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-preproc_dwi.bval'), 
            bvec_file             = op.join(paths_dwi, participant+'_ses-04_acq-HCPdir99_space-ACPC_desc-preproc_dwi.bvec'), 
            output_dir            = output_dir, 
            bundle_info           = bundle_dict, 
            brain_mask_definition = brain_mask_definition, 
            scalars               = scalars, 
            tracking_params       = tracking_params
        ) #   reg_template_spec     = op.join(paths_local, 'analysis','MNI152NLin2009cAsym','anat','MNI152NLin2009cAsym_res-08_rec-wsinc_T1w.nii.gz'),

        # myafq.cmd_outputs(cmd = "rm")
        bundle = myafq.export_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs pyAFQ for a list of participants.")
    parser.add_argument(
        "--participants_file",
        type=str,
        required=True,
        help="Path to a text file containing participant IDs (one per line)."
    )
    args = parser.parse_args()

    # Read participants from file
    with open(args.participants_file, "r") as f:
        participants = [line.strip() for line in f if line.strip()]

    main(participants)