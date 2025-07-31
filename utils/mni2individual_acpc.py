#This script was developped to resample some potential exclusion ROIs such as template ROIs from AFQ_data folder or Julich ROIs from MNI space to individual subject diffusion space (ACPC)
# For each participant
#Loic Daumail - 07/25/2025
import os
import os.path as op
import ants
import argparse


def main(participant_file, paths_local):

    for participant in participant_file:
        # participant = 'sub-EBxLxQPx1957' #'sub-EBxGxCCx1986'
        # paths_local = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis','ampb')
        # # directories
        paths_mni = op.join(paths_local, 'analysis', 'MNI152NLin2009cAsym', 'anat')
        #paths_templates ='/Users/ldaumail3/AFQ_data/templates'
        paths_templates ='/Users/ldaumail3/Documents/research/brain_atlases/julich/d-f1fe19e8-99bd-44bc-9616-a52850680777-probabilistic-maps_PMs_227-areas-Area-Te-1.1'
        paths_qsiprep = op.join(paths_local, 'derivatives', 'qsiprep', participant, 'anat')
        paths_ACPC =  op.join(paths_local, 'analysis', 'julich_space-ACPC_rois', participant) #op.join(paths_local, 'analysis', 'AFQ-templates_space-ACPC_rois', participant)
        os.makedirs(paths_ACPC, exist_ok=True)
        ## Create MNI to ACPC space registration
        # load acpc t1 (fixed)
        acpc_t1 = op.join(paths_qsiprep, participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
        acpc_t1_img = ants.image_read(acpc_t1)

        # load acpc brain mask
        acpc_brain_mask = op.join(paths_qsiprep, participant+'_space-ACPC_desc-brain_mask.nii.gz')
        acpc_brain_mask_img = ants.image_read(acpc_brain_mask)

        # mni t1 (moving)
        mni_t1 = op.join(paths_mni, 'MNI152NLin2009cAsym_res-08_rec-wsinc_T1w.nii.gz')
        mni_t1_img = ants.image_read(mni_t1)

        # registration
        reg = ants.registration(
            fixed = acpc_t1_img,
            moving = mni_t1_img,
            type_of_transform = 'SyN',#'SyN', #SyN here, as qsiprep T1 and MNI152NLin2009cAsym are different brains. For same brains, use 'Rigid'
            mask = acpc_brain_mask_img,  
            reg_iterations = (1000, 500, 250, 100),  
            verbose = True
        )

        ## Create binary masks resampled from MNI to ACPC space
        #hoc1 = V1, hoc5 = MT, Te-2.1 = PT, STS1 = STS1, Methatalamus_CGL = LGN, hoc6 = PO =parieto-occipital visual area, Area-8v1 = FEF
        template_list = ['Area-Te-1.1_rh_MNI152.nii.gz', 'Area-Te-1.1_lh_MNI152.nii.gz'] #[f for f in os.listdir(paths_templates) if "prob_map" not in f] 
        mask_name = ['rhHeschl', 'lhHeschl']

        for mask, file in zip(mask_name, template_list):
            transformed_mask_path = op.join(paths_ACPC, 'ses-04', 'anat', participant+'_ses-04_desc-'+mask+'03SyN_mask.nii.gz')
            if os.path.exists(transformed_mask_path):
                print("File exists!")
            else:
                print("File does not exist. Creating it now")
                # load julich mask
                template_mask = op.join(paths_templates, file)
                template_mask_img = ants.image_read(template_mask)

                # binarize
                template_mask_img[template_mask_img >= 0.1] = 1
                template_mask_img[template_mask_img < 0.1] = 0

                # apply transformation: MNI Julich mask → ACPC space
                mytx = reg['fwdtransforms']
                transformed_mask = ants.apply_transforms(
                    moving = template_mask_img, 
                    fixed = acpc_t1_img, 
                    transformlist = mytx, 
                    interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
                )


                # save transformed mask
                os.makedirs(op.join(paths_ACPC, 'ses-04','anat'), exist_ok=True)
                ants.image_write(transformed_mask, transformed_mask_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACPC masks for a list of participants.")
    parser.add_argument(
        "--participants_file",
        type=str,
        required=True,
        help="Path to a text file containing participant IDs (one per line)."
    )
    parser.add_argument(
        "--paths_local",
        type=str,
        required=True,
        help="Base path to the local project directory."
    )
    args = parser.parse_args()
    # Read participants from file
    with open(args.participants_file, "r") as f:
        participants = [line.strip() for line in f if line.strip()]

    main(participants, args.paths_local)


