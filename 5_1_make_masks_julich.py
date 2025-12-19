#This script was developped to resample Julich ROIs from MNI space to individual subject diffusion space (ACPC)
# For each participant
#Loic Daumail - 12/04/2025
import os
import os.path as op
import ants
import argparse
import re


def main(participant_file):

    for participant in participant_file:
        #participant = 'sub-NSxGxBAx1970'
        # directories
        paths_local = op.join('/Users','ldaumail3','Documents','research','ampb_mt_tractometry_analysis','ampb')
        paths_mni = op.join(paths_local, 'analysis', 'MNI152NLin2009cAsym', 'anat')
        paths_julich = op.join(paths_local, 'analysis', 'MNI152NLin2009cAsym', 'julich')
        paths_qsiprep = op.join(paths_local, 'derivatives', 'qsiprep', participant, 'anat')
        paths_ACPC = op.join(paths_local, 'analysis', 'julich_space-ACPC_rois', participant)
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
        prob_roi_list = ['Area-hOc1_lh_MNI152.nii.gz', 'Area-Te-2.1_lh_MNI152.nii.gz', 'Area-STS1_lh_MNI152.nii.gz', 'Metathalamus-CGL_lh_MNI152.nii.gz', 'Area-hOc6_lh_MNI152.nii.gz', 'Area-8v1_lh_MNI152.nii.gz',
                        'Area-hOc1_rh_MNI152.nii.gz', 'Area-Te-2.1_rh_MNI152.nii.gz', 'Area-STS1_rh_MNI152.nii.gz', 'Metathalamus-CGL_rh_MNI152.nii.gz', 'Area-hOc6_rh_MNI152.nii.gz', 'Area-8v1_rh_MNI152.nii.gz']
        mask_name = ['lhV1', 'lhPT', 'lhSTS1', 'lhLGN', 'lhPO', 'lhFEF',
                    'rhV1', 'rhPT', 'rhSTS1', 'rhLGN', 'rhPO', 'rhFEF']

        for mask, prob_roi in zip(mask_name,prob_roi_list):
            hemi  = re.sub(".*(lh|rh).*", "\\1", mask)
            hemi  = "L" if hemi == "lh" else "R"
            roi = re.sub(".*(?:lh|rh)(.*)", "\\1", mask)
            transformed_mask_path = op.join(paths_ACPC, 'ses-concat', 'anat', participant+'_hemi-'+hemi+'_space-ACPC_desc-'+roi+'_mask.nii.gz')
            #participant+'_hemi-L_space-ACPC_label-'+roi_name+'_mask.nii.gz'
            if os.path.exists(transformed_mask_path):
                print("File exists!")
            else:
                print("File does not exist. Creating it now")
                # load julich mask
                julich_mask = op.join(paths_julich, prob_roi)
                julich_mask_img = ants.image_read(julich_mask)

                # binarize
                julich_mask_img[julich_mask_img >= 0.3] = 1
                julich_mask_img[julich_mask_img < 0.3] = 0

                # apply transformation: MNI Julich mask → ACPC space
                mytx = reg['fwdtransforms']
                transformed_mask = ants.apply_transforms(
                    moving = julich_mask_img, 
                    fixed = acpc_t1_img, 
                    transformlist = mytx, 
                    interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
                )


                # save transformed mask
                os.makedirs(op.join(paths_ACPC, 'ses-concat', 'anat'), exist_ok=True)
                ants.image_write(transformed_mask, transformed_mask_path)


            ##Create pulvinar mask

        for hemi in ['lh', 'rh']:
            #Thalamus_PUi = inferior pulvinar, Thalamus_PUm = medial pulvinar
            prob_PUi_roi = 'Thalamus-PUi_'+hemi+'_MNI152.nii.gz'
            prob_PUm_roi = 'Thalamus-PUm_'+hemi+'_MNI152.nii.gz'
            prob_PUl_roi = 'Thalamus-PUl_'+hemi+'_MNI152.nii.gz'
            PU_mask_name = 'PU'

            hemi  = "L" if hemi == "lh" else "R"
            transformed_mask_path = op.join(paths_ACPC, 'ses-concat', 'anat', participant+'_hemi-'+hemi+'_space-ACPC_desc-'+PU_mask_name+'_mask.nii.gz')
            if os.path.exists(transformed_mask_path):
                print("File exists!")
            else:
                print("File does not exist. Creating it now")
                # load julich masks
                PUi_julich_mask = op.join(paths_julich, prob_PUi_roi)
                PUi_julich_mask_img = ants.image_read(PUi_julich_mask)

                PUm_julich_mask = op.join(paths_julich, prob_PUm_roi)
                PUm_julich_mask_img = ants.image_read(PUm_julich_mask)

                PUl_julich_mask = op.join(paths_julich, prob_PUl_roi)
                PUl_julich_mask_img = ants.image_read(PUl_julich_mask)

                # binarize
                PUi_julich_mask_img[PUi_julich_mask_img >= 0.3] = 1
                PUi_julich_mask_img[PUi_julich_mask_img < 0.3] = 0

                PUm_julich_mask_img[PUm_julich_mask_img >= 0.3] = 1
                PUm_julich_mask_img[PUm_julich_mask_img < 0.3] = 0

                PUl_julich_mask_img[PUl_julich_mask_img >= 0.3] = 1
                PUl_julich_mask_img[PUl_julich_mask_img < 0.3] = 0

                img1_data = PUi_julich_mask_img.numpy()
                img2_data = PUm_julich_mask_img.numpy()
                img3_data = PUl_julich_mask_img.numpy()

                union_img = (img1_data > 0) | (img2_data > 0) | (img3_data > 0)
                union_img = union_img.astype(PUi_julich_mask_img.dtype)  # Keep same data type
                
                # Convert back to ANTs image
                julich_mask_img = ants.from_numpy(union_img, origin=PUi_julich_mask_img.origin, spacing=PUi_julich_mask_img.spacing, direction=PUi_julich_mask_img.direction)


                #Resample binary mask into ACPC
                    # apply inverse transformation: MNI Julich mask → ACPC space
                mytx = reg['fwdtransforms']
                transformed_mask = ants.apply_transforms(
                    moving = julich_mask_img, 
                    fixed = acpc_t1_img, 
                    transformlist = mytx, 
                    interpolator = "genericLabel" ## keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
                )

                # # save transformed mask
                ants.image_write(transformed_mask, transformed_mask_path)

                #Create LIP-VIP mask
        for hemi in ['lh', 'rh']:
                
            prob_hIP1_roi = 'Area-hIP1_'+hemi+'_MNI152.nii.gz'
            prob_hIP2_roi = 'Area-hIP2_'+hemi+'_MNI152.nii.gz'
            prob_hIP3_roi = 'Area-hIP3_'+hemi+'_MNI152.nii.gz'
            hIP_mask_name = 'hIP'
            hemi  = "L" if hemi == "lh" else "R"
            transformed_mask_path = op.join(paths_ACPC, 'ses-concat', 'anat', participant+'_hemi-'+hemi+'_space-ACPC_desc-'+hIP_mask_name+'_mask.nii.gz')
            if os.path.exists(transformed_mask_path):
                print("File exists!")
            else:
                print("File does not exist. Creating it now")
                # load julich masks
                hIP1_julich_mask = op.join(paths_julich, prob_hIP1_roi)
                hIP1_julich_mask_img = ants.image_read(hIP1_julich_mask)

                hIP2_julich_mask = op.join(paths_julich, prob_hIP2_roi)
                hIP2_julich_mask_img = ants.image_read(hIP2_julich_mask)

                hIP3_julich_mask = op.join(paths_julich, prob_hIP3_roi)
                hIP3_julich_mask_img = ants.image_read(hIP3_julich_mask)

                # binarize
                hIP1_julich_mask_img[hIP1_julich_mask_img >= 0.3] = 1
                hIP1_julich_mask_img[hIP1_julich_mask_img < 0.3] = 0

                hIP2_julich_mask_img[hIP2_julich_mask_img >= 0.3] = 1
                hIP2_julich_mask_img[hIP2_julich_mask_img < 0.3] = 0

                hIP3_julich_mask_img[hIP3_julich_mask_img >= 0.3] = 1
                hIP3_julich_mask_img[hIP3_julich_mask_img < 0.3] = 0

                img1_data = hIP1_julich_mask_img.numpy()
                img2_data = hIP2_julich_mask_img.numpy()
                img3_data = hIP3_julich_mask_img.numpy()

                union_img = (img1_data > 0) | (img2_data > 0) | (img3_data > 0)
                union_img = union_img.astype(hIP1_julich_mask_img.dtype)  # Keep same data type
                
                # Convert back to ANTs image
                julich_mask_img = ants.from_numpy(union_img, origin=hIP1_julich_mask_img.origin, spacing=hIP1_julich_mask_img.spacing, direction=hIP1_julich_mask_img.direction)

                #Resample binary mask into ACPC
                    # apply inverse transformation: MNI Julich mask → ACPC space
                mytx = reg['fwdtransforms']
                transformed_mask = ants.apply_transforms(
                    moving = julich_mask_img, 
                    fixed = acpc_t1_img, 
                    transformlist = mytx, 
                    interpolator = "genericLabel" ## keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
                )
                # # save transformed mask
                ants.image_write(transformed_mask, transformed_mask_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACPC masks for a list of participants.")
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
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate ACPC masks for a list of participants.")
#     parser.add_argument(
#         "--participants_list",
#         type=str,
#         required=True,
#         help="Comma-separated list of participant IDs, e.g. sub-001,sub-002,sub-003"
#     )
#     args = parser.parse_args()
#     participants = args.participants_list.split(",")
#     main(participants)

