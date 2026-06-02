#Since pyAFQ default atlas does not include the superior and posterior thalamic radiations
#We create additional thalamus ROIs and cortical ROIs using AICHA atlas (default atlas used for pyAFQ)

#1. combine ROIs from AICHA atlas
#2. Resample the new ROI into ACPC space

import os
import os.path as op
import ants
import argparse
import sys
# import numpy as np
current_dir = op.dirname(op.abspath(__file__))
project_dir = op.abspath(op.join(current_dir, '..'))  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.dilate_mask import dilate_mask


def main(participants_file, bids_path, roi_name, lh_rois, rh_rois):
    '''
    Bash command line (posterior ROI) :
    python 18_1_generate_thalamic_radiations_rois.py --participants_file /Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils/study2_subjects_updated.txt \
            --bids_path /Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb --roi_name "superior" \
            --lh_rois 77 79 81 83 85 107 109 111 113 115 117 119 121 123 125 127 129 131 133 135 137 139 141 143 265 267 269 271 273 275 277 279 281 283 285 287 289 291 293 299 301 303 191 193 195 185 177 175 99 103 105 \
            --rh_rois 78 80 82 84 86 108 110 112 114 116 118 120 122 124 126 128 130 132 134 136 138 140 142 144 266 268 270 272 274 276 278 280 282 284 286 288 290 292 294 300 302 304 192 194 196 186 178 176 100 104 106

            
            #STRs
            --lh_rois 257 259 261 263 229 231 233 15 17 221 223 225 63 65 67 69 51 53 55 57 59 61 71 73 75 \
            --rh_rois 258 260 262 264 230 232 234 16 18 222 224 226 64 66 68 70 52 54 56 58 60 62 72 74 76 

            #PTRs:
            --lh_rois 77 79 81 83 85 107 109 111 113 115 117 119 121 123 125 127 129 131 133 135 137 139 141 143 265 267 269 271 273 275 277 279 281 283 285 287 291 293 299 301 303 191 193 195 185 177 175 99 103 105 \
            --rh_rois 78 80 82 84 86 108 110 112 114 116 118 120 122 124 126 128 130 132 134 136 138 140 142 144 266 268 270 272 274 276 278 280 282 284 286 288 292 294 300 302 304 192 194 196 186 178 176 100 104 106
    '''
    # roi_name = 'posterior'
    mni_aicha_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases','AICHA')
    
    #Load atlas file
    mni_aicha = ants.image_read(op.join(mni_aicha_path, 'AICHA.nii'))

    #Thalamus ROIs:
    # lh_rois = [367, 369, 371, 373, 375, 377, 379, 381, 383]
    # rh_rois = [368, 370, 372, 374, 376, 378, 380, 382, 384]

    #PTRs cortex ROIs:
    #lh_rois = [77, 79, 81, 83, 85,107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 283, 285, 287, 291, 293, 299, 301, 303, 191,193, 195, 185, 177, 175, 99, 103, 105]
    #rh_rois = [78, 80, 82, 84, 86, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 284, 286, 288, 292, 294, 300, 302, 304, 192, 194, 196, 186, 178, 176, 100, 104, 106]

    #STRs cortex ROIs:
    #lh_rois = [257, 259, 261, 263, 229, 231, 233, 15, 17, 221, 223, 225, 63,65,67,69, 51, 53, 55, 57, 59, 61,71, 73, 75, 257, 259, 261, 263]
    #rh_rois = [258,260, 262, 264, 230, 232, 234, 16, 18, 222, 224, 226, 64,66,68,70, 52, 54, 56, 58, 60, 62, 72, 74, 76, 258,260,262,264]

    for participant in participants_file:
        # participant = 'sub-NSxGxHNx1952'
        # bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
        save_dir = op.join(bids_path, 'analysis','ROIs','AICHA_space-ACPC_rois', participant)
        os.makedirs(save_dir, exist_ok=True)
        qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'anat')
        acpc_t1_path       = op.join(qsiprep_path, participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
        acpc_t1_img       = ants.image_read(acpc_t1_path)
        acpc_brain_mask_img = ants.image_read(op.join(qsiprep_path, participant+'_space-ACPC_desc-brain_mask.nii.gz'))
        mni_t1_img = ants.image_read(op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases','Wang_2015', 'MNI152_T1_1mm.nii.gz'))
        
        # MNI to ACPC T1 registration
        reg = ants.registration(
            fixed = acpc_t1_img,
            moving = mni_t1_img,
            type_of_transform = 'SyN',#'SyN', #SyN here, as qsiprep T1 and MNI152NLin2009cAsym are different brains. For same brains, use 'Rigid'
            mask = acpc_brain_mask_img,  
            reg_iterations = (1000, 500, 250, 100),  
            verbose = True
        )
        # -------------------------------
        ## Load  ROIs from each hemisphere and combine them
        # -------------------------------

        for hemi_fs in ['lh', 'rh']:
                #Register and Transform mask from MNI to fs native space
                rois = lh_rois if hemi_fs == "lh" else rh_rois
                mni_img = mni_aicha * 0
                for roi in rois:
                    mni_img = mni_img + (mni_aicha == roi)

                hemi  = "L" if hemi_fs == "lh" else "R"
                transformed_path = op.join(save_dir, f"{participant}_hemi-{hemi}_space-ACPC_desc-{roi_name}_mask.nii.gz")
                if os.path.exists(transformed_path):
                    print("File exists!")
                else:
                    print("File does not exist. Creating it now")
                

                # import numpy as np
                # import matplotlib.pyplot as plt

                # # Example array
                # data = mni_mst_img.numpy()
                # # data[data < 0] = 0  # make some zeros

                # # Keep only non-zero values
                # nonzero_vals = data[data != 0]

                # # Plot histogram
                # plt.hist(nonzero_vals, bins=50, color='steelblue', edgecolor='black')
                # plt.xlabel("Value")
                # plt.ylabel("Count")
                # plt.title("Histogram of Non-Zero Values")
                # plt.show()

                #Resample binary mask into ACPC
                # Apply transform  MNI mask → ACPC space
                mytx = reg['fwdtransforms']
                transformed_mask = ants.apply_transforms(
                    moving = mni_img, 
                    fixed = acpc_t1_img, 
                    transformlist = mytx, 
                    interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
                )
                
                # # save transformed mask
                ants.image_write(transformed_mask, transformed_path)

                ##Dilate and save
                input_mask = transformed_path
                output_mask = op.join(save_dir,  f"{participant}_hemi-{hemi}_space-ACPC_desc-{roi_name}_mask_dilated.nii.gz")

                dilate_mask(input_mask, output_mask, dilate = 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACPC masks for a list of participants.")
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
        help="Path to bids formated data."
    )
    parser.add_argument(
        "--roi_name",
        type=str,
        required=True,
        help="Roi name to create."
    )
    parser.add_argument(
        "--lh_rois",
        type=int,
        nargs='+',
        required=True,
        help="Roi LUT numbers from lh."
    )
    parser.add_argument(
        "--rh_rois",
        type=int,
        nargs='+',
        required=True,
        help="Roi LUT numbers from rh."
    )
    args = parser.parse_args()

    # Read participants from file
    with open(args.participants_file, "r") as f:
        participants = [line.strip() for line in f if line.strip()]
    #main(participants_file = participants, bids_path = args.bids_path)
    main(participants_file = participants, bids_path = args.bids_path, roi_name = args.roi_name, lh_rois = args.lh_rois, rh_rois = args.rh_rois)

