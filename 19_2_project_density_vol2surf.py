#After we created density maps with 19_tracts_hMTplus_overlap.py and assesse overlap of tracts density maps with hMT+
#We project the overlapping ones onto the cortical surface

import os
import os.path as op
import sys
import numpy as np
from scipy.ndimage import binary_erosion
import ants
import subprocess
import re

current_dir = op.dirname(op.abspath(__file__))
project_dir = op.abspath(op.join(current_dir, '..'))  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.overlap_masks import overlap_masks

def project_density_vol2surf(participants_file, tract_list, bids_path, pyAFQ_path, projdist):
    '''
    project_density_vol2surf --participants_file ./utils/study2_subjects_updated.txt --tract_list ["MTxLGNxPU"] --bids_path /Use
    rs/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb --pyAFQ_path /Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/deriva
    tives/pyAFQ/wmgmi_wb --projdist 0
    '''
    with open(participants_file, "r") as f:
        participants = [line.strip().lstrip("/") for line in f if line.strip()]
        # participants = [p for p in participants if p != "sub-NSxLxQUx1953"]

    for participant in participants :
        # participant = 'sub-EBxGxCCx1986' #sub-NSxGxBAx1970
        # tract_name = 'LeftMTxLGN'
        # bids_path = op.join('/Users','ldaumail3','Documents','research', 'ampb_mt_tractometry_analysis', 'ampb')
        analysis_path = op.join(bids_path, 'analysis')
        roi_path = op.join(analysis_path, 'ROIs','wang_space-ACPC_rois', participant)
        fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
        out_path = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps', participant, 'pyAFQ33_wb_red')
        os.makedirs(out_path, exist_ok=True)

        #Files 
        #Registration from ACPC to fs T1 space
        qsiprep_path = op.join(bids_path, 'derivatives', 'qsiprep', participant)
        acpc_t1 = op.join(qsiprep_path, 'anat', participant+'_space-ACPC_desc-preproc_T1w.nii.gz')
        acpc_t1_img = ants.image_read(acpc_t1)
        fs_brain_mask = op.join(fs_path, participant, 'mri', 'brain.mgz')
        fs_brain_mask_img = ants.image_read(fs_brain_mask)
        # 1) Load the FreeSurfer conformed reference volume
        fs_ref_path = op.join(fs_path, participant, 'mri', 'T1.mgz')  # or 'T1.mgz'
        fs_ref_img  = ants.image_read(fs_ref_path)
        # fs_ref_img.numpy().shape
        reg = ants.registration(
            fixed = fs_ref_img,
            moving = acpc_t1_img,
            type_of_transform = 'Rigid',#'SyN', #SyN here, as qsiprep T1 and fsaverage are different brains. For same brains, use 'Rigid'
            mask = fs_brain_mask_img,  
            reg_iterations = (1000, 500, 250, 100),  
            verbose = True
        )
        for tract in tract_list:
            hemisphere = ["L", "R"]#"L" if "Left" in tract_name else "R" if "Right" in tract_name else None
            #new_tract_name = tract_name.replace("Left", "") if "Left" in tract_name else tract_name.replace("Right", "") if "Right" in tract_name else None
            # roi_names = [new_name] #, 'RightMTmaskxLGN'
            for h, hemi in enumerate(hemisphere):
                whemi = "Left" if hemi == "L" else "Right"
                match = re.search(r"^(Left|Right)?", tract)
                tract_hemi = match.group(1) # This will be 'Left', 'Right', or None
                if tract_hemi in [whemi, None]: #If the hemisphere value (Left/Right) matches the actual hemi, or if the hemi is not specified, do the next steps
                    # ----------------------------
                    # STEP 1: Load volumetric ROI 
                    # ----------------------------
                    mt_mask_file = op.join(roi_path, participant+'_hemi-'+hemi+'_space-ACPC_label-MT_mask_dilated.nii.gz')
                    # mt_mask_img = ants.image_read(mt_mask_file)
                    # wmgmi_mask_img = ants.image_read(wmgmi_mask_file)

                    #--- STEP 2: Calculate overlap mask between MT mask and WMGMI from pyAFQ (used to catch endpoint values)
                    afq_path = op.join(pyAFQ_path, participant)
                    wmgmi_mask_file = op.join(afq_path, f"{participant}_ses-concat_acq-HCPdir99_desc-wmgmi_mask.nii.gz")

                    input_files = [wmgmi_mask_file, mt_mask_file]
                    mt_wmgmi_file = op.join(roi_path, participant+'_hemi-'+hemi+'_space-ACPC_label-MTxWMGMI_mask.nii.gz')
                    overlap_masks(input_files, mt_wmgmi_file)


                    # -----------------------------
                    ### --- STEP 3: load streamline density map
                    # -----------------------------
                    #roi_names[h] 
                    density_path = op.join(out_path, f"{participant}_ses-concat_desc-{tract}_tdi_map.nii.gz")
                    density_map_img = ants.image_read(density_path)

                    # -----------------------------
                    ### --- STEP 4: Resample overlapped mask and density into FS reference grid
                    # -----------------------------

                    # apply transformation: ACPC → FS space
                    mytx = reg['fwdtransforms']
                    density_fs_img = ants.apply_transforms(
                        moving = density_map_img, 
                        fixed = fs_ref_img, 
                        transformlist = mytx, 
                        interpolator = "genericLabel" # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)
                    )
                    mt_wmgmi_mask_img = ants.image_read(mt_wmgmi_file)
                    mt_mask_fs_img = ants.apply_transforms(
                        moving = mt_wmgmi_mask_img, 
                        fixed = fs_ref_img, 
                        transformlist = mytx, 
                        interpolator = "genericLabel"
                    ) # # keep it as a binary mask without smoothing it out, great for parcellations ("nearestNeighbor" is also discrete but can introduce aliasing)

                # -----------------------------
                    ### --- STEP 5: generate volumetric ROI surface voxels ---
                    # -----------------------------
                    mt_wmgmi_mask_fs = mt_mask_fs_img.numpy()
                    mt_wmgmi_vox_fs   = np.argwhere(mt_wmgmi_mask_fs.astype(bool))    # ROI voxel coordinates (N,3) in FS voxel indices (i,j,k)
                    
                    # mt_mask_fs.shape
                    # outer_vox_fs.shape
                    # -----------------------------
                    ### --- STEP 6: Extract MTxWMGMI intensities from density map ---
                    # -----------------------------
                    # --- Get MTxWMGMI voxel intensities ---
                    density = density_fs_img.numpy()
                    intensities = density[
                        mt_wmgmi_vox_fs[:, 0],
                        mt_wmgmi_vox_fs[:, 1],
                        mt_wmgmi_vox_fs[:, 2]
                    ]
                    # density.shape
                    # Make an empty array with same shape as reference
                    out_data = np.zeros(density_fs_img.shape, dtype=np.float32)
                    # Fill only the MTxWMGMI voxels with your intensities
                    out_data[mt_wmgmi_vox_fs[:, 0], mt_wmgmi_vox_fs[:, 1], mt_wmgmi_vox_fs[:, 2]] = intensities
                    # out_data.shape
                    out_img = ants.from_numpy(out_data, origin=density_fs_img.origin, spacing=density_fs_img.spacing, direction=density_fs_img.direction)
                    # print(out_img)
                    # -----------------------------
                    # STEP 6: Save as NIfTI (.nii.gz)
                    # -----------------------------
                    mt_density_path = op.join(analysis_path, "tdi_maps", "wmgmi_voxels", participant,f"{participant}_hemi-{hemi}_space-fsnative_label-wang{tract}xWMGMI_density2.nii.gz")
                    os.makedirs(op.dirname(mt_density_path), exist_ok=True)
                    ants.image_write(out_img, mt_density_path)

                    # --- Plotting ---

                    # mt_density_map_img = ants.image_read(mt_density_path)
                    # intensities = mt_density_map_img.numpy()
                    # intensities.shape
                    # # Get voxel coordinates of nonzero entries
                    # coords = np.argwhere(intensities > 0)   # (N, 3) -> i, j, k voxel indices
                    # values = intensities[intensities > 0]   # (N,)   -> corresponding density values
                    # import matplotlib.pyplot as plt
                    # fig = plt.figure(figsize=(10, 8))
                    # ax = fig.add_subplot(111, projection='3d')

                    # img = ax.scatter(
                    #     coords[:, 0],
                    #     coords[:, 1],
                    #     coords[:, 2],
                    #     c=values,
                    #     cmap='Spectral',
                    #     s=60
                    # )
                    # ax.view_init(elev=90, azim=90)
                    # hemi_name = "Left" if hemi == "L" else "Right"
                    # ax.set_title(f"LGNxMT Endpoint Density in {hemi_name} MTxWMGMI")
                    # fig.colorbar(img, ax=ax, label='Endpoint Count')

                    # plt.show()

                    ## STEP 7: Perform volume to surface projection of MTxWMGMI ROI density values
                    # Output file name
                    out_fsnative_file = os.path.join(out_path, f"{participant}_hemi-{hemi}_space-fsnative_label-wang{tract}_desc-fsprojdensity{projdist}mm2.mgh")

                    hemi_fs  = "lh" if hemi == "L" else "rh"
                    # Build mri_vol2surf command
                    cmd = [
                        "mri_vol2surf",
                        "--mov", mt_density_path,
                        "--regheader", participant,
                        "--hemi", hemi_fs,
                        "--surf", "white",
                        "--projdist", f"-{projdist}",
                        "--sd", fs_path,
                        "--out", out_fsnative_file
                    ] #"--projfrac", "-0.3",  "--regheader", participant

                    # Run the command
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})

                    # Resample to fsaverage space
                    source_density_file = out_fsnative_file
                    out_fsaverage_file = op.join(out_path, f"{participant}_hemi-{hemi_fs}_space-fsaverage_label-wang{tract}_desc-fsprojdensity{projdist}mm2.mgh")

                    cmd = ["mri_surf2surf",
                    "--srcsubject", participant, 
                    "--trgsubject", "fsaverage",
                    "--hemi", hemi_fs, 
                    "--sval", source_density_file, 
                    "--tval", out_fsaverage_file ]

                    # Run the command
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, check=True, env={**os.environ, "SUBJECTS_DIR": fs_path})

participants_file = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb', 'code', 'utils', 'study2_subjects_updated.txt')
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')


# tract_list=['LPTR', 'RPTR', 'LSTR', 'RSTR', 'LeftInferiorLongitudinal', 'RightInferiorLongitudinal', 
#     'LeftInferiorFrontooccipital', 'RightInferiorFrontooccipital', 
#     'LeftSuperiorLongitudinalI', 'RightSuperiorLongitudinalI',
#     'LeftSuperiorLongitudinalII', 'RightSuperiorLongitudinalII',
#     'LeftSuperiorLongitudinalIII', 'RightSuperiorLongitudinalIII',
#     'LeftAnteriorVerticalOccipital', 'RightAnteriorVerticalOccipital', 
#     'LeftPosteriorVerticalOccipital', 'RightPosteriorVerticalOccipital',
#     'LeftArcuate', 'RightArcuate',
#     'LeftPosteriorArcuate', 'RightPosteriorArcuate'] #'CallosumOccipital'

tract_list=['LeftEarlyVisual', 'RightEarlyVisual', 'LeftOpticRadiation', 'RightOpticRadiation', 'LeftTemporoparietal', 'RightTemporoparietal',
            'LPTR', 'RPTR', 'LSTR', 'RSTR', 'LeftInferiorLongitudinal', 'RightInferiorLongitudinal', 
            'LeftInferiorFrontooccipital', 'RightInferiorFrontooccipital', 
            'LeftSuperiorLongitudinalI', 'RightSuperiorLongitudinalI',
            'LeftSuperiorLongitudinalII', 'RightSuperiorLongitudinalII',
            'LeftSuperiorLongitudinalIII', 'RightSuperiorLongitudinalIII',
            'LeftAnteriorVerticalOccipital', 'RightAnteriorVerticalOccipital', 
            'LeftPosteriorVerticalOccipital', 'RightPosteriorVerticalOccipital',
            'LeftArcuate', 'RightArcuate',
            'LeftPosteriorArcuate', 'RightPosteriorArcuate']

pyAFQ_path = op.join(bids_path, 'derivatives', 'pyafq', 'wmgmi_wb', 'afq33-wb_6rounds')

projdist = '3'

project_density_vol2surf(participants_file, tract_list, bids_path, pyAFQ_path, projdist)