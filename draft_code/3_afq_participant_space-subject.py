import os
import os.path as op
import plotly
import ants
from dipy.io.image import load_nifti, load_nifti_data
import numpy as np
import sys
utils = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils'
sys.path.append(op.expanduser(f'{utils}'))

from AFQ.api.participant import ParticipantAFQ
from AFQ.definitions.image import RoiImage, ImageFile, ScalarImage
import AFQ.api.bundle_dict as abd

### 1: Register freesurfer T1 to diffusion space
proj_path = op.expanduser("~/Documents/research/ampb_mt_tractometry_analysis/tests/pyAFQ_tests/afq-functionalROI2")
participant = "sub-NSxLxYKx1964"
# Step 1: Load the freesurfer T1
FS_T1 = op.join("/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/freesurfer",participant,"ses-04/anat/T1.mgz")
FS_T1_copy = op.join(proj_path, "derivatives/freesurfer",participant,"ses-04/anat", participant+"_ses-04_desc-freesurfer_T1w.nii.gz")
freesurferCommand = f'mri_convert {FS_T1} {FS_T1_copy}'
os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')
FST1 = op.join(proj_path,"derivatives/freesurfer", participant,"ses-04/anat/"+participant+"_ses-04_desc-freesurfer_T1w.nii.gz")  # Replace with your atlas file
fs_t1_nii = ants.image_read(FST1)
#Step 2: Load qsiprep ACPC T1 and mask
QSIPREPT1 = op.join(proj_path,"derivatives/qsiprep", participant,"anat/"+participant+"_space-ACPC_desc-preproc_T1w.nii.gz")
qsiprep_t1_nii = ants.image_read(QSIPREPT1)

QSIPREP_T1_MASK = op.join(proj_path,"derivatives/qsiprep", participant,"anat/"+participant+"_space-ACPC_desc-brain_mask.nii.gz")
qsiprep_t1_mask = ants.image_read(QSIPREP_T1_MASK)

#Step 3: register FST1 to QSIPREPT1
# Apply registration
reg = ants.registration(
    fixed=qsiprep_t1_nii,
    moving=fs_t1_nii,
    type_of_transform= 'Rigid', #"Rigid",  # Corresponds to --transform Rigid[ 0.1 ]
    mask=qsiprep_t1_mask,  # Corresponds to --masks [ ${QSIPREP_T1_MASK} NULL ]
    reg_iterations=(1000, 500, 250, 100),  # Corresponds to --convergence [ 1000x500x250x100, 1e-06, 10 ]
    verbose=True
)

# Save outputs
#ants.image_write(reg["warpedmovout"], os.path.join(proj_path, "derivatives/functionalROIs/sub-NSxLxPQx1973/ses-04/anat", "sub-NSxLxPQx1973_transform_from-fs_to-ACPC_Warped.nii.gz"))
# ants.write_transform(reg["fwdtransforms"], os.path.join(proj_path, "derivatives/functionalROIs/sub-NSxLxPQx1973/ses-04/anat", "sub-NSxLxPQx1973_transform_from-fs_to-ACPC_fwdtransforms"))


mytx = reg['fwdtransforms']


# Step 5: Resample ROIs to diffusion (ACPC) space

#load binary masks
FSLHMT = op.join(proj_path,"derivatives/freesurfer", participant,"ses-04/anat/"+participant+"_ses-04_space-fsnative_desc-lhMT_mask.nii.gz")  # Replace with your atlas file
fs_lhmt_nii = ants.image_read(FSLHMT)

FSLHPT = op.join(proj_path,"derivatives/freesurfer", participant,"ses-04/anat/"+participant+"_ses-04_space-fsnative_desc-lhPT_mask.nii.gz")  # Replace with your atlas file
fs_lhpt_nii = ants.image_read(FSLHPT)


roi_lhmt_mask_warped = ants.apply_transforms( fixed = qsiprep_t1_nii, 
                                       moving = fs_lhmt_nii, 
                                       transformlist = mytx,                                       
                                       interpolator  = 'genericLabel', 
                                       ) #whichtoinvert = [True, False]
# Save the ROI mask 
lhmt_mask_path = op.join(proj_path,"derivatives/freesurfer", participant,"ses-04/anat/"+participant+"_ses-04_space-ACPC_desc-lhMT_mask.nii.gz")
ants.image_write(roi_lhmt_mask_warped, lhmt_mask_path)

##second roi

roi_lhpt_mask_warped = ants.apply_transforms( fixed = qsiprep_t1_nii, 
                                       moving = fs_lhpt_nii, 
                                       transformlist = mytx,                                       
                                       interpolator  = 'genericLabel', 
                                       ) #whichtoinvert = [True, False]
# Save the ROI mask 
lhpt_mask_path = op.join(proj_path,"derivatives/freesurfer", participant,"ses-04/anat/"+participant+"_ses-04_space-ACPC_desc-lhPT_mask.nii.gz")
ants.image_write(roi_lhpt_mask_warped, lhpt_mask_path)


#T1 
#Resample T1


mov_fs_t1 = ants.apply_transforms( fixed = qsiprep_t1_nii, 
                                       moving = fs_t1_nii, 
                                       transformlist = mytx,                                       
                                       interpolator  = 'genericLabel', 
                                       ) #whichtoinvert = [True, False]

mov_resampled_T1_path = op.join(proj_path,"derivatives/freesurfer", participant,"ses-04/anat/"+participant+"_ses-04_desc-freesurfertoACPC_T1w.nii.gz")
ants.image_write(mov_fs_t1, mov_resampled_T1_path)


# ############ Create a White Matter mask #####
# #First, convert it to nifti format
# wm_path = op.join("/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/freesurfer",participant,"mri","wm.mgz")
# wm_copy = op.join(proj_path, "derivatives/freesurfer",participant,"ses-04/anat","wm.nii.gz")

# freesurferCommand = f'mri_convert {wm_path} {wm_copy}'
# os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')

# #Then, create binary mask
# labels_img = ants.image_read(os.path.expanduser(wm_copy))
# white_matter = labels_img.numpy() == 110  # Boolean mask

# #Third, save binary mask
# wm_mask_img = ants.from_numpy(white_matter.astype(np.uint8), origin=labels_img.origin, spacing=labels_img.spacing, direction=labels_img.direction)
# # wm_mask_path = op.join(proj_path, "derivatives/freesurfer",participant,"ses-04/anat",participant+"_desc-wm_mask.nii.gz")
# # ants.image_write(wm_mask_img, wm_mask_path)

# #Resample WM mask from individual space to diffusion space (ACPC)

################## Run pyAFQ     ##################
bids_path = "/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/tests/pyAFQ_tests/afq-functionalROI2"#op.expanduser("~/Documents/research/ampb_mt_tractometry_analysis/tests/pyAFQ_tests/afq-functionalROI")#anat"
preprocDir = op.join(bids_path, 'derivatives/qsiprep')
subjNames = [f for f in os.listdir(preprocDir) if f not in {".DS_Store", "dataset_description.json", "dwiqc.json","logs",".bidsignore"} and not f.endswith((".html"))]
# Loop through each subject folder in the directory
for participant in subjNames:
    #First we have to look for the diffusion files of interest for this subject (dwi, bval, bvec)
    diff_path = op.join(preprocDir, participant, 'ses-04', 'dwi')
    patterns = [
    "_desc-preproc_dwi.nii.gz",
    "_desc-preproc_dwi.bval",
    "_desc-preproc_dwi.bvec"]

    # Find files
    matched_files = {pattern: [] for pattern in patterns}

    for file in os.listdir(diff_path):
        for pattern in patterns:
            if file.endswith(pattern):
                matched_files[pattern].append(op.join(diff_path, file))
    dwi_path = matched_files[patterns[0]][0]
    bval_path = matched_files[patterns[1]][0]
    bvec_path = matched_files[patterns[2]][0]

    out_dir = op.join(bids_path, 'derivatives/afq/', participant)
    os.makedirs(out_dir, exist_ok=True)

    #ROI paths
    lh_mt_path = op.join(bids_path, "derivatives/freesurfer",participant,"ses-04/anat/",participant+"_ses-04_desc-lhMT_mask.nii.gz") 
    lh_pt_path = op.join(bids_path, "derivatives/freesurfer",participant,"ses-04/anat/",participant+"_ses-04_desc-lhPT_mask.nii.gz") 

#Here, ROis must be provided in the same space as subject's. Here must be ACPC space from qsiprep
    bundle_kwargs = {
        "cross_midline": False,
        "space": "subject"
    } #        "qb_thresh": 4,
    bundles = abd.BundleDict({
    "L_MT_PT": {
        "start": lh_mt_path,
        "end": lh_pt_path,
        **bundle_kwargs
        }})  #need to resample ROIs to dwi data, as dwi has different dimensions than qsiprep T1w despite being in the same space. the deprecated  resample_subject_to= dwi_path does not need to be specified anymore

#Might need to use globglob for the paths
    brain_mask_definition = ImageFile(path = op.join(bids_path, "derivatives/qsiprep",participant,"ses-04/dwi/",participant+"_ses-04_acq-HCPdir99_space-ACPC_desc-brain_mask.nii.gz")) 
    scalars = [
    "dki_fa", "dki_md", "dki_mk", "dki_awf", 
    "fwdti_fa", "fwdti_md", "fwdti_fwf"
    ]
    
    tracking_params = {
        "seed_mask": RoiImage(use_endpoints = True), 
        "n_seeds": 10, 
    }
    # tracking_params = dict(stop_mask=brain_mask_definition,
    #                         stop_threshold=0.05,
    #                         n_seeds=200000,
    #                         rng_seed=42,
    #                         random_seeds = True ) # ScalarImage("dki_fa") seed_mask=RoiImage(use_endpoints=True) stop_threshold=0
    myafq = ParticipantAFQ(
        dwi_data_file =         dwi_path,
        bval_file =             bval_path,
        bvec_file =             bvec_path,
        output_dir =            out_dir,
        bundle_info =           bundles,
        scalars =               scalars,
        tracking_params =       tracking_params,
        brain_mask_definition = brain_mask_definition) #tracking_params=tracking_params,
    myafq.cmd_outputs(cmd = "rm")
    bundle_html = myafq.export_all()
# bundle_html = myafq.export("indiv_bundles_figures")
# bundle_html = myafq.export_all()
# bundle_html = myafq.export("all_bundles_figure")
# bundle_html = myafq.export("bundles")
    plotly.io.show(bundle_html["01"]["L_MT_PT"])





