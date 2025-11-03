#This script aims to perform ROI based tractography
#Loic Daumail - 30/01/2025
import os
import os.path as op
import shutil
import matplotlib.pyplot as plt
import nibabel as nib
import plotly
import numpy as np
import sys

utils = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils'
sys.path.append(op.expanduser(f'{utils}'))

from AFQ.api.group import GroupAFQ
import AFQ.data.fetch as afd
from AFQ.definitions.image import RoiImage
import AFQ.api.bundle_dict as abd

qsiprep_path = os.path.expanduser("~/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep")

#########################################################################
################################ ROIs ###################################
# Create ROI directory (bids compatible) and add ROIs in it
roi_path = os.path.expanduser("~/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/functionalROIs")
os.makedirs(roi_path, exist_ok=True)
#Add dataset description sidecar file
shutil.copy2(op.join(qsiprep_path, "dataset_description.json"), op.join(roi_path, "dataset_description.json"))

#Functionally defined regions of interest
func_def_path = os.path.expanduser("~/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis")

#loop through participants
bids_path = os.path.expanduser("~/Documents/research/ampb_mt_tractometry_analysis/ampb")#anat"
preprocDir = op.join(bids_path, 'derivatives/qsiprep')
subjNames = [f for f in os.listdir(preprocDir) if f not in {".DS_Store", "dataset_description.json", "dwiqc.json","logs",".bidsignore"} and not f.endswith((".html"))]

#Copy scans/ROIs of interest to ROI dir
for participant in subjNames:

    freesurfer_subject_folder = os.path.expanduser(op.join("~/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/freesurfer", participant, "mri"))
    for session in ["ses-03", "ses-04"]:
 # Copy and rename anatomical to freesurfer ROI folder
        roi_mask_path = op.join(roi_path, participant,session, "anat")
        os.makedirs(roi_mask_path, exist_ok=True) 
        FS_T1 = op.join(freesurfer_subject_folder, "T1.mgz")
        FS_T1_copy = op.join(roi_mask_path, participant+"_"+session+"_desc-freesurfer_T1w.nii.gz")
        # Run mri_convert
        freesurferCommand = f'mri_convert {FS_T1} {FS_T1_copy}'
        os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')

 # Copy Roi 
        FS_leftMT = op.join(func_def_path, participant, "roi",  participant+"_hemi-L_space-fsnative_label-MT_mask.nii.gz")
        FS_leftMT_copy = op.join(roi_mask_path, participant+"_"+session+"_space-fsnative_desc-lhMT_mask.nii.gz")
        # Run mri_convert
        freesurferCommand = f'mri_convert {FS_leftMT} {FS_leftMT_copy}'
        os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')

        FS_leftPT = op.join(func_def_path, participant, "roi",  participant+"_hemi-L_space-fsnative_label-PT_mask.nii.gz")
        FS_leftPT_copy = op.join(roi_mask_path, participant+"_"+session+"_space-fsnative_desc-lhPT_mask.nii.gz")
        # Run mri_convert
        freesurferCommand = f'mri_convert {FS_leftPT} {FS_leftPT_copy}'
        os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')


### Get DWI files from qsiprep and create new bids compatible directory
#First create bids compatible directory
dwi_path = os.path.expanduser("~/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep_dwi")
os.makedirs(dwi_path, exist_ok=True) 
#Add dataset description sidecar file
shutil.copy2(op.join(qsiprep_path, "dataset_description.json"), op.join(dwi_path, "dataset_description.json"))

for participant in subjNames:
     for session in ["ses-04"]: #"ses-03",
        qsiprep_subject_folder = os.path.expanduser(op.join(qsiprep_path, participant, session, "dwi"))
        dwi_subject_path = op.join(dwi_path, participant,session, "dwi")
        os.makedirs(dwi_subject_path, exist_ok=True) 
        #Copy dwi files to new directory
        dwi_file = op.join(qsiprep_subject_folder, participant+"_"+session+"_acq-HCPdir99_space-ACPC_desc-preproc_dwi.nii.gz")
        dwi_file_copy = op.join(dwi_subject_path, participant+"_"+session+"_acq-HCPdir99_space-ACPC_desc-preproc_dwi.nii.gz")
        shutil.copy2(dwi_file, dwi_file_copy)
        #Bval
        bval_file = op.join(qsiprep_subject_folder, participant+"_"+session+"_acq-HCPdir99_space-ACPC_desc-preproc_dwi.bval")
        bval_file_copy = op.join(dwi_subject_path, participant+"_"+session+"_acq-HCPdir99_space-ACPC_desc-preproc_dwi.bval")
        shutil.copy2(bval_file, bval_file_copy)
        #Bvec
        bval_file = op.join(qsiprep_subject_folder, participant+"_"+session+"_acq-HCPdir99_space-ACPC_desc-preproc_dwi.bvec")
        bval_file_copy = op.join(dwi_subject_path, participant+"_"+session+"_acq-HCPdir99_space-ACPC_desc-preproc_dwi.bvec")
        shutil.copy2(bval_file, bval_file_copy)

    ################## Run pyAFQ     ##################
bids_path = os.path.expanduser("~/Documents/research/ampb_mt_tractometry_analysis/ampb")#anat"

#Set tractography parameters
tracking_params = dict(n_seeds=10000,
                    random_seeds=True,
                    rng_seed=42,
                    seed_mask=RoiImage(use_endpoints=True))
#Define custom bundle dict
bundles = abd.BundleDict({
    "L_OR": {
        "start": {
            "scope": "functionalROIs",
            "suffix": "mask",
            "desc": "lhMT"
        },
        "end": {
            "scope": "functionalROIs",
            "suffix": "mask",
            "desc": "lhPT"
        },
        "cross_midline": False,
        "space": "subject"
    }})
#Initialize groupAFQ object
myafq = GroupAFQ(
    bids_path=bids_path,
    preproc_pipeline='qsiprep_dwi',
    tracking_params=tracking_params,
    bundle_info=bundles)

bundle_html = myafq.export("indiv_bundles_figures")
plotly.io.show(bundle_html["01"]["L_OR"])




###Remove files
# target_pattern = "T1w"

# Loop through all subject directories
# for participant in subjNames:
#     for session in ["ses-03", "ses-04"]:
#         subject_dir = os.path.join(preprocDir, participant, session, "anat")
        
#         # Ensure it's a directory
#         if os.path.isdir(subject_dir):  
#             for file in os.listdir(subject_dir):
#                 # Check if the file starts with the target base name
#                 if target_pattern in file:
#                     file_path = os.path.join(subject_dir, file)
#                     print(f"Removing: {file_path}")
#                     os.remove(file_path)

# ### Remove folders and contents
# import shutil 
# for participant in subjNames:
#     for session in ["ses-03", "ses-04"]:
#         subject_dir = os.path.join(preprocDir, participant, session, "anat")

#         if os.path.exists(subject_dir):
#             shutil.rmtree(subject_dir)

########
#     freesurfer_subject_folder = os.path.expanduser(op.join(
#         "~", "Documents", "research", "ampb_mt_tractometry_analysis",
#         "ampb", "derivatives", "freesurfer",
#         participant,
#         "mri"))

#     seg_file = nib.load(op.join(
#         freesurfer_subject_folder, "aparc.a2009s+aseg.mgz"))
#     left_MT = seg_file.get_fdata() == 1015
#     nib.save(
#         nib.Nifti1Image(
#             left_MT.astype(np.float32),
#             seg_file.affine),
#         op.join(
#             freesurfer_subject_folder,
#             participant+"_leftMT_mask.nii.gz"))

# # Fetch LV1 ROI
# # which was already generated using the process above
# afd.fetch_stanford_hardi_lv1()

# import os.path as 

# import plotly
# import numpy as np
# import shutil

# from AFQ.api.group import GroupAFQ
# import AFQ.api.bundle_dict as abd
# from AFQ.definitions.image import ImageFile, RoiImage
# np.random.seed(1234)

# ROI_DIR = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/work/sub-NSxGxHNx1952/MT_roi_glasser_mask_space-ACPC.nii.gz'
# BIDS_DIR = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb'
# bundles = abd.BundleDict({
#     "MT": {
#         "include": [
#             ROI_DIR],
#         "exclude": [
#            ],

#         "cross_midline": False,

#         "mahal": {
#             "clean_rounds": 20,
#             "length_threshold": 4,
#             "distance_threshold": 2}
#     }})

# brain_mask_definition = ImageFile(
#     suffix="mask",
#     filters={'desc': 'brain',
#              'space': 'T1w',
#              'scope': 'qsiprep'})

# my_afq = GroupAFQ(
#     bids_path=BIDS_DIR,
#     preproc_pipeline="qsiprep",
#     output_dir=op.join(BIDS_DIR, "derivatives", "afq_MT"),
#     brain_mask_definition=brain_mask_definition,
#     tracking_params={"n_seeds": 10,
#                      "directions": "prob",
#                      "odf_model": "CSD",
#                      "seed_mask": RoiImage()},
#     segmentation_params={"parallel_segmentation": {"engine": "serial"}},
#     bundle_info=bundles)

# my_afq.export_all()
