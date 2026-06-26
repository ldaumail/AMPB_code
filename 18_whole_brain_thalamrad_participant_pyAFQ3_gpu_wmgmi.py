#Run pyAFQ in subject ACPC space
#08/20/2025
import re
import os.path as op
import argparse
import AFQ.api.bundle_dict as abd
from AFQ.api.participant import ParticipantAFQ
from AFQ.definitions.image import ImageFile, RoiImage
import AFQ.data.fetch as afd

BUNDLES_KWARGS = {
  "cross_midline": False,
  "space": "subject",
}

def main(dwi_data_file, bval_file, bvec_file, t1_file, template_dir, participant, output_dir):
  # define custom bundles dictionary
  clean_rounds = 15
  distance_threshold = 3
  bundles = abd.BundleDict({
    "L_PTR": {
      "include": [
          op.join(template_dir, 'AICHA_space-ACPC_rois', participant, participant+'_hemi-L_space-ACPC_desc-thalamus_mask.nii.gz'),
          op.join(template_dir, 'AICHA_space-ACPC_rois', participant, participant+'_hemi-L_space-ACPC_desc-posterior_mask.nii.gz')],
    #   "exclude": [
    #       template_dir + 'SLFt_roi2_L.nii.gz'],

      "cross_midline": False,

      "mahal": {
          "clean_rounds": clean_rounds,
          "length_threshold": 4,
          "distance_threshold": distance_threshold}
  },
  "L_STR": {
      "include": [
          op.join(template_dir, 'AICHA_space-ACPC_rois', participant, participant+'_hemi-L_space-ACPC_desc-thalamus_mask.nii.gz'),
          op.join(template_dir, 'AICHA_space-ACPC_rois', participant, participant+'_hemi-L_space-ACPC_desc-superior_mask.nii.gz')],
      # "exclude": [
      #     template_dir + 'SLFt_roi2_L.nii.gz'],

      "cross_midline": False,

      "mahal": {
          "clean_rounds": clean_rounds,
          "length_threshold": 4,
          "distance_threshold": distance_threshold}
  },
  "R_PTR": {
      "include": [
          op.join(template_dir, 'AICHA_space-ACPC_rois', participant, participant+'_hemi-R_space-ACPC_desc-thalamus_mask.nii.gz'),
          op.join(template_dir, 'AICHA_space-ACPC_rois', participant, participant+'_hemi-R_space-ACPC_desc-posterior_mask.nii.gz')],
      # "exclude": [
      #     template_dir + 'SLFt_roi2_L.nii.gz'],

      "cross_midline": False,

      "mahal": {
          "clean_rounds": clean_rounds,
          "length_threshold": 4,
          "distance_threshold": distance_threshold}
  },
  "R_STR": {
      "include": [
          op.join(template_dir, 'AICHA_space-ACPC_rois', participant, participant+'_hemi-R_space-ACPC_desc-thalamus_mask.nii.gz'),
          op.join(template_dir, 'AICHA_space-ACPC_rois', participant, participant+'_hemi-R_space-ACPC_desc-superior_mask.nii.gz')],
      # "exclude": [
      #     template_dir + 'SLFt_roi2_L.nii.gz'],

      "cross_midline": False,

      "mahal": {
          "clean_rounds": clean_rounds,
          "length_threshold": 4,
          "distance_threshold": distance_threshold}
  },
#   "L_MT-PTxSTS1": {
#       "include": [
#           op.join(template_dir, 'julich_space-ACPC_rois', participant, 'ses-concat','anat', participant+'_hemi-L_space-ACPC_desc-PTxSTS1_mask.nii.gz'),
#           op.join(template_dir, 'wang_space-ACPC_rois', participant, participant+'_hemi-L_space-ACPC_label-MT_mask_dilated.nii.gz')],
#     #   "exclude": [
#     #       template_dir + 'SLFt_roi2_L.nii.gz'],

#       "cross_midline": False,

#       "mahal": {
#           "clean_rounds": clean_rounds,
#           "length_threshold": 4,
#           "distance_threshold": distance_threshold}
#   },
#   "L_MT-LGNxPU": {
#       "include": [
#           op.join(template_dir, 'julich_space-ACPC_rois', participant, 'ses-concat','anat', participant+'_hemi-L_space-ACPC_desc-LGNxPU_mask.nii.gz'),
#           op.join(template_dir, 'wang_space-ACPC_rois', participant, participant+'_hemi-L_space-ACPC_label-MT_mask_dilated.nii.gz')],
#       # "exclude": [
#       #     template_dir + 'SLFt_roi2_L.nii.gz'],

#       "cross_midline": False,

#       "mahal": {
#           "clean_rounds": clean_rounds,
#           "length_threshold": 4,
#           "distance_threshold": distance_threshold}
#   },
#     "L_MT-FEF": {
#       "include": [
#           op.join(template_dir, 'julich_space-ACPC_rois', participant, 'ses-concat','anat', participant+'_hemi-L_space-ACPC_desc-FEF_mask.nii.gz'),
#           op.join(template_dir, 'wang_space-ACPC_rois', participant, participant+'_hemi-L_space-ACPC_label-MT_mask_dilated.nii.gz')],
#       # "exclude": [
#       #     template_dir + 'SLFt_roi2_L.nii.gz'],

#       "cross_midline": False,

#       "mahal": {
#           "clean_rounds": clean_rounds,
#           "length_threshold": 4,
#           "distance_threshold": distance_threshold}
#   },
#   "R_MT-PTxSTS1": {
#       "include": [
#           op.join(template_dir, 'julich_space-ACPC_rois', participant, 'ses-concat','anat', participant+'_hemi-R_space-ACPC_desc-PTxSTS1_mask.nii.gz'),
#           op.join(template_dir, 'wang_space-ACPC_rois', participant, participant+'_hemi-R_space-ACPC_label-MT_mask_dilated.nii.gz')],
#       # "exclude": [
#       #     template_dir + 'SLFt_roi2_L.nii.gz'],

#       "cross_midline": False,

#       "mahal": {
#           "clean_rounds": clean_rounds,
#           "length_threshold": 4,
#           "distance_threshold": distance_threshold}
#   },
#   "R_MT-LGNxPU": {
#       "include": [
#           op.join(template_dir, 'julich_space-ACPC_rois', participant, 'ses-concat','anat', participant+'_hemi-R_space-ACPC_desc-LGNxPU_mask.nii.gz'),
#           op.join(template_dir, 'wang_space-ACPC_rois', participant, participant+'_hemi-R_space-ACPC_label-MT_mask_dilated.nii.gz')],
#       # "exclude": [
#       #     template_dir + 'SLFt_roi2_L.nii.gz'],

#       "cross_midline": False,

#       "mahal": {
#           "clean_rounds": clean_rounds,
#           "length_threshold": 4,
#           "distance_threshold": distance_threshold}
#   },
#   "R_MT-FEF": {
#       "include": [
#           op.join(template_dir, 'julich_space-ACPC_rois', participant, 'ses-concat','anat', participant+'_hemi-R_space-ACPC_desc-FEF_mask.nii.gz'),
#           op.join(template_dir, 'wang_space-ACPC_rois', participant, participant+'_hemi-R_space-ACPC_label-MT_mask_dilated.nii.gz')],
#       # "exclude": [
#       #     template_dir + 'SLFt_roi2_L.nii.gz'],

#       "cross_midline": False,

#       "mahal": {
#           "clean_rounds": clean_rounds,
#           "length_threshold": 4,
#           "distance_threshold": distance_threshold}
#   }
  })

  bundles = bundles + abd.default_bd() + abd.slf_bd() + abd.callosal_bd()
#  bundles = abd.default_bd() + abd.slf_bd() + abd.callosal_bd()

#   scalars = [
#             "dki_fa", "dki_md", "dki_mk", "dki_awf", 
#             "fwdti_fa", "fwdti_md", "fwdti_fwf"
#         ]

  # define tracking parameters
  tracking_params = {
    "n_seeds": 2000000,
    "random_seeds": True, 
    "seed_mask": RoiImage(
            use_waypoints=False,
            use_endpoints=True,
            only_wmgmi=True), 
    "trx": True
  }


  # define segmentation parameters
  #"dist_to_atlas": 0, "cleaning_params": {"distance_threshold": 3}
  #"dist_to_atlas" specifies the distance from the target ROIs that tracts need to reach. if = 0, tracts need to reach the surface of ROI, or enter it. If 4 mm = needs to be within 4mm of ROI surface. 
  #"distance_threshold" in cleaning params is the Mahalanobis distance in number of STDEVs. We adjust it to exclude outlier streamlines.
  # segmentation_params = {
  #       "cleaning_params": {"distance_threshold": 3, "clean_rounds": 2}
  #       } #"dist_to_atlas": 0 
       
  # define ParticipantAFQ object
  myafq = ParticipantAFQ(
    dwi_data_file         = dwi_data_file, 
    bval_file             = bval_file,
    bvec_file             = bvec_file,
    t1_file               = t1_file,
    output_dir            = output_dir,
    bundle_info           = bundles,
    tracking_params       = tracking_params, 
    # segmentation_params   = segmentation_params,
    tractography_ngpus    = 1
  )
  
  ## call export_all, starts tractography
  #myafq.clobber(dependent_on="track") #only remove things related to tractography
  myafq.clobber(dependent_on="recog")
  myafq.export_all(xforms = False)
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dwi_data_file", type = str)
  parser.add_argument("--bval_file", type = str)
  parser.add_argument("--bvec_file", type = str)
  parser.add_argument("--t1_file", type = str)
  parser.add_argument("--template_dir", type = str)
  parser.add_argument("--participant", type = str)
  parser.add_argument("--output_dir", type = str)
  args = parser.parse_args()
  
  main(
    dwi_data_file  = args.dwi_data_file,
    bval_file      = args.bval_file, 
    bvec_file      = args.bvec_file,
    t1_file        = args.t1_file, 
    template_dir   = args.template_dir,
    participant    = args.participant,
    output_dir     = args.output_dir
  )

    
