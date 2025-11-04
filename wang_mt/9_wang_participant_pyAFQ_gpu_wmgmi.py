#Run pyAFQ in subject ACPC space
#08/20/2025
import re
import argparse
import AFQ.api.bundle_dict as abd
from AFQ.api.participant import ParticipantAFQ
from AFQ.definitions.image import ImageFile, RoiImage

BUNDLES_KWARGS = {
  "cross_midline": False,
  "space": "subject",
}

def main(dwi_data_file, bval_file, bvec_file, t1_file, roi_files, output_dir):
  # define custom bundles dictionary
  bundles = {} # intialize empty bundles dictionary
  for start_file, end_file in roi_files: # for each roi (start/stop) file  
    # define tract label from start and end files 
    hemisphere  = re.sub(".+_hemi-([LR])_.*", "\\1", start_file)
    hemisphere  = "Left" if hemisphere == "L" else "Right"
    start_label = re.sub(".+_label-(\\w+)_.+", "\\1", start_file)
    end_label   = re.sub(".+_label-(\\w+)_.+", "\\1", end_file)
    tract_label = f"{hemisphere}{start_label}x{end_label}"
    
    # add tract to bundles dictionary
    bundles[tract_label] = {
      "start": start_file,
      "end": end_file,
      **BUNDLES_KWARGS
    }
  # define bundles as BundleDict
  bundles = abd.BundleDict(bundles, resample_subject_to=True)


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
  segmentation_params = {
        "cleaning_params": {"distance_threshold": 3, "clean_rounds": 2}
        } #"dist_to_atlas": 0 
       
  # define ParticipantAFQ object
  myafq = ParticipantAFQ(
    dwi_data_file         = dwi_data_file, 
    bval_file             = bval_file,
    bvec_file             = bvec_file,
    t1_file               = t1_file,
    output_dir            = output_dir,
    bundle_info           = bundles,
    tracking_params       = tracking_params, 
    segmentation_params   = segmentation_params,
    tractography_ngpus    = 1
  )
  
  # call export_all, starts tractography
  myafq.export_all(xforms = False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dwi_data_file", type = str)
  parser.add_argument("--bval_file", type = str)
  parser.add_argument("--bvec_file", type = str)
  parser.add_argument("--t1_file", type = str)
  parser.add_argument("--roi_files", type = str, nargs = 2, action = "append")
  parser.add_argument("--output_dir", type = str)
  args = parser.parse_args()
  
  main(
    dwi_data_file  = args.dwi_data_file,
    bval_file      = args.bval_file, 
    bvec_file      = args.bvec_file,
    t1_file        = args.t1_file,  
    roi_files      = args.roi_files,
    output_dir     = args.output_dir
  )

    
