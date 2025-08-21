import re
import argparse
import AFQ.api.bundle_dict as abd
from AFQ.api.participant import ParticipantAFQ
from AFQ.definitions.image import ImageFile, RoiImage


BUNDLES_KWARGS = {
  "cross_midline": False,
  "space": "subject",
}


def main(dwi_data_file, bval_file, bvec_file, mask_file, stop_mask_file, 
         roi_files, output_dir):
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
  bundles = abd.BundleDict(bundles)

  # define brain main image file
  brain_mask_definition = ImageFile(path = mask_file)
  
  # define tracking parameters
  tracking_params = {
    "n_seeds": 2000000,
    "random_seeds": True, 
    "seed_mask": RoiImage(use_endpoints = True), 
    "stop_mask": ImageFile(path = stop_mask_file),
    "trx": True
  }
  
  # define segmentation parameters
  segmentation_params = {
    "cleaning_params": {
      "clean_rounds": 2, #TODO: adjust as needed
    }
  }

  # define ParticipantAFQ object
  myafq = ParticipantAFQ(
    dwi_data_file         = dwi_data_file, 
    bval_file             = bval_file,
    bvec_file             = bvec_file,
    output_dir            = output_dir,
    bundle_info           = bundles,
    brain_mask_definition = brain_mask_definition, 
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
  parser.add_argument("--mask_file", type = str)
  parser.add_argument("--stop_mask_file", type = str)
  parser.add_argument("--roi_files", type = str, nargs = 2, action = "append")
  parser.add_argument("--output_dir", type = str)
  args = parser.parse_args()
  
  main(
    dwi_data_file  = args.dwi_data_file,
    bval_file      = args.bval_file, 
    bvec_file      = args.bvec_file, 
    mask_file      = args.mask_file, 
    stop_mask_file = args.stop_mask_file,
    roi_files      = args.roi_files,
    output_dir     = args.output_dir
  )