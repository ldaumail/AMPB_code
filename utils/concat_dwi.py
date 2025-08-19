import re
import glob
import argparse
import numpy as np
import os
import os.path as op
import nibabel as nib
# bids_path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb'
# participant = 'sub-NSxGxBAx1970'
# qsiprep_dir =  op.join(bids_path, 'derivatives', 'qsiprep', participant)
# out_path = op.join(bids_path, 'derivatives', 'qsiprep', participant, 'ses-concat')
# os.makedirs(out_path, exist_ok=True)
# out_prefix =  op.join(out_path, participant+'_ses-concat_acq-HCPdir99_space-ACPC_desc-preproc_dwi') 

def concat_sessions(qsiprep_dir, output_prefix):  
  # define diffusion image pattern for concatenation
  dwi_pattern = "*_desc-preproc_dwi"

  # locate all diffusion images, concatenate, and save to file
  dwi_list   = glob.glob(op.join(qsiprep_dir, "ses-*", "dwi", f"*_{dwi_pattern}.nii.gz"))
  dwi_list   = [x for x in dwi_list if "ses-concat" not in x]
  dwi_image  = [nib.load(x) for x in dwi_list]
  dwi_concat = nib.funcs.concat_images(dwi_image, axis = 3)
  dwi_concat.to_filename(f"{output_prefix}.nii.gz")
  print(f"Saved: {output_prefix}.nii.gz")

  # locate all bval files and combine the file contents
  bval_list   = [x.replace(".nii.gz", ".bval") for x in dwi_list]
  bval_concat = np.concatenate([np.loadtxt(x) for x in bval_list], axis = 0)
  np.savetxt(f"{output_prefix}.bval", bval_concat, fmt = "%d")
  print(f"Saved: {output_prefix}.bval")

  # locate all bvec files and combine the file contents
  bvec_list   = [x.replace(".nii.gz", ".bvec") for x in dwi_list]
  bvec_concat = np.concatenate([np.loadtxt(x) for x in bvec_list], axis = 1)
  np.savetxt(f"{output_prefix}.bvec", bvec_concat, fmt = "%.8f")
  print(f"Saved: {output_prefix}.bvec")


if __name__ == "__concat_sessions__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--qsiprep_dir", type = str)
  parser.add_argument("--output_file", type = str)
  args = parser.parse_args()

  concat_sessions( qsiprep_dir = args.qsiprep_dir, output_file = args.output_file)


  # ses03img = nib.load('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-EBxGxCCx1986/ses-03/dwi/sub-EBxGxCCx1986_ses-03_acq-HCPdir99_space-ACPC_desc-preproc_dwi.nii.gz')
  # ses03img.affine 
  # ses04img = nib.load('/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep/sub-EBxGxCCx1986/ses-04/dwi/sub-EBxGxCCx1986_ses-04_acq-HCPdir99_space-ACPC_desc-preproc_dwi.nii.gz')
  # ses04img.affine