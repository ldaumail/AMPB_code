import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation


def overlap_masks(input_files, output_file):
  # create image with all input files multiplied together (overlap)
  info_image   = nib.load(input_files[0])
  output_image = np.ones(info_image.shape)
  for input_file in input_files: # for each input file
    output_image *= nib.load(input_file).get_fdata()
  
  # save the output image
  output_image = (output_image > 0) * 1.0 # convert to binary mask
  output_image = nib.Nifti1Image(output_image, affine = info_image.affine)
  nib.save(output_image, output_file)
  print(f"Saved: {output_file}")


if __name__ == "__overlap_masks__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_files", type = str, nargs = "+")
  parser.add_argument("--output_file", type = str)
  args = parser.parse_args()

  overlap_masks(
    input_files = args.input_files,
    output_file = args.output_file,
  )