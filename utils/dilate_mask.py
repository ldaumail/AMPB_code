import argparse
import nibabel as nib
from scipy.ndimage import binary_dilation


def dilate_mask(input_file, output_file, dilate = 1):
  # load the input file
  input_image = nib.load(input_file)

  # dilate the mask with a binary dilation
  mask_dilated = binary_dilation(input_image.get_fdata(), iterations = dilate)

  # save the dilated mask
  mask_dilated = nib.Nifti1Image(mask_dilated * 1.0, affine = input_image.affine)
  nib.save(mask_dilated, output_file)
  print(f"Saved: {output_file}")


if __name__ == "__dilate_mask__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type = str)
  parser.add_argument("--output_file", type = str)
  parser.add_argument("--dilate", type = int, default = 1)
  args = parser.parse_args()

  dilate_mask(
    input_file  = args.input_file,
    output_file = args.output_file,
    dilate      = args.dilate
  )