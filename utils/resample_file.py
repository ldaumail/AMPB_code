import ants
import argparse


def resample_file(input_file, target_file, output_file, interpolator = "linear"):
  # read input and target file
  input_image  = ants.image_read(input_file)
  target_image = ants.image_read(target_file)

  # resample input image to target image
  resampled_image = ants.resample_image_to_target(
    image  = input_image, 
    target = target_image,
    interp_type = interpolator
  )
  ants.image_write(resampled_image, output_file) 
  print(f"Saved: {output_file}")


if __name__ == "__resample_file__": 
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type = str)
  parser.add_argument("--target_file", type = str)
  parser.add_argument("--output_file", type = str)
  parser.add_argument("--interpolator", type = str, default = "linear")
  args = parser.parse_args()

  resample_file(
    input_file   = args.input_file,
    target_file  = args.target_file,
    output_file  = args.output_file, 
    interpolator = args.interpolator
  )