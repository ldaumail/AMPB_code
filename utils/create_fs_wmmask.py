import ants
import argparse
import numpy as np
import os.path as op


def create_fs_wmmask(fs_dir, target_brain, target_sample, output_fname):
  # read freesurfer source image and qsiprep target images
  fs_brain      = ants.image_read(op.join(fs_dir, "mri", "brain.mgz"))
  fs_aseg       = ants.image_read(op.join(fs_dir, "mri", "aseg.mgz"))
  target_brain  = ants.image_read(target_brain)  # target anatomical resolution
  target_sample = ants.image_read(target_sample) # target reference sampling

  # coregister freesurfer anatomical to target anatomical
  mtx = ants.registration(
    fixed  = target_brain, 
    moving = fs_brain, 
    type_of_transform = "Rigid"
  )

  # apply coregistration to freesurfer aseg
  coreg_aseg = ants.apply_transforms(
    fixed  = target_brain,
    moving = fs_aseg,
    transformlist = mtx["fwdtransforms"],
    interpolator  = "genericLabel"
  )

  # resample coregistered aseg image to target resolution
  resample_aseg = ants.resample_image_to_target(
    image  = coreg_aseg,
    target = target_sample, 
    interp_type = "genericLabel"
  )

  # binarize aseg to white matter mask
  wm_mask = np.zeros(resample_aseg.shape) # initialize
  keep_labels  = [2, 10, 11, 12, 13, 28,  # left hemisphere
                  41, 49, 50, 51, 52, 60, # right hemisphere
                  77, 251, 252, 253, 254, 255] # WMH and CC
  for label in keep_labels: # for each label to keep
    wm_mask = np.logical_or(wm_mask, resample_aseg.numpy() == label)
  wm_mask = ants.new_image_like(image = resample_aseg, data = wm_mask * 1.0)

  # write freesurfer white matter mask
  ants.image_write(wm_mask, output_fname)
  print(f"Saved: {output_fname}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--fs_dir", type = str)
  parser.add_argument("--target_brain", type = str)
  parser.add_argument("--target_sample", type = str)
  parser.add_argument("--output_fname", type = str)
  args = parser.parse_args()

  create_fs_wmmask(
    fs_dir         = args.fs_dir,
    target_brain   = args.target_brain,
    target_sample  = args.target_sample,
    output_fname   = args.output_fname
  )
