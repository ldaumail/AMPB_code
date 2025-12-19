#!/bin/bash

# Check if input file is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 participants.txt"
  exit 1
fi

PARTICIPANT_FILE=$1

SOURCE_DIR="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/ROIs/julich_space-ACPC_rois"
DEST_DIR="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/ROIs/new_julich_space-ACPC_rois"

HEMIS=("L" "R")
LABELS=("LGNxPU" "PTxSTS1")

while IFS= read -r participant; do
  [ -z "$participant" ] && continue

  SRC_DIR="$SOURCE_DIR/$participant/ses-concat/anat"
  DST_DIR="$DEST_DIR/$participant"

  if [ ! -d "$SRC_DIR" ]; then
    echo "ROI directory not found for $participant"
    continue
  fi

  mkdir -p "$DST_DIR"

  for hemi in "${HEMIS[@]}"; do
    for label in "${LABELS[@]}"; do
      file="${participant}_hemi-${hemi}_space-ACPC_label-${label}_mask.nii.gz"

      if [ -f "$SRC_DIR/$file" ]; then
        cp "$SRC_DIR/$file" "$DST_DIR/"
        echo "Copied $file"
      else
        echo "Missing $file"
      fi
    done
  done

done < "$PARTICIPANT_FILE"

echo "All done!"




# Define the source and destination directories
# SOURCE_DIR="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/ROIs/julich_space-ACPC_rois"
# DEST_DIR="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/ROIs/new_julich_space-ACPC_rois"

# Loop through each participant ID in the text file
# while IFS= read -r participant_id; do
  # Skip empty lines
#  [ -z "$participant_id" ] && continue

  # Define source roi directory
#  SRC_ROI_DIR="$SOURCE_DIR/$participant_id/ses-concat/anat"

#  if [ -d "$SRC_ROI_DIR" ]; then
    # Make destination directory
#    mkdir -p "$DEST_DIR/$participant_id"

    # Copy ROI files
#    cp -r "$SRC_ROI_DIR"/* "$DEST_DIR/$participant_id/"

#    echo "Copied files for $participant_id"
#  else
#    echo "ROI directory not found for $participant_id"
#  fi
#done < "$PARTICIPANT_FILE"

#echo "All done!"


