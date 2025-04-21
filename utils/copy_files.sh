#!/bin/bash
# Check if input file is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 participants.txt"
  exit 1
fi

# Read participant list file
PARTICIPANT_FILE=$1

# Define the source and destination directories
SOURCE_DIR="/Volumes/cos-lab-wpark78/YangYang/analysis"
DEST_DIR="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/functional_surf_roi"

# Loop through each participant ID in the text file
while IFS= read -r participant_id; do
  # Skip empty lines
  [ -z "$participant_id" ] && continue

  # Define source roi directory
  SRC_ROI_DIR="$SOURCE_DIR/$participant_id/roi"

  if [ -d "$SRC_ROI_DIR" ]; then
    # Make destination directory
    mkdir -p "$DEST_DIR/$participant_id"

    # Copy ROI files
    cp -r "$SRC_ROI_DIR"/* "$DEST_DIR/$participant_id/"

    echo "Copied files for $participant_id"
  else
    echo "ROI directory not found for $participant_id"
  fi
done < "$PARTICIPANT_FILE"

echo "All done!"

# # Loop through all matching directories in the source directory
# for dir in "$SOURCE_DIR"/sub-*; do
#     if [ -d "$dir/roi" ]; then
#         # Extract the directory name (e.g., sub-EBxGxCCx1986)
#         subdir_name=$(basename "$dir")

#         # Create the corresponding directory structure in the destination
#         mkdir -p "$DEST_DIR/$subdir_name"

#         # Copy all files from the roi directory to the new location
#         cp -r "$dir/roi"/* "$DEST_DIR/$subdir_name/"

#         echo "Copied files from $dir/roi to $DEST_DIR/$subdir_name/"
#     fi
# done

# echo "All files copied successfully!"
