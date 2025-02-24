#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="/Volumes/cos-lab-wpark78/AMPB/analysis"
DEST_DIR="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis"

# Loop through all matching directories in the source directory
for dir in "$SOURCE_DIR"/sub-*; do
    if [ -d "$dir/roi" ]; then
        # Extract the directory name (e.g., sub-EBxGxCCx1986)
        subdir_name=$(basename "$dir")

        # Create the corresponding directory structure in the destination
        mkdir -p "$DEST_DIR/$subdir_name/roi"

        # Copy all files from the roi directory to the new location
        cp -r "$dir/roi"/* "$DEST_DIR/$subdir_name/roi/"

        echo "Copied files from $dir/roi to $DEST_DIR/$subdir_name/roi"
    fi
done

echo "All files copied successfully!"
