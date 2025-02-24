#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 folder_list.txt"
    exit 1
fi

# Define the directory containing folders
target_dir="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis"

# Loop through all folders in the target directory
for folder in "$target_dir"/*; do
    folder_name=$(basename "$folder")

    # Check if the folder name exists in the provided list
    if ! grep -qx "$folder_name" "$1"; then
        echo "Removing unused folder: $folder_name"
        rm -rf "$folder"
    fi
done

echo "Cleanup complete!"
