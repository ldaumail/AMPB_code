import shutil
import os
##Remove folders not in study list
# Define the paths
parent_directory = "/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/fonctional_roi"
allowed_file = "/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils/study2_subjects.txt"

# Read the allowed folder names from the text file
with open(allowed_file, 'r') as f:
    allowed_folders = set(line.strip() for line in f)

# Iterate over all folders in the parent directory
for folder_name in os.listdir(parent_directory):
    folder_path = os.path.join(parent_directory, folder_name)

    # Check if the path is a directory and not in the allowed list
    if os.path.isdir(folder_path) and folder_name not in allowed_folders:
        print(f"Removing folder: {folder_path}")
        shutil.rmtree(folder_path)  # Use shutil.rmtree(folder_path) if the folder is not empty
