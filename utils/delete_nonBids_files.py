#Adjust format to bids by deleting some files
#Loic Daumail 11/22/2024

import os
import re

# Base directory
base_dir = "/Users/ldaumail3/Documents/Research/AMPB_MT_tractometry_analysis/AMPB"

# Regular expression pattern to match .mat files ending with YYYYMMDD_HHMM
pattern = re.compile(r"_\d{8}_\d{4}\.mat$")

# Loop through all subject directories
for subject in os.listdir(base_dir):
    subject_dir = os.path.join(base_dir, subject)
    
    # Ensure it's a directory
    if os.path.isdir(subject_dir):
        for session in os.listdir(subject_dir):
            session_dir = os.path.join(subject_dir, session)
            
            # Ensure it's a directory
            if os.path.isdir(session_dir):
                # REMOVE .mat experiment info files
                func_dir = os.path.join(session_dir, "func")
                
                # Check if "func" directory exists
                if os.path.exists(func_dir):
                    for file in os.listdir(func_dir):
                        file_path = os.path.join(func_dir, file)
                        
                        # Match files with the pattern and remove them
                        if pattern.search(file):
                            print(f"Removing: {file_path}")
                            os.remove(file_path)
                #REMOVE MPM files (not needed for diffusion and fmri analysis)
                anat_dir = os.path.join(session_dir, "anat")
                
                # Check if "anat" directory exists
                if os.path.exists(anat_dir):
                    for file in os.listdir(anat_dir):
                        file_path = os.path.join(anat_dir, file)
                        
                        # Check if "MPM" is in the filename and remove the file
                        if "MPM" in file:
                            print(f"Removing: {file_path}")
                            os.remove(file_path)

#Remove sub-EBxGxCCx1986_ses-04_acq-headMTw_RB1COR files
target_base_name = "sub-EBxGxCCx1986_ses-04_acq-headMTw_RB1COR"

# Loop through all subject directories
for subject in os.listdir(base_dir):
    subject_dir = os.path.join(base_dir, subject)
    
    # Ensure it's a directory
    if os.path.isdir(subject_dir):
        for session in os.listdir(subject_dir):
            session_dir = os.path.join(subject_dir, session)
            
            # Ensure it's a directory
            if os.path.isdir(session_dir):
                fmap_dir = os.path.join(session_dir, "fmap")
                
                # Check if "fmap" directory exists
                if os.path.exists(fmap_dir):
                    for file in os.listdir(fmap_dir):
                        # Check if the file starts with the target base name
                        if file.startswith(target_base_name):
                            file_path = os.path.join(fmap_dir, file)
                            print(f"Removing: {file_path}")
                            os.remove(file_path)

#Remove more files: 

# File pattern to match
target_pattern = "acq-siemensGre_phasediff"

# Loop through all subject directories
for subject in os.listdir(base_dir):
    subject_dir = os.path.join(base_dir, subject)
    
    # Ensure it's a directory
    if os.path.isdir(subject_dir):
        for session in os.listdir(subject_dir):
            session_dir = os.path.join(subject_dir, session)
            
            # Ensure it's a directory
            if os.path.isdir(session_dir):
                fmap_dir = os.path.join(session_dir, "fmap")
                
                # Check if "fmap" directory exists
                if os.path.exists(fmap_dir):
                    for file in os.listdir(fmap_dir):
                        # Check if the file contains the target pattern
                        if target_pattern in file:
                            file_path = os.path.join(fmap_dir, file)
                            print(f"Removing: {file_path}")
                            os.remove(file_path)


# File pattern to match

target_pattern = "TB1EPI.json"

# Loop through all subject directories
for subject in os.listdir(base_dir):
    subject_dir = os.path.join(base_dir, subject)
    
    # Ensure it's a directory
    if os.path.isdir(subject_dir):
        for session in os.listdir(subject_dir):
            session_dir = os.path.join(subject_dir, session)
            
            # Ensure it's a directory
            if os.path.isdir(session_dir):
                fmap_dir = os.path.join(session_dir, "fmap")
                
                # Check if "fmap" directory exists
                if os.path.exists(fmap_dir):
                    for file in os.listdir(fmap_dir):
                        # Check if the file contains the target pattern
                        if target_pattern in file:
                            file_path = os.path.join(fmap_dir, file)
                            print(f"Removing: {file_path}")
                            os.remove(file_path)