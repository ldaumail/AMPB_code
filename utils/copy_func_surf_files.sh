#!/usr/bin/env bash

# Root BIDS directory
BIDS_ROOT="/Volumes/cos-lab-wpark78/AMPB"
# Output directory
OUT_ROOT="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/fMRI_data"

# Tasks and hemispheres to search for
tasks=("mtlocal" "ptlocal")
hemis=("L" "R")
runs=("1" "2" "3" "4" "5" "6") 
# Define the path to your file containing participant names
participant_list="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils/study2_subjects_updated.txt"

# Use a 'while read' loop to process the file line by line
while read -r participant_name; do
    # ----------------------------------------------------------------------
    # FIX: Define BIDS entity variables based on the current loop variable
    # ----------------------------------------------------------------------
    pname="$participant_name"
    
    # Skip empty lines or comments
    [[ -z "$pname" || "$pname" =~ ^# ]] && continue

    echo "--- Processing $pname ---"

    # Create output directory for this participant
    outdir="$OUT_ROOT/$pname/func"
    mkdir -p "$outdir"

    # Determine session(s)
    if [ "$pname" == "sub-NSxLxQUx1953" ]; then
        sessions=("ses-05")
    else
        sessions=("ses-01" "ses-01b" "ses-02")
    fi

    # Loop through sessions
    for sesname in "${sessions[@]}"; do
        src_path="$BIDS_ROOT/derivatives/fmriprep/$pname/$sesname/func"
        
        # Check if the source directory for the session/func exists
        if [ ! -d "$src_path" ]; then
            echo "⚠️ WARN: Source directory does not exist: $src_path"
            continue # Skip to the next session if the directory is missing
        fi
        
        # Loop over tasks and hemispheres
        for task in "${tasks[@]}"; do
            for hemi in "${hemis[@]}"; do
		for run in "${runs[@]}"; do
                
                	# Define the pattern with the wildcard
                	pattern="${pname}_${sesname}_task-${task}_run-${run}_hemi-${hemi}_space-fsnative_bold.func.gii"
                
                	# --- Explicit Existence Check Logic ---
                
                	# Use find to list files that match the pattern
                	found_files=("$src_path"/$pattern)
                
                	# Check if the list contains only the literal pattern (meaning no files were found)
                	if [[ "${found_files[0]}" == "$src_path/$pattern" && ! -f "$src_path/$pattern" ]]; then
                    	# Report the full path that was searched
                    	echo "❌ MISS: No source files found matching: $src_path/$pattern"
                	else
                    	# Find and copy files
                    	for file in "${found_files[@]}"; do
                        	if [[ -f "$file" ]]; then
                            	echo "✅ FOUND and Copying: $file → $outdir/"
                            	cp "$file" "$outdir/"
                       	 fi
                    	done
                	fi
		done
            done
        done
    done
done < "$participant_list"
echo "Done!"
