#!/bin/bash

# Paths
src_base="/Volumes/cos-lab-wpark78/AMPB"
dst_base="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/fMRI_data"
participant_list="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils/study2_subjects_updated.txt"   # <-- change this path!

# Loop over participants in the text file
while read -r sub_id; do
    # Skip empty lines or comments
    [[ -z "$sub_id" || "$sub_id" =~ ^# ]] && continue

    dst_dir="$dst_base/$sub_id/eventfiles"
    mkdir -p "$dst_dir"

    # Determine session(s)
    if [ "$sub_id" == "sub-NSxLxQUx1953" ]; then
        sessions=("ses-05")
    else
        sessions=("ses-01" "ses-01b")
    fi

    # Copy event files
    for ses in "${sessions[@]}"; do
        src_path="$src_base/$sub_id/$ses/func"
        if [ -d "$src_path" ]; then
            cp "$src_path"/*_events.tsv "$dst_dir"/ 2>/dev/null
        fi
    done

    echo "Copied events for $sub_id → $dst_dir"
done < "$participant_list"
