#!/bin/bash

# Check if the input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 participants.txt"
    exit 1
fi

# Define paths
paths_main="/Volumes/cos-lab-wpark78/Loic_backup/ampb"
paths_out="${paths_main}/data"
config_file="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/code/utils/config.json"

# Create the output directory if it doesn't exist
mkdir -p "$paths_out"

# Loop through each participant from the input file
while IFS= read -r participant; do
    echo "Processing participant: $participant"
#    paths_data="${paths_main}/dicom250223/VisDep-AMPB/SUBJECTS/${participant}/SESSIONS"
    dicom_source="${paths_main}/dicom250223/VisDep-AMPB/SUBJECTS/${participant}/SESSIONS"

    # Loop through sessions
    for session in "ses-01" "ses-01b" "ses-02" "ses-03" "ses-04"; do
        echo "  Processing session: $session"
        # Run dcm2bids
        /Users/ldaumail3/Documents/applications/dcm2bids_macOS_3.2.0/dcm2bids \
            --dicom_dir "${dicom_source}/${session}" \
            --participant "${participant}" \
            --session "${session}" \
            --config "${config_file}" \
            --output_dir "${paths_out}" \
            --force_dcm2bids \
            --clobber

    done

done < "$1"

echo "All participants processed!"
