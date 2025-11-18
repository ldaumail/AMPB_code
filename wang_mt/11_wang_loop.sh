#!/bin/bash

# List of tract names you want to loop over
tracts=(
  MTxLGN
  MTxPT
)

# Paths (edit if needed)
participants_file="../utils/study2_subjects_updated.txt"
bids_path="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb"
pyAFQ_path="/Volumes/cos-lab-wpark78/LoicDaumail/ampb/derivatives/pyafq/wmgmi_wang" #"/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/pyAFQ/wmgmi_wang"
projdist=3

# Loop through tract list
for tract in "${tracts[@]}"; do
    echo "Running tract: $tract"

    python ./11_wang_endpoint_density_mri_vol2surf.py \
        --participants_file "$participants_file" \
        --tract_name "$tract" \
        --bids_path "$bids_path" \
        --pyAFQ_path "$pyAFQ_path" \
        --projdist "$projdist"

done

echo "All tracts processed."

