#!/bin/bash

# List of tract names you want to loop over
tracts=(
  MTxPTxSTS1 MTxFEF
)

# Paths (edit if needed)
bids_path="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb"
participants_file="$bids_path/code/utils/study2_subjects_updated.txt" #study2_subjects_updated.txt
pyAFQ_path="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/pyAFQ/wmgmi_wang" #"/Volumes/cos-lab-wpark78/LoicDaumail/ampb/derivatives/pyafq/wmgmi_wang" #"/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/pyAFQ/wmgmi_wang"
projdist=0

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