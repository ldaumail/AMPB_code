#!/bin/bash

# List of tract names you want to loop over
#
tracts=(RPTR RSTR LPTR LSTR InferiorLongitudinal InferiorFrontooccipital SuperiorLongitudinal VerticalOccipital Arcuate PosteriorArcuate AnteriorThalamic CingulumCingulate Corticospinal Uncinate)

# Paths (edit if needed)
bids_path="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb"
participants_file="$bids_path/code/utils/study2_subjects_updated.txt" #study2_subjects_updated.txt
pyAFQ_path="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/pyAFQ/wmgmi_wb/afq3beta-wb_6rounds" #"/Volumes/cos-lab-wpark78/LoicDaumail/ampb/derivatives/pyafq/wmgmi_wang" #"/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/pyAFQ/wmgmi_wang"
density_path="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/tdi_maps/dipy_wmgmi_tdi_maps"
density_folder="afq3beta-wb_6rounds"
projfrac=0

# Loop through tract list
for tract in "${tracts[@]}"; do
    echo "Running tract: $tract"

    python ./19_2_project_density_vol2surf.py \
        --participants_file "$participants_file" \
        --tract_name "$tract" \
        --bids_path "$bids_path" \
        --pyAFQ_path "$pyAFQ_path" \
        --density_path "$density_path" \
        --density_folder "$density_folder" \
        --projfrac "$projfrac"

done

echo "All tracts processed."