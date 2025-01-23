#!/usr/bin/env bash

participant="sub-NSxLxATx1954"

paths_main="/Volumes/Leviathan/AMPB"
paths_data="${paths_main}/${participant}"
paths_out="${paths_main}/data"
config_file="${paths_main}/config.json"


for session in "ses-01" "ses-02" "ses-03" "ses-04"; do

  dcm2bids \
    --dicom_dir "${paths_data}/${session}" \
    --participant "${participant}" \
    --session "${session}" \
    --config "${config_file}" \
    --output_dir "${paths_out}" \
    --force_dcm2bids \
    --clobber

done
