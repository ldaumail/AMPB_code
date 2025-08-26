#!/usr/bin/env bash

# Variables --------------------------------------------------------------------

paths_main="${HOME}/scratch/afq_ampb/code"
paths_logs="${HOME}/scratch/logs"

output_dir="pyAFQ/cleaning_rounds2" 

hemisphere_list=( "Left" "Right" )
roi_list=("MTxLGN" "MTxPT")
participant_list=("sub-EBxGxEYx1965" "sub-EBxGxPEx1959" "sub-EBxGxYZx1949"
"sub-EBxGxZAx1990" "sub-EBxLxHHx1949" "sub-EBxLxQPx1957" "sub-EBxLxTZx1956"
"sub-NSxGxBAx1970" "sub-NSxGxHKx1965" "sub-NSxGxHNx1952" "sub-NSxGxIFx1991"
"sub-NSxLxATx1954" "sub-NSxLxPQx1973" "sub-NSxLxQUx1953" "sub-NSxLxYKx1964")
#   "sub-EBxGxCCx1986"
# -----------------------------------------------------------------------------

for participant in "${participant_list[@]}"; do # for each participant
  for hemisphere in "${hemisphere_list[@]}"; do # for each hemisphere
    for roi in "${roi_list[@]}"; do # for each ROI
      job_name="${participant}_${hemisphere}${roi}"

      log_dir="${paths_logs}/${hemisphere}${roi}"
      if [[ ! -d "${log_dir}" ]]; then mkdir -p "${log_dir}"; fi

      sbatch \
        --job-name "${job_name}" \
        --output "${log_dir}/${job_name}.output" \
        --error "${log_dir}/${job_name}.error" \
        --export=participant="${participant}",output_dir="${output_dir}",hemisphere="${hemisphere}",roi="${roi}" \
        "${paths_main}/9_participant_pyAFQ_gpu.sbatch"
    done
  done
done
