~#!/bin/bash

# Check if input file is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 participants.txt"
  exit 1
fi

PARTICIPANT_FILE=$1

SOURCE_DIR="/Volumes/cos-lab-wpark78/LoicDaumail/ampb/derivatives/pyafq/wmgmi_wang/afq-LeftMTxLGNxPU/"
DEST_DIR="/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/ROIs/wm_gm_csf_masks_pyAFQ202dev"

LABELS=("csf" "wm" "gm")

while IFS= read -r participant; do
  [ -z "$participant" ] && continue

  SRC_DIR="$SOURCE_DIR/$participant/tractography"
  DST_DIR="$DEST_DIR/$participant"

  if [ ! -d "$SRC_DIR" ]; then
    echo "ROI directory not found for $participant"
    continue
  fi

  mkdir -p "$DST_DIR"

  for label in "${LABELS[@]}"; do
    file="${participant}_ses-concat_acq-HCPdir99_desc-${label}_probseg.nii.gz"

    if [ -f "$SRC_DIR/$file" ]; then
      cp "$SRC_DIR/$file" "$DST_DIR/"
      echo "Copied $file"
    else
      echo "Missing $file"
    fi
  done

done < "$PARTICIPANT_FILE"

echo "All done!"



