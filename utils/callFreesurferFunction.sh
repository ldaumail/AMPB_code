#!/bin/bash
while getopts s: option
do
case "${option}"
in
s) STRING=${OPTARG};;
esac
done

export FREESURFER_HOME=/Applications/freesurfer/8.0.0-beta
export SUBJECTS_DIR=$FREESURFER_HOME/subjects
source $FREESURFER_HOME/SetUpFreeSurfer.sh

$STRING
