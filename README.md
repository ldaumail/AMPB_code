This Repository includes code for the data preprocessing and analysis of the AMPB dataset.

1: 1_dcm2bids.sh aims to convert source dicom data into BIDS structure. In order to use this script properly, you will need to first build a new project directory with a bids scaffold using the command dcm2bids_scaffold, then place the source data (dicoms) in the /sourcedata folder of the scaffold. You then want to use dcm2bids_helper to look up .json sidecar files to see if some criteria can distinguish runs from each other, to use them to build your configuration file for running dcm2bids.
Tip: in order to run this script, you need to make sure that you activated the virtual environment on the terminal, such as a conda environment in which you installed dcm2bids and dcm2niix following this procedure: (https://unfmontreal.github.io/Dcm2Bids/3.2.0/get-started/install/)
This script is ran with the list of participants you want to convert the data into bids as an input "bash 1_dcm2bids.sh participants.txt"


2: 2_apptainer_qsiprep_requeue.sbatch: this script launches batch jobs in parallel on the PACE clusters. The BIDS data from each individual is preprocessed using qsiprep. when logged in on PACE, just use the command line : "sbatch 2_apptainer_qsiprep_requeue.sbatch"

3: 3_make_masks_julich.py: here, we imported probabilistic ROIs from the Julich brain atlas ( https://search.kg.ebrains.eu/instances/f1fe19e8-99bd-44bc-9616-a52850680777 ) and convert them into binary masks. The script also registers and resamples them into individual diffusion space using antspyx.

4: 4_* scripts: convert the label functional MT ROI files obtained from our own experiments into volumetric binary masks resampled into individual diffusion space.
