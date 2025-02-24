This Repository includes code for the data preprocessing and analysis of the AMPB dataset.

1: 1_bis_dcm2bids.sh aims to convert source dicom data into BIDS structure. In order to use this script properly, you will need to first buil a new project directory with a bids scaffold using the command dcm2bids_scaffold, then place the source data (dicoms) in the /sourcedata folder of the scaffold.


: apptainer_qsiprep.sh : this script launches batch jobs in parallel on the PACE clusters. The BIDS data from each individual is preprocessed using qsiprep. when logged in on PACE, just use the command line : "sbatch apptainer_qsiprep.sh"

