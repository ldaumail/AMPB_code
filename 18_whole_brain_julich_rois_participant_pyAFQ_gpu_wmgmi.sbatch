#!/bin/bash
# Perform tractography on each subject with pyAFQ
# First need to build pyafq image for apptainer: apptainer build pyafq_latest.sif docker://ghcr.io/nrdg/pyafq:latest
#
#  Created by ldaumail3 on 12/17/24.
#
#SBATCH --account=gts-wpark78                 # Charge account
#SBATCH -N1 			               # Number of nodes and cores per node
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=128G                      #--mem-per-gpu=10G you can also adjust the cpu memory, however both parameters are mutually exclusive: --mem=64G
#SBATCH -t150                                 # Duration of the job (8*60 = 480 mins; 900 = 15 hours)
#SBATCH -qinferno                              # QOS name embers = free, leftover  cpus / inferno = paid, credits usage
#SBATCH -o/storage/home/hcoda1/0/ldaumail3/scratch/afq_ampb/derivatives/pyAFQ/outReports/Report-%j.out                       # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail preferences
#SBATCH --mail-user=ldaumail3@gatech.edu      # E-mail address for notifications
#SBATCH --array=1 #-13%13                             # Job IDs to run simultaneously from SUBJ list (limited to 4 simultaneously with %4)
#SBATCH --requeue				#Resume job if previously crashed

# for calculating the amount of time the job takes
begin=`date +%s`
echo $HOSTNAME

# Modules -------------------------------------------------------------
#module load apptainer
module purge
module load cuda/12.1
module load gcc/12.3.0
# Directories ------------------------------------------------------------------

APPTAINER_DIR="${HOME}/r-wpark78-0/images"
DATA_HOME="${HOME}/scratch/afq_ampb"
DERIVATIVES_DIR="${DATA_HOME}/derivatives"
DATA_DIR="${DERIVATIVES_DIR}/qsiprep/${participant}"
ANALYSIS_DIR="${DATA_HOME}/analysis"
OUT_DIR="${DERIVATIVES_DIR}/${output_dir}"
if [[ ! -d "${OUT_DIR}" ]]; then mkdir -p "${OUT_DIR}"; fi


# Run pyAFQ --------------------------------------------------------------------

# define variables
hemistr="${hemisphere:0:1}"

# prepare output directory
curr_afq="afq-wb_10rounds"
curr_out="${OUT_DIR}/${curr_afq}/${participant}"
if [[ ! -d "${curr_out}" ]]; then mkdir -p "${curr_out}"; fi

fname="${participant}_ses-concat_desc-bundles_tractography.trx"
if [[ ! -f "${curr_out}/${fname}" ]]; then 
  echo "Running AFQ for ${curr_afq}..." 

  apptainer run \
    --bind "${DATA_DIR}:/data" \
    --bind "${ANALYSIS_DIR}:/analysis" \
    --bind "${curr_out}/..:/out" \
    --nv \
    "${APPTAINER_DIR}/pyafq_gpu_cuda_12_gmwmi2.sif" \
    "${DATA_HOME}/code/18_whole_brain_julich_rois_participant_pyAFQ_gpu_wmgmi.py" \
    --dwi_data_file "/data/ses-concat/dwi/${participant}_ses-concat_acq-HCPdir99_space-ACPC_desc-preproc_dwi.nii.gz" \
    --bval_file "/data/ses-concat/dwi/${participant}_ses-concat_acq-HCPdir99_space-ACPC_desc-preproc_dwi.bval" \
    --bvec_file "/data/ses-concat/dwi/${participant}_ses-concat_acq-HCPdir99_space-ACPC_desc-preproc_dwi.bvec" \
    --t1_file "/data/anat/${participant}_space-ACPC_desc-preproc_T1w.nii.gz" \
    --template_dir "/analysis/julich_space-ACPC_rois/" \
    --participant "${participant}" \
    --output_dir "/out/${participant}"
fi


# getting end time to calculate time elapsed
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed
