#!/bin/bash
#  apptainer_qsiprep.sh
#  
#
#  Created by ldaumail3 on 12/17/24.
#  
#SBATCH --account=gts-wpark78                 # Charge account
#SBATCH -N1 --ntasks-per-node=8               # Number of nodes and cores per node
#SBATCH --mem-per-cpu=10G                      # Memory per core
#SBATCH -t480                                 # Duration of the job (8*60 = 480 mins)
#SBATCH -qembers                              # QOS name #embers = free leftover cpus, risk for job to be killed by other jobs; inferno = paid for cpus, faster, no risk.
#SBATCH -oReport-%j.out                       # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail preferences
#SBATCH --mail-user=ldaumail3@gatech.edu      # E-mail address for notifications
#SBATCH --array=1-16%1                        # Job IDs to run simultaneously from SUBJ list (limited to 4 simultaneously with %4)

SUBJ=(NSxGxBAx1970 EBxGxCCx1986 EBxLxHHx1949 EBxGxZAx1990 NSxLxATx1954 EBxGxEYx1965 NSxGxHNx1952 NSxLxYKx1964 EBxLxQPx1957 EBxGxYZx1949 EBxLxTZx1956 NSxGxIFx1991 EBxGxPEx1959 NSxGxHKx1965 NSxLxQUx1953 NSxLxPQx1973)

# for calculating the amount of time the job takes
begin=`date +%s`
echo $HOSTNAME

# Loading modules
#module load apptainer

apptainer run \
    --bind $HOME/scratch/ampb:/bids \
    --bind $HOME/scratch/ampb/derivatives/qsiprep:/out \
    --bind $HOME/scratch/ampb/work:/work \
    --bind $HOME/p-wpark78-0/software/license.txt:/license.txt \
    --bind $HOME/p-wpark78-0/images:/opt/images \
$HOME/p-wpark78-0/images/qsiprep-latest.sif \
    --skip-bids-validation \
     /bids /out participant \
    --participant-label ${SUBJ[$SLURM_ARRAY_TASK_ID-1]} \
    -w /work \
    --ignore fieldmaps \
    --output-resolution 1.5 \
    --mem-mb 75000 \
    --nprocs 8 \
    --low-mem --stop-on-first-crash \
    --fs-license-file /license.txt

# getting end time to calculate time elapsed
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed
