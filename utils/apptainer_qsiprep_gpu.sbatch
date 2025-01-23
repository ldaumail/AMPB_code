#!/bin/bash
#  apptainer_qsiprep.sh
#  
#
#  Created by ldaumail3 on 12/17/24.
#  
#SBATCH --account=gts-wpark78                 # Charge account
#SBATCH -N1                                   # Number of nodes, no need to  specify the number of cores per node --ntasks-per-node, 12 cores are allocated to V100 by default
#SBATCH --gres=gpu:V100:1                          #use cpu-small partition
#SBATCH --mem-per-gpu=12G                      # Memory per core
#SBATCH -t900                                 # Duration of the job (8*60 = 480 mins; 900 = 15 hours)
#SBATCH -qinferno                              # QOS name embers = free, leftover  cpus / inferno = paid, credits usage
#SBATCH -o./outReports/Report-%j.out                       # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail preferences
#SBATCH --mail-user=ldaumail3@gatech.edu      # E-mail address for notifications
#SBATCH --array=1-4%1                             # Job IDs to run simultaneously from SUBJ list (limited to 4 simultaneously with %4)
#SBATCH --requeue				#Resume job if previously crashed

#SUBJ=(NSxGxBAx1970 EBxGxCCx1986 EBxLxHHx1949 EBxGxZAx1990 NSxLxATx1954 EBxGxEYx1965 NSxGxHNx1952 NSxLxYKx1964 EBxLxQPx1957 EBxGxYZx1949 EBxLxTZx1956 NSxGxIFx1991 EBxGxPEx1959 NSxGxHKx1965 NSxLxQUx1953 NSxLxPQx1973)
#damaged NSxLxATx1954 EBxGxYZx1949 EBxLxTZx1956 #still need to run them maybe with new data
SUBJ=(EBxGxYZx1949 NSxGxHNx1952 EBxLxTZx1956 NSxLxATx1954) # #NSxGxHNx1952
#SUBJ=(NSxGxBAx1970 EBxGxCCx1986 EBxLxHHx1949 EBxGxZAx1990 NSxLxYKx1964 NSxLxPQx1973 EBxGxEYx1965 EBxLxQPx1957 NSxGxIFx1991 EBxGxPEx1959 NSxGxHKx1965 NSxLxQUx1953) 
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

#    -B /mnt:mnt \
# getting end time to calculate time elapsed
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed
