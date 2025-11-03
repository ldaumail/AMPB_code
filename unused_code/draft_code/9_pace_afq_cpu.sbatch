#!/bin/bash
# Perform tractography on each subject with pyAFQ
# First need to build pyafq image for apptainer: apptainer build pyafq_latest.sif docker://ghcr.io/nrdg/pyafq:latest 
#
#  Created by ldaumail3 on 12/17/24.
#  
#SBATCH --account=gts-wpark78                 # Charge account
#SBATCH -N1 --ntasks-per-node=10               # Number of nodes and cores per node
#SBATCH --mem=128G				##SBATCH --mem-per-cpu=15G                      # Memory per core
#SBATCH -t2000                                 # Duration of the job (8*60 = 480 mins; 900 = 15 hours)
#SBATCH -qinferno                              # QOS name embers = free, leftover  cpus / inferno = paid, credits usage
#use cpu-small partition
#SBATCH -o/storage/home/hcoda1/0/ldaumail3/scratch/afq_ampb/derivatives/afq/outReports/Report-%j.out                       # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail preferences
#SBATCH --mail-user=ldaumail3@gatech.edu      # E-mail address for notifications
#SBATCH --array=1-16%16                             # Job IDs to run simultaneously from SUBJ list (limited to 4 simultaneously with %4)
#SBATCH --requeue				#Resume job if previously crashed

SUBJ=(sub-EBxLxTZx1956) #(sub-NSxGxBAx1970 sub-EBxGxCCx1986 sub-EBxLxHHx1949 sub-EBxGxZAx1990 \
#sub-NSxLxATx1954 sub-EBxGxEYx1965 sub-NSxGxHNx1952 sub-NSxLxYKx1964 \
#sub-EBxLxQPx1957 sub-EBxLxTZx1956 sub-NSxGxIFx1991 \
#sub-EBxGxPEx1959 sub-NSxGxHKx1965 sub-NSxLxQUx1953 sub-NSxLxPQx1973)

# for calculating the amount of time the job takes
begin=`date +%s`
echo $HOSTNAME

# Loading modules
#module load apptainer

apptainer exec \
$HOME/p-wpark78-0/images/my_afq_container.sif \
     python /storage/home/hcoda1/0/ldaumail3/scratch/afq_ampb/code/6_whole_brain_pyAFQ.py \
--participants_list ${SUBJ[$SLURM_ARRAY_TASK_ID-1]} \
--paths_local $HOME/scratch/afq_ampb 

#    -B /mnt:mnt \
# getting end time to calculate time elapsed
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed
