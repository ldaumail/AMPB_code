#Loic Daumail 11/22/2024    
#Preprocessing of diffusion MRI data using qsiprep
import os
import os.path as op
# 
# qsiprep (preprocessing)
version = "latest" #version use is 1.0.0rc2.dev0+g789be41.d20241119
bids_dir = op.join(op.expanduser('~'), 'Documents/research/ampb_mt_tractometry_analysis/ampb')
output_dir = op.join(op.expanduser('~'), 'Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/qsiprep')
work_dir = op.join(op.expanduser('~'), 'Documents/research/ampb_mt_tractometry_analysis/ampb/work')


participants = [f for f in os.listdir(bids_dir) if f not in {".DS_Store", "dataset_description.json", "code","derivatives", "work"}]
participants_cleaned = [f.replace("sub-", "") for f in participants]
print(participants_cleaned)


for participant in participants_cleaned:
    cmd = f"docker run --rm " \
        f"--volume {bids_dir}:/bids " \
        f"--volume {output_dir}:/out " \
        f"--volume {work_dir}:/work " \
        f"--volume $FREESURFER_HOME/license.txt:/license.txt " \
        f"pennlinc/qsiprep:{version} " \
        f"--skip-bids-validation " \
        f"/bids /out participant " \
        f"--participant-label {participant} " \
        f"-w /work " \
        f"--ignore fieldmaps " \
        f"--output-resolution 1.5 " \
        f"--mem-mb 16000 " \
        f"--nprocs 2 " \
        f"--low-mem " \
        f"--fs-license-file /license.txt " 

    os.system(cmd)


# docker run -ti --rm \
#     -v /filepath/to/data/dir \
#     -v /filepath/to/output/dir \
#     -v ${FREESURFER_HOME}/license.txt:/opt/freesurfer/license.txt \
#     pennlinc/qsiprep:latest \
#     /filepath/to/data/dir /filepath/to/output/dir participant \
#     --fs-license-file /opt/freesurfer/license.txt

# cmd = f"docker run --rm " \
#       f"--mount type=bind,src={indir},dst=/data " \
#       f"--mount type=bind,src={outdir},dst=/out " \
#       f"--memory=32g " \
#       f"--memory-swap=32g " \
#       f"nipreps/mriqc:{version} " \
#       f"--nprocs {num_procs} " \
#       f"--verbose-reports " \
#       f"/data /out "