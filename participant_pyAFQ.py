#This script calculates FA and runs tractography on diffusion maps previously processed with pyAFQ
#Last edited 11/21/2024 - Loic Daumail 
#Code needs an update for participant level analysis
import os
import os.path as op
#import matplotlib
#matplotlib.use('Qt5Agg')  # Or 'Qt5Agg' if you have PyQt5 installed
import matplotlib.pyplot as plt
#plt.ion()
import nibabel as nib
import plotly
import pandas as pd

from AFQ.api.participant import ParticipantAFQ
import AFQ.data.fetch as afd
import AFQ.viz.altair as ava

# afd.organize_stanford_data(clear_previous_afq="track")

tracking_params = dict(n_seeds=25000,
                       random_seeds=True,
                       rng_seed=2022,
                       trx=True,
                       num_chunks=False)

directory = "/Users/ldaumail3/Documents/AMPB_MT_tractometry_analysis/AMPB"#anat"

myafq = ParticipantAFQ(
    bids_path=directory, 
    preproc_pipeline='qsiprep',
    tracking_params=tracking_params,
    viz_backend_spec='plotly_no_gif')

preprocDir = op.join(directory, 'derivatives/qsiprep')

#First get subjects names that we are going to loop through in the directory
# subjNames = [f for f in os.listdir(preprocDir) if f not in {".DS_Store", "dataset_description.json", "dwiqc.json","logs"} and not f.endswith((".html"))]
# # Loop through each subject folder in the directory
# for folder in subjNames:
#     #Store the session numbers of a given subject
#     session =  [s for s in os.listdir(op.join(directory, "derivatives/afq", folder)) if s not in {".DS_Store"}]
#     #Loop through the sessions
#     for ses in session:
# afq_dir = op.join(directory, "derivatives/afq", folder, ses)
#If the FA was not calculated yet, need to run this step
# if not any("FA_dwi.nii.gz" in file for file in os.listdir(afq_dir)):
print("No file contains 'FA_dwi.nii.gz' in its name. Calculating FA and other parameters...")        
FA_fname = myafq.export("dti_fa")#[folder]['04']

#Extract profiles
myafq.export_all()
print("Finished calculating FA...") 
#Look at the FA output. 1: load it into python workspace
FA_img = nib.load(FA_fname)
FA = FA_img.get_fdata()
# 2: plot iFA output on medial slice
print("Plotting FA...")        

fig, ax = plt.subplots(1)
ax.matshow(FA[:, :, FA.shape[-1] // 2], cmap='viridis')
ax.axis("off")
# Save the figure
subject_dir = op.join(directory, "derivatives/afq", folder)
output_filename = folder + "_medial_fig"
output_path = op.join(subject_dir, output_filename)
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)  # Close the figure to free up memory
# fig.show()

#Run probabilistic tractography
print("Run probabilistic tractography")        
myafq.export('profiles')

bundle_html = myafq.export("all_bundles_figure")
plotly.io.show(bundle_html[folder]['04'][0])

fig_files = myafq.export("tract_profile_plots")[folder]['04']

# profiles_df = myafq.combine_profiles()
# altair_df = ava.combined_profiles_df_to_altair_df(profiles_df)
# altair_chart = ava.altair_df_to_chart(altair_df)
# altair_chart.display()
