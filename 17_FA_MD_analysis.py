#Loic Daumail 04/01/2026
#FA analysis across tracts

import os.path as op
import os
import nibabel as nib
import numpy as np
import pandas as pd
import sys

current_dir = op.dirname(op.abspath(__file__))
sys.path.append(current_dir)
import utils.cluster_stats as cls

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
pyAFQ_dir = op.join(bids_path, "derivatives", "pyAFQ", "wmgmi_wang")
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
# -----------------------
#1 Generate dki FA / dki MD arrays
#-------------------------

# ✅ Fixed tract order (keep consistent across subjects!)
tract_order = ['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF'] #'MTxLGNxPU', 'MTxPTxSTS1', 
participants_list = op.join(bids_path, 'code', 'utils', 'study2_subjects_updated.txt')
with open(participants_list, 'r') as f: #read file and create file object that you can read line by line iteratively
    participants = sorted([line.strip() for line in f if line.strip()])  #line.strip() for line in f = reads line by line and removes white spaces "if line.strip()"" removes empty lines

hemis = ["L", "R"]

# Initialize storage dictionary
dkifa_data = {hemi: [] for hemi in hemis}
dkimd_data = {hemi: [] for hemi in hemis}

for participant in participants:
    if not participant.startswith("sub-"):
        continue
    print(f"\n🔹 Participant: {participant}")
    # -----------------
    # Loop by hemisphere
    # -----------------
    for hemi in hemis:
        hemi_pyAFQ = "Left" if hemi == "L" else "Right"
        print(f"   🧩 Hemisphere: {hemi}")

        subj_fa = []
        subj_md = []

        # Loop through *tracts in fixed order*
        for tract in tract_order:
            fa_file_path = op.join(pyAFQ_dir,f"afq-{hemi_pyAFQ}{tract}", participant, f"{participant}_ses-concat_acq-HCPdir99_desc-profiles_tractography.csv")

            df = pd.read_csv(fa_file_path)
            fa_data = df["dki_fa"].to_numpy()
            subj_fa.append(fa_data)

            md_data = df["dki_md"].to_numpy()
            subj_md.append(md_data)

        # Stack into one array: shape (n_tracts, n_nodes)
        subj_fa = np.stack(subj_fa, axis=0)  # (3, n_nodes)
        dkifa_data[hemi].append(subj_fa)
        subj_md = np.stack(subj_md, axis=0)  # (3, n_nodes)
        dkimd_data[hemi].append(subj_md)


# Convert to numpy arrays
for hemi in hemis:
    dkifa_data[hemi] = np.squeeze(np.stack(dkifa_data[hemi], axis=0))  # (n_subjects, n_tracts, n_vertices)
    dkimd_data[hemi] = np.squeeze(np.stack(dkimd_data[hemi], axis=0))  # (n_subjects, n_tracts, n_vertices)
    print(f"✅ {hemi}-hemisphere shape: {dkifa_data[hemi].shape}")


#===================================================================
#==================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path as op

def plotnodes(dkifa_data, groups, hemis, save_path, tract_names, variable):
    """
    dkifa_data: dict like {"L": array, "R": array}
        each array shape = (participants, tracts, nodes)
    groups: list/array of length n_participants ("EB" or "NS")
    hemis: ["L", "R"]
    tract_names: optional list of tract names
    """

    # assume both hemis have same shape
    n_participants, n_tracts, n_nodes = dkifa_data[hemis[0]].shape

    fig, axes = plt.subplots(
        n_tracts, 2,
        figsize=(12, 4 * n_tracts),
        sharex=True,
        sharey=True
    )

    # ensure axes is 2D even if n_tracts = 1
    if n_tracts == 1:
        axes = np.array([axes])

    for t in range(n_tracts):
        for h, hemi in enumerate(hemis):

            data_array = dkifa_data[hemi][:, t, :]  # (participants, nodes)

            # -------------------------
            # Build dataframe
            # -------------------------
            df = pd.DataFrame({
                "participant": np.repeat(np.arange(n_participants), n_nodes),
                "node": np.tile(np.arange(n_nodes), n_participants),
                f"{variable}": data_array.reshape(-1),
            })

            df["group"] = np.repeat(groups, n_nodes)

            # -------------------------
            # Plot (mean + error)
            # -------------------------
            ax = axes[t, h]

            sns.lineplot(
                data=df,
                x="node",
                y=f"{variable}",
                hue="group",
                estimator="mean",
                errorbar="se",   # 👈 adds error bars
                marker=None,
                ax=ax
            )

            # titles
            if t == 0:
                ax.set_title(f"{hemi} hemisphere",  fontsize=22, fontweight='bold')

            if h == 0:
                tract_label = tract_names[t] if tract_names else f"Tract {t+1}"
                ax.set_ylabel(f"{tract_label}\ndki {variable}", fontsize=18, fontweight='bold')
            else:
                ax.set_ylabel("")

            if t == n_tracts - 1:
                ax.set_xlabel("Node", fontsize=18, fontweight='bold')
            else:
                ax.set_xlabel("")

            # cleaner legend (only once)
            # if not (t == 0 and h == 1):
            ax.get_legend().remove()
            sns.despine() #remove box edges around plot
    # keep one legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        op.join(save_path, f"dki_{variable}_nodes_by_tract.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.savefig(
        op.join(save_path, f"dki_{variable}_nodes_by_tract.svg"),
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

# using fnc for class
groups = ["EB" if "EB" in p else "NS" for p in participants]
save_path = op.join(bids_path, "analysis", "plots")
os.makedirs(save_path, exist_ok=True)
tract_order = ['Thalamo-cortical', 'Temporal', 'Frontal']
plotnodes(dkifa_data, groups, hemis, save_path, tract_order, "FA")


plotnodes(dkimd_data, groups, hemis, save_path, tract_order, "MD")

#============================
# Stats
#============================
cluster_results = {hemi: [] for hemi in hemis}
for hemi in hemis:
    cluster_permutation = {tract: [] for tract in tract_order}
    for t, tract in enumerate(tract_order):
        yvar = np.squeeze(dkimd_data[hemi][:,t,:])
        cluster_permutation[tract] = cls.run_cluster_test(yvar, alpha=0.05, n_iter=1000)
    cluster_results[hemi] = cluster_permutation
    