import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import random
import nibabel as nib
import os
import os.path as op
from nibabel.freesurfer import read_label
from nilearn import plotting
import matplotlib.pyplot as plt

# ----------------------------
# Inputs preparation
# ----------------------------

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
density_dir = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps')
func_dir = op.join(bids_path, 'analysis', 'fMRI_data')
fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
# -----------------------
#1 Generate endpoint densities array
#-------------------------

# ✅ Fixed tract order (keep consistent across subjects!)
tract_order = ['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF'] #'MTxLGNxPU', 'MTxPTxSTS1', 
participants = sorted([p for p in os.listdir(density_dir) if p.startswith("sub-")])
hemis = ["L", "R"]

# Initialize storage dictionary
density_data = {hemi: [] for hemi in hemis}

for participant in participants:
    if not participant.startswith("sub-"):
        continue
    print(f"\n🔹 Participant: {participant}")
    # -----------------
    # Loop by hemisphere
    # -----------------
    for hemi in hemis:
        # for tract in ['MTmaskxLGN', 'MTmaskxPT', 'MTmaskxSTS1', 'MTmaskxPU', 'MTmaskxFEF', 'MTmaskxhIP', 'MTmaskxV1']:
        print(f"   🧩 Hemisphere: {hemi}")
        hemi_fs = "lh" if hemi == "L" else "rh"
        subj_dir = op.join(density_dir, participant, 'wang_MT')
        subj_densities = []

        # Loop through *tracts in fixed order*
        for tract in tract_order:
            
            # Find file matching this tract and hemisphere
            matches = [f for f in os.listdir(subj_dir) if f"wang{tract}" in f and f"hemi-{hemi_fs}" in f and "fsaverage" in f and f.endswith("fsprojdensity0mm2.mgh")]

            if not matches:
                print(f"   ⚠️ Missing: {tract} ({hemi}) for {participant}")
                subj_densities.append(np.zeros_like(subj_densities[0]) if subj_densities else None)
                continue

            # Load the file
            file_path = op.join(subj_dir, matches[0])
            img = nib.load(file_path)
            data = img.get_fdata().astype(np.float32)
            subj_densities.append(data)

        # Stack into one array: shape (n_tracts, n_vertices)
        subj_densities = np.stack(subj_densities, axis=0)  # (7, n_vertices)
        density_data[hemi].append(subj_densities)

# for i, arr in enumerate(density_data[hemi]):
#     print(f"{hemi} element {i}: shape = {arr.shape}")
    
# Convert to numpy arrays

for hemi in hemis:
    density_data[hemi] = np.squeeze(np.stack(density_data[hemi], axis=0))  # (n_subjects, n_tracts, n_vertices)
    print(f"✅ {hemi}-hemisphere shape: {density_data[hemi].shape}")

#-------------------------
# Generate Beta contrasts array
#-------------------------

contrast_order = ["motionXstationary"]
contrast_data = {hemi: [] for hemi in hemis}

for participant in participants:
    if not participant.startswith("sub-"):
        continue

    print(f"\n🔹 Participant: {participant}")
    contrasts_dir = op.join(func_dir, participant, 'glm', 'contrasts')

    for hemi in hemis:
        print(f"   🧩 Hemisphere: {hemi}")

        # Dictionary for this subject: {contrast → list of runs}
        subj_contrast_runs = {contrast: [] for contrast in contrast_order}

        for run in range(0, 6):
            print(f"  🧩 Run: {run+1}")

            for contrast in contrast_order:
                print(f"      🧩 Contrast: {contrast}")

                # Build required filename parts
                if "EB" in participant:
                    required = ["ptlocal", f"hemi-{hemi}", f"run-{run+1}", "fsaverage", contrast, "tstat"]
                else:
                    required = ["mtlocal", f"hemi-{hemi}", f"run-{run+1}", "fsaverage", contrast, "tstat"]

                # Match file
                matches = [f for f in os.listdir(contrasts_dir) if all(k in f for k in required)]

                if not matches:
                    print("        ⚠️ No matching contrast file")
                    continue

                # Load run file
                contrast_file = op.join(contrasts_dir, matches[0])
                img = nib.load(contrast_file)
                data = img.get_fdata().astype(np.float32)

                # Add run for this contrast
                subj_contrast_runs[contrast].append(data)

        # After all runs → convert each contrast’s list into an array
        subj_final = {}
        for contrast in contrast_order:
            runs = subj_contrast_runs[contrast]
            if len(runs) == 0:
                continue
            subj_final[contrast] = np.stack(runs, axis=0)  
            # shape = (n_runs, n_vertices)

        # Save this subject's data for this hemisphere
        contrast_data[hemi].append(subj_final)
#contrast_data[hemi][0][contrast][0].shape #dim 1 = hemisphere initial, dim 2 = participant number, dim 3 = contrast type, dim 4: run number of each contrast map

#================== Prepare and Normalize X and Y data for model fit =============
hemis = ["L", "R"]
contrast = contrast_order[0]   # e.g. "motionXstationary"
n_subj = len(participants)
n_tracts = len(tract_order)
norm_density_data = {hemi: {} for hemi in hemis}
norm_contrast_data = {hemi: {} for hemi in hemis}
for h, hemi in enumerate(hemis):
    densities = density_data[hemi]        # X; (subj, tract, vertices)
    subj_contrasts = contrast_data[hemi]  # Y; list: one dict per participant
    # ----------------------------
    # Load MT ROI
    # ----------------------------
    wang_hmt_path = op.join(
        '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
        f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
    )
    surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
    wang_hmt_vertices = np.where(surf_roi > 0)[0]
    print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")
    # Densities within Wang MT only (subj, tract, masked_vertices)
    if n_tracts == 1:
        densities_masked = densities[:, wang_hmt_vertices]
    else:
        densities_masked = densities[:, :, wang_hmt_vertices]

    n_masked = len(wang_hmt_vertices)

    for s, participant in enumerate(participants):
        #Prepare subject's Y = functional maps
        # get this subject's run maps of the chosen contrast
        subj_dict = subj_contrasts[s]
        if contrast not in subj_dict:
            print(f"Subject {s} missing contrast {contrast} for hemi {hemi}, skipping")
            continue
        
        C_full = subj_dict[contrast]              # (n_runs, n_vertices_fullspace)
 
        # mask ROI
        C = np.squeeze(C_full[:, wang_hmt_vertices] )         # (n_runs, n_masked)
        #zscore the runs for a given participant
        n_runs = C.shape[0]
        print(f"\nSubject {s+1}: {n_runs} runs (hemi {hemi})")
        zscored_C = np.array([(C[r_num,:] - np.mean(C[r_num,:]))/np.std(C[r_num,:]) for r_num in range(n_runs)])

        norm_contrast_data[hemi].setdefault(s, {})
        norm_contrast_data[hemi][s][contrast] = zscored_C
        
        #Prepare subject's X = zscored density maps for each tract
        zscored_densities = np.full((n_tracts, n_masked), np.nan)
        for tract_idx in range(n_tracts):

            print(f" Tract {tract_idx+1}/{n_tracts} z-scored")

            # anatomical vector for this subject and tract (length = n_masked)
            # anat_vec = densities_masked[s, tract_idx, :]  # shape (n_masked,)
            #Zscore the density data of the tract of this participant
            if n_tracts == 1:
                if np.std(densities_masked[s, :]) == 0:
                    zscored_densities[tract_idx,:] = 0
                else:
                    zscored_densities[tract_idx,:] = (densities_masked[s, :] - np.mean(densities_masked[s, :]))/np.std(densities_masked[s, :])
            else:
                if np.std(densities_masked[s, tract_idx, :]) == 0:
                    zscored_densities[tract_idx,:] = 0
                else:
                    zscored_densities[tract_idx,:] = (densities_masked[s, tract_idx, :] - np.mean(densities_masked[s, tract_idx, :]))/np.std(densities_masked[s, tract_idx, :])


        norm_density_data[hemi].setdefault(s, {})
        norm_density_data[hemi][s] = zscored_densities

def vertex_bootstrap_reliability(C, n_boot=1000, frac=1):
    """
    C: array (n_runs, n_vertices)
    n_boot: number of bootstrap samples
    frac: fraction of vertices sampled per bootstrap
    Returns mean bootstrap reliability
    """
    n_runs, n_vertices = C.shape

    n_sample = int(frac * n_vertices)
    rs = []

    for _ in range(n_boot):
        verts = np.random.choice(n_vertices, n_sample, replace=True)

        # split runs as before (fixed run split)
        half = n_runs // 2
        A = C[:half, verts].mean(axis=0)
        B = C[half:, verts].mean(axis=0)

        r, _ = pearsonr(A, B)
        if not np.isnan(r):
            rs.append(r)

    return np.mean(rs)

## Fit Regression Models to each group
verbose = True
alpha = 1.0  # choose fixed regularization strength

hemis = ["L", "R"]
groups = ["EB", "NS"]
contrast = contrast_order[0]   # e.g. "motionXstationary"

# get n_subj, n_tracts
n_subj = len(participants)
n_tracts = len(tract_order)
n_groups = len(groups)
rs   = np.full((n_subj, len(hemis)), np.nan)
rsquared = np.full(( n_subj, len(hemis)), np.nan) #goodness of fit
reliability = np.full((n_subj, len(hemis)), np.nan)
r_all = np.full((n_subj, len(hemis)), np.nan)
rnd_run_idx = np.full((n_subj, 3, len(hemis)), np.nan)
trained_coefs = np.zeros((n_tracts, n_groups, len(hemis)))  # scalar summary per tract/run

n_boot = 1000
trained_coefs_null = np.zeros((n_tracts, n_boot, n_groups, len(hemis)))

predicted_maps = {hemi: [] for hemi in hemis}
for h, hemi in enumerate(hemis):
        #h = 0
        #hemi = "L"
        #s = 0
        #'sub-EBxGxCCx1986'

    # ----------------------------
    # Load MT ROI
    # ----------------------------
    wang_hmt_path = op.join(
        '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
        f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
    )
    surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
    wang_hmt_vertices = np.where(surf_roi > 0)[0]
    print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")
    n_masked = len(wang_hmt_vertices)

    # Predicted and coef storage preallocation
    predicted = np.full((n_subj, n_masked), np.nan)  # predicted maps per run
    _, _, n_vertices  = density_data[hemi].shape #get total number of vertices within fsaverage hemisphere
    
    n_target_runs = 3
    all_C = []              # will store (n_subj, 3, n_vertices)
    rnd_run_idx = np.full((n_subj, n_target_runs, len(hemis)), np.nan)
    for s, participant in enumerate(participants):

        C_full = norm_contrast_data[hemi][s][contrast]   # (n_runs_available, n_vertices)
        n_runs_available = C_full.shape[0]

        # ---- choose runs ----
        if "NS" in participant:
            run_idx = np.arange(3)
        elif "EB" in participant:
            run_idx = np.random.choice(
                n_runs_available, size=3, replace=False
            )

        rnd_run_idx[s, :, h] = run_idx

        # ---- extract runs ----
        C_sel = C_full[run_idx, :]     # (3, n_vertices)
        all_C.append(C_sel)
    all_C = np.stack(all_C, axis=0)
    C_mean = all_C.mean(axis=1)


    # -------------------------
    # main loop
    # -------------------------

    for g, group in enumerate(groups):
        
        participants_group = [p for p in participants if group in p]
        n_g_subj = len(participants_group)
        #Define X and Y of the given group
        group_norm_density = np.stack([norm_density_data[hemi][p] for p in range(n_subj) if participants[p] in participants_group], axis = 0)
        group_C_mean = np.stack([C_mean[p,:] for p in range(n_subj) if participants[p] in participants_group], axis = 0)

        if verbose:
            print(f"Fitting group {group}")

        # X: density maps for this participant
        # shape: (n_vertices_masked, n_tracts)
        X = group_norm_density.transpose(0,2,1).reshape(-1,3)#norm_density_data[hemi][s].T

        # y: functional contrast map
        # shape: (n_vertices_masked,)
        y = group_C_mean.reshape(-1,1) #C_mean[s, :]

        # Fit regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)

        # Store coefficients
        trained_coefs[:, g, h] = ridge.coef_.copy()

        # # Predict (optional)
        # y_pred = ridge.predict(X)
        # predicted[s, :] = y_pred

        # Reliability (if needed)
        # reliability[s, h] = vertex_bootstrap_reliability(all_C[s,:,:])


    #Null beta distributions
    for n in range(n_boot):
        participants1 = np.random.choice(n_subj, 7, replace=False)
        participants2 = [p for p in range(n_subj) if p not in participants1]

        group_norm_density1 = np.stack([norm_density_data[hemi][p] for p in participants1], axis = 0)
        group_C_mean1 = np.stack([C_mean[p,:] for p in participants1], axis = 0)
        X1 = group_norm_density1.transpose(0,2,1).reshape(-1,3)
        y1 = group_C_mean1.reshape(-1,1)

        ridge1 = Ridge(alpha=alpha)
        ridge1.fit(X1, y1)

        group_norm_density2 = np.stack([norm_density_data[hemi][p] for p in participants2], axis = 0)
        group_C_mean2 = np.stack([C_mean[p,:] for p in participants2], axis = 0)
        X2 = group_norm_density2.transpose(0,2,1).reshape(-1,3)
        y2 = group_C_mean2.reshape(-1,1)

        ridge2 = Ridge(alpha=alpha)
        ridge2.fit(X2, y2)

        trained_coefs_null[:,n, 0, h] = ridge1.coef_.copy()
        trained_coefs_null[:,n, 1, h] = ridge2.coef_.copy()


null_dist_diff = trained_coefs_null[:,:,0,:] - trained_coefs_null[:,:,1,:]
sample_diff = trained_coefs[:,0,:] - trained_coefs[:,1,:]

#Plot

import numpy as np
import matplotlib.pyplot as plt

n_tracts = null_dist_diff.shape[0]
n_hemi = null_dist_diff.shape[2]

fig, axes = plt.subplots(n_tracts, 2, figsize=(10, 4 * n_tracts))

# Ensure axes is always 2D
if n_tracts == 1:
    axes = np.array([axes])

for t in range(n_tracts):
    for h in range(n_hemi):

        ax = axes[t, h]

        # Null distribution for this tract & hemisphere
        null_vals = null_dist_diff[t, :, h]

        # Density histogram
        ax.hist(null_vals, bins=40, density=True)

        # 95% interval
        lower = np.percentile(null_vals, 2.5)
        upper = np.percentile(null_vals, 97.5)

        ax.axvline(lower,color="black")
        ax.axvline(upper,color="black")

        # Observed value
        observed = sample_diff[t, h]
        ax.axvline(observed, color="red")

        hemi_label = "Left" if h == 0 else "Right"

        ax.set_title(f"Tract {tract_order[t]} - {hemi_label}")
        ax.set_xlabel("Difference")
        ax.set_ylabel("Density")
saveDir = op.join(bids_path, 'analysis', 'plots')
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, "permutation_group_diff_betas_combined_tracts.png"),
            dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()