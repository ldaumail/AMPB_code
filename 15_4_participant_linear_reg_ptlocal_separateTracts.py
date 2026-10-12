import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
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
tract_order = ['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF']

participants = sorted([p for p in os.listdir(density_dir) if p.startswith("sub-")])
# participants = ['sub-EBxLxQPx1957','sub-EBxLxTZx1956']
hemis = ["L", "R"]
projfrac = '1'
# Initialize storage dictionary
density_data = {hemi: [] for hemi in hemis}
density_mask = {hemi: [] for hemi in hemis}

for participant in participants:
    if not participant.startswith("sub-"):
        continue
    print(f"\n🔹 Participant: {participant}")
    # -----------------
    # Loop by hemisphere
    # -----------------
    for hemi in hemis:
        print(f"   🧩 Hemisphere: {hemi}")
        hemi_fs = "lh" if hemi == "L" else "rh"
        subj_dir = op.join(density_dir, participant, 'wang_MT')
        subj_densities = []
        subj_present = []

        # Loop through *tracts in fixed order*
        for tract in tract_order:
            # Find file matching this tract and hemisphere
            matches = [f for f in os.listdir(subj_dir) if f"{tract}" in f and f"hemi-{hemi_fs}" in f and "fsaverage" in f and f.endswith(f"fsprojdensity{projfrac}frac.mgh")]

            if not matches:
                print(f"   ⚠️ Missing: {tract} ({hemi}) for {participant}")
                # subj_densities.append(np.zeros_like(subj_densities[0]) if subj_densities else None)
                subj_densities.append(np.zeros([163842, 1, 1])) #if there is no match, replace by map of zeros
                subj_present.append(np.zeros([163842, 1, 1]))
                continue

            # Load the file
            file_path = op.join(subj_dir, matches[0])
            img = nib.load(file_path)
            data = img.get_fdata().astype(np.float32)
            subj_densities.append(data)
            subj_present.append(np.ones([163842, 1, 1]))

        # Stack into one array: shape (n_tracts, n_vertices)
        subj_densities = np.stack(subj_densities, axis=0)  # (n_tracts, n_vertices)
        subj_present = np.stack(subj_present, axis=0)
        density_data[hemi].append(subj_densities)
        density_mask[hemi].append(subj_present)

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
localizer_order = ["mtlocal", "ptlocal"]
contrast_data = {hemi: [] for hemi in hemis}

for participant in participants:
    if not participant.startswith("sub-"):
        continue

    print(f"\n🔹 Participant: {participant}")
    contrasts_dir = op.join(func_dir, participant, 'glm', 'contrasts')
    # EB --> ptlocal only
    # NS --> mtlocal + ptlocal
    if "EB" in participant:
            localizers = ["ptlocal"]
    elif "NS" in participant:
            localizers = ["mtlocal", "ptlocal"]

    for hemi in hemis:
        print(f"   🧩 Hemisphere: {hemi}")

        # Dictionary for this subject: 
        # Structure: {localizer: contrast: [runs]}
        subj_contrast_runs = {
            localizer: {
             contrast: [] 
             for contrast in contrast_order
             }
             for localizer in localizers
        }

        for run in range(0, 6):
            print(f"  🧩 Run: {run+1}")
            for localizer in localizers:
                print(f" Localizer: {localizer}")
                for contrast in contrast_order:
                    print(f"      🧩 Contrast: {contrast}")

                    # Build required filename parts
                    required = [localizer, f"hemi-{hemi}", f"run-{run+1}", "fsaverage", contrast, "tstat"]

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
                    subj_contrast_runs[localizer][contrast].append(data)

        # After all runs → convert each contrast’s list into an array
        subj_final = {}
        for localizer in localizers:
            subj_final[localizer] = {}

            for contrast in contrast_order:
                runs = subj_contrast_runs[localizer][contrast]
                if len(runs) == 0:
                    continue
                subj_final[localizer][contrast] = np.stack(runs, axis=0)  
                # shape = (n_runs, n_vertices)

        # Save this subject's data for this hemisphere
        contrast_data[hemi].append(subj_final)
#contrast_data[hemi][0][contrast][0].shape #dim 1 = hemisphere initial, dim 2 = participant number, dim 3 = contrast type, dim 4: run number of each contrast map

#================== Prepare and Normalize X and Y data for model fit =============
hemis = ["L", "R"]
contrast = contrast_order[0]   # e.g. "motionXstationary"
localizer_order = ["mtlocal", "ptlocal"]
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

        # EB --> ptlocal only
        # NS --> mtlocal + ptlocal
        if "EB" in participant:
                localizers = ["ptlocal"]
        elif "NS" in participant:
                localizers = ["mtlocal", "ptlocal"]

        norm_contrast_data[hemi][s] = {}
        for localizer in localizers:
            norm_contrast_data[hemi][s][localizer] = {}
            #Prepare subject's Y = functional maps
            # get this subject's run maps of the chosen contrast
            C_full = subj_contrasts[s][localizer][contrast]              # (n_runs, n_vertices_fullspace)
    
            # mask ROI
            C = np.squeeze(C_full[:, wang_hmt_vertices] )         # (n_runs, n_masked)
            #zscore the runs for a given participant
            n_runs = C.shape[0]
            print(f"\nSubject {s+1}: {n_runs} runs (hemi {hemi})")
            zscored_C = np.array([(C[r_num,:] - np.mean(C[r_num,:]))/np.std(C[r_num,:]) for r_num in range(n_runs)])

            # norm_contrast_data[hemi].setdefault(s, {})
            norm_contrast_data[hemi][s][localizer][contrast] = zscored_C
            
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

#================Useful functions ============
#Define a goodness of fit function
def r2_score(y_t, y_p):
    ss_res = np.sum((y_t - y_p)**2)
    ss_tot = np.sum((y_t - np.mean(y_t))**2)
    return 1 - ss_res / ss_tot


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


def noise_normalized_r(y_true, y_pred, reliability):
    """
    y_true: (n_vertices,)
    y_pred: (n_vertices,)
    reliability: split-half reliability of y_true

    Returns noise-normalized r
    """
    if reliability <= 0 or np.isnan(reliability):
        return np.nan

    r, _ = pearsonr(y_true, y_pred)
    return r / np.sqrt(reliability)

## Fit Regression Models to each participant
verbose = True

hemis = ["L", "R"]
contrast = contrast_order[0]   # e.g. "motionXstationary"
localizer_order = ["mtlocal", "ptlocal"]

# get n_subj, n_tracts
n_subj = len(participants)
n_tracts = len(tract_order)
n_localizers = len(localizer_order)
n_hemis = len(hemis)
n_target_runs = 3

rs   = np.full((n_subj, n_tracts, n_localizers, n_hemis), np.nan)
rsquared = np.full((n_subj, n_tracts, n_localizers, n_hemis), np.nan) #goodness of fit
reliability = np.full((n_subj, n_tracts, n_localizers, n_hemis), np.nan)
rnd_run_idx = np.full((n_subj, n_target_runs, n_localizers, n_hemis), np.nan) #storage of run indices of EB runs used for this analysis
trained_coefs = np.zeros((n_subj, n_tracts, n_localizers, n_hemis))  # scalar summary per tract/run
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
    predicted = np.full((n_localizers, n_subj, n_tracts, n_masked), np.nan)  # predicted maps per run
    _, _, n_vertices  = density_data[hemi].shape #get total number of vertices within fsaverage hemisphere
    

    all_C = np.full((n_localizers, n_subj, n_target_runs, n_masked), np.nan)             # will store (n_subj, 3, n_vertices)
    for s, participant in enumerate(participants):
        # EB --> ptlocal only
        # NS --> mtlocal + ptlocal
        if "EB" in participant:
                localizers = ["ptlocal"]
        elif "NS" in participant:
                localizers = ["mtlocal", "ptlocal"]
        for l, localizer in enumerate(localizers):
            
            C_full = norm_contrast_data[hemi][s][localizer][contrast]   # (n_runs_available, n_vertices)
            n_runs_available = C_full.shape[0]

            # ---- choose runs ----
            if "NS" in participant:
                run_idx = np.arange(n_target_runs)
            elif "EB" in participant:
                run_idx = np.random.choice(
                    n_runs_available, size=n_target_runs, replace=False
                )

            rnd_run_idx[s, :, l, h] = run_idx #store runs indices of runs used for this analysis

            # ---- extract runs ----
            C_sel = C_full[run_idx, :]     # (3, n_vertices)
            all_C[l,s,] = C_sel
    # all_C = np.stack(all_C[localizer], axis=0)
    C_mean = all_C.mean(axis=2)


    # -------------------------
    # main loop
    # -------------------------

    for s, participant in enumerate(participants):

        if verbose:
            print(f"Fitting participant {participant}")
        # EB --> ptlocal only
        # NS --> mtlocal + ptlocal
        if "EB" in participant:
                localizers = ["ptlocal"]
        elif "NS" in participant:
                localizers = ["mtlocal", "ptlocal"]
        for l, localizer in enumerate(localizers):
            for tract_idx in range(n_tracts):

                if verbose:
                    print(f" Tract {tract_idx+1}/{n_tracts}")

                X = np.squeeze(norm_density_data[hemi][s][tract_idx,]).reshape(-1, 1)
                y = np.squeeze(C_mean[l, s, :]) #.reshape(-1, 1)


                # Train linear model (multi-output regression)
                linreg = LinearRegression()
                linreg.fit(X, y)

                trained_coefs[s, tract_idx, l, h] = linreg.coef_.copy()
                # linreg.intercept_

                # Predict (optional)
                y_pred = linreg.predict(X)
                predicted[l, s, tract_idx, :] = y_pred
            

                # Reliability (if needed)
                reliability[s, tract_idx, l, h] = vertex_bootstrap_reliability(all_C[l,s,:,:])

                # Optional evaluation
                r, p = pearsonr(y, y_pred)
                rs[s, tract_idx, l, h] = r
                mse_full = mean_squared_error(y, y_pred)
                print(f"r={r:.4f}, MSE={mse_full:.4e}, p={p:.4e}")

    predicted_maps[hemi] = predicted

#
import sys
current_dir = op.dirname(op.abspath(__file__))
sys.path.append(current_dir)
from plots_utils_15_4 import pearsonr_plot, beta_plot, avg_density_plot

#Performance plots
savefilename = f"wb_pearson_linearreg_separate_tracts_{projfrac}frac.png"
pearsonr_plot(rs, reliability, participants, tract_order, bids_path, savefilename)

savefilename = f"wb_beta_barplot_linearreg_separate_tracts_{projfrac}frac.png"
ylim = [-0.4, 0.4]
beta_plot(trained_coefs, participants, tract_order, bids_path, savefilename, ylim)


#Data plots
img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "pyAFQ3beta_targ_participants_linearreg",  "surface_pngs", "mean_density")
tract_order2 = ['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF'] 
avg_density_plot(norm_density_data, participants, tract_order2, bids_path, img_out_dir, projfrac)

#------------------------------------------------------------
# Save tables for statistical analyses
#------------------------------------------------------------

#-----------------------------
# Organize Beta coefs in table
#-----------------------------
import pandas as pd
n_tracts = len(tract_order)
hemi_labels = ["L", "R"]
rows = []

for h in range(2):     # left/right hemispheres
    for t in range(n_tracts):
        for s, participant in enumerate(participants):
            gp = "EB" if "EB" in participant else "NS"
            # pearson = rs[:, s, h]
            # EB --> ptlocal only
            # NS --> mtlocal + ptlocal
            if "EB" in participant:
                    localizers = ["ptlocal"]
            elif "NS" in participant:
                    localizers = ["mtlocal", "ptlocal"]
            for l, localizer in enumerate(localizers):
                rows.append({
                "Subject": s,
                "Localizer": localizer,
                "Tract": tract_order[t],
                "Hemisphere": hemi_labels[h],
                "Group": gp,
                "Beta": trained_coefs[s,t,l,h] #r_all[s, h] #
            })

df = pd.DataFrame(rows)

# --------------------------------------
# Organize Pearson's r in table
# --------------------------------------

hemi_labels = ["L", "R"]
rows = []

for h in range(2):     # left/right hemispheres
    for t in range(n_tracts):
        for s, participant in enumerate(participants):
            gp = "EB" if "EB" in participant else "NS"
            # pearson = rs[:, s, h]
            # EB --> ptlocal only
            # NS --> mtlocal + ptlocal
            if "EB" in participant:
                    localizers = ["ptlocal"]
            elif "NS" in participant:
                    localizers = ["mtlocal", "ptlocal"]
            for l, localizer in enumerate(localizers):
                rows.append({
                "Subject": s,
                "Localizer": localizer,
                "Tract": tract_order[t],
                "Hemisphere": hemi_labels[h],
                "Group": gp,
                "Correlation": rs[s,t,l,h] #r_all[s, h] #
            })

df = pd.DataFrame(rows)