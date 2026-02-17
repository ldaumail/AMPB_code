#Loic Daumail
#Started on 11/10/2025
#Trains a linear regression to predict functional activation based on tract end point densitiess

import numpy as np
from sklearn.linear_model import RidgeCV
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

#================Useful functions ============
#Define a goodness of fit function
def r2_score(y_t, y_p):
    ss_res = np.sum((y_t - y_p)**2)
    ss_tot = np.sum((y_t - np.mean(y_t))**2)
    return 1 - ss_res / ss_tot

#Noise ceiling
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

#---------------------------
## Fit linear model to data
#---------------------------
# Cross validation params
alphas = np.logspace(-4, 4, 25)      # Ridge alpha grid (adjust if needed)
inner_cv = 5                         # internal CV to choose alpha (could use LeaveOneOut if many subjects)
verbose = True

hemis = ["L", "R"]
groups = ["EB", "NS"]
contrast = contrast_order[0]   # e.g. "motionXstationary"

# get n_subj, n_tracts
n_subj = len(participants)
n_tracts = len(tract_order)

rnd_run_idx = np.full((n_subj, 3, len(hemis)), np.nan)

#Performance metrics
rs   = np.full((n_subj, len(hemis)), np.nan)
rsquared = np.full(( n_subj, len(hemis)), np.nan) #goodness of fit
reliability = np.full((n_subj, len(hemis)), np.nan)
noise_norm_r = np.full((n_subj, len(hemis)), np.nan)
mses = np.full((n_subj, len(hemis)), np.nan) #
delta_mse = np.full((n_tracts, n_subj, len(hemis)), np.nan) #

#Model outputs
predicted_maps = {hemi: [] for hemi in hemis}
trained_coefs = np.zeros((n_tracts, n_subj, len(hemis)))  # scalar summary per tract/run

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
    n_vertices  = len(surf_roi)#density_data[hemi].shape #get total number of vertices within fsaverage hemisphere
    
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

    for g, group in enumerate(groups):
        participants_group = [p for p in participants if group in p]
        n_g_subj = len(participants_group)
        #Define X and Y of the given group
        group_norm_density = [norm_density_data[hemi][p] for p in range(n_subj) if participants[p] in participants_group]
        group_C_mean = [C_mean[p,:] for p in range(n_subj) if participants[p] in participants_group]
        
        if verbose:
                print(f" Cross-validating {group} group")
        # -------------------------
        # main loop
        # -------------------------
        for test_idx in range(n_g_subj):
            #Prepare subject's Y = functional maps
            # get this subject's run maps of the chosen contrast
            
            # training participants indices
            train_idx = [i for i in range(n_g_subj) if i != test_idx]   

            if verbose:
                print(f" Left out participant {participants_group[test_idx]}")


            X_train = np.vstack([group_norm_density[i].T for i in train_idx])   # (n_train*n_masked, n_tracts)
            y_train = np.hstack([group_C_mean[i] for i in train_idx])  # (n_train*n_masked,)


            # Save X_train maps
            # ref_img_for_save = nib.load(wang_hmt_path)
            # ref_affine = ref_img_for_save.affine
            # ref_header = ref_img_for_save.header
            # map_dir = op.join(bids_path, 'analysis', 'example_maps', 'density_maps', participant)
            # os.makedirs(map_dir, exist_ok=True)
            # dens_maps_all = np.vstack([zscored_densities for _r in train_idx]).reshape(n_tracts, len(train_idx), n_masked).transpose(2, 0, 1)
            # idx = 0
            # for tr in range(len(train_idx)):
            #     for t, tract in enumerate(tract_order):
            #         dens_map = dens_maps_all[:,t, tr] #dens_maps_all[idx, :]
            #         idx += 1 #
            #         dens_full = np.full((n_vertices), np.nan)
            #         dens_full[wang_hmt_vertices] = dens_map
            #         dens_map = dens_full.reshape((1, 1, n_vertices)).astype(np.float32)
            #         dens_out = op.join(map_dir, f"{participant}_hemi-{hemi}_label-{tract}_trrun-{tr+1}_desc-training_density.mgz")
            #         nib.save(nib.MGHImage(dens_map, ref_affine, ref_header), dens_out)
            # Save y_train maps
            # ref_img_for_save = nib.load(wang_hmt_path)
            # ref_affine = ref_img_for_save.affine
            # ref_header = ref_img_for_save.header
            # map_dir = op.join(bids_path, 'analysis', 'example_maps', 'beta_maps', participant)
            # os.makedirs(map_dir, exist_ok=True)
            # beta_maps_all = np.squeeze(zscored_C[train_idx, :]).transpose(1,0)
            # idx = 0
            # for tr in range(len(train_idx)):
            #     dens_map = beta_maps_all[:, tr] #dens_maps_all[idx, :]
            #     idx += 1 #
            #     dens_full = np.full((n_vertices), np.nan)
            #     dens_full[wang_hmt_vertices] = dens_map
            #     dens_map = dens_full.reshape((1, 1, n_vertices)).astype(np.float32)
            #     dens_out = op.join(map_dir, f"{participant}_hemi-{hemi}_trrun-{tr+1}_desc-training_beta.mgz")
            #     nib.save(nib.MGHImage(dens_map, ref_affine, ref_header), dens_out)


            # Train linear model (multi-output regression)
            ridge = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error", cv=inner_cv)
            ridge.fit(X_train, y_train)

            
            all_test_idx = test_idx if group == "EB" else test_idx+7
            trained_coefs[:, all_test_idx,h] = ridge.coef_.copy() #[0:n_tracts-1]
            # ridge.intercept_

            X_test = group_norm_density[test_idx].T 
            y_pred_std = ridge.predict(X_test)
            # y_pred = (y_pred_std*np.std(C[test_idx,:]) + np.mean(C[test_idx,:])).ravel()

            predicted[all_test_idx, :] = y_pred_std
            reliability[all_test_idx, h] = vertex_bootstrap_reliability(all_C[all_test_idx,:,:])

            # Evaluate this test_participant if verbose
            y_participant_true = np.squeeze(group_C_mean[test_idx])
            r_participant, p_participant = pearsonr(np.squeeze(y_participant_true), y_pred_std)
            rs[all_test_idx,h] = r_participant
            noise_norm_r[all_test_idx,h] = noise_normalized_r(np.squeeze(y_participant_true), y_pred_std,reliability[all_test_idx, h])
            mse_participant_full = mean_squared_error(np.squeeze(y_participant_true), y_pred_std)
            rsquared[all_test_idx,h] = r2_score(np.squeeze(y_participant_true), y_pred_std)

            print(f"Participant r:{r_participant:.4f}, MSE={mse_participant_full:.4e}, p={p_participant:.4e}")

            # Nested models ridge regression
            for t in range(n_tracts):

                X_train_red = np.delete(X_train, t, axis=1)
                ridgereg_red = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error", cv=inner_cv)
                ridgereg_red.fit(X_train_red, y_train)

                X_test_red = np.delete(X_test, t, axis=1)
                y_test_pred_red = ridgereg_red.predict(X_test_red).ravel()
                mse_red = mean_squared_error(y_participant_true, y_test_pred_red)
                delta_mse[t, all_test_idx, h] = mse_red - mse_participant_full


    #------------------
    #Performance metrics
    #------------------
    #overall correlation across all participants (concatenated)
    predicted_full = np.full((n_subj, n_vertices), np.nan)
    predicted_full[:, wang_hmt_vertices] = predicted

    true_full = np.full((n_subj, n_vertices), np.nan)
    true_full[:, wang_hmt_vertices] = C_mean

    # Coefficients already organized: (n_subj, n_tracts)
    coefs_arr = trained_coefs.copy()

    # Mean coefficient per tract
    mean_coefs = np.mean(coefs_arr, axis=0)

    print("\nDone.")

    predicted_maps[hemi] = predicted
    print(f"\nFinished hemisphere {hemi}")

#------------------------------------------------------------------------
# Full Model plots
#------------------------------------------------------------------------

    # Optionally save predicted maps per subject using a reference image
# ----------------------------------------------------------
# Save predicted maps 
# ----------------------------------------------------------
# ----------------------------------------------------------
# Save predicted maps (all 3 runs, with NaN-check)
# ----------------------------------------------------------

N_RUNS_SAVED = 3

predicted_full = np.full((n_subj, N_RUNS_SAVED, n_vertices), np.nan)
predicted_full[:, :, wang_hmt_vertices] = predicted

out_dir = op.join(bids_path, 'analysis', 'example_maps', 'predicted_beta_maps')
os.makedirs(out_dir, exist_ok=True)

ref_img_for_save = nib.load(wang_hmt_path)
ref_affine = ref_img_for_save.affine
ref_header = ref_img_for_save.header

for s in range(n_subj):

    participant = participants[s]
    subj_dir = op.join(out_dir, participant)
    os.makedirs(subj_dir, exist_ok=True)

    if "EB" in participant:
        task = "ptlocal"
    elif "NS" in participant:
        task = "mtlocal"
    else:
        task = "localizer"

    # -----------------------------------------------------
    # Save predicted map for each tract and each run
    # -----------------------------------------------------
    for r in range(N_RUNS_SAVED):

        pred_map = predicted_full[s, r, :]      # (n_vertices,)
        pred_map = pred_map.reshape((1, 1, n_vertices)).astype(np.float32)

        # --- NEW CHECK: skip run if map is empty ---
        if np.isnan(pred_map).all():
            continue

        pred_out = op.join(
            subj_dir,
            f"{participant}_hemi-{hemi}_run-{r+1}_desc-predicted_contrast.mgz"
        )

        nib.save(nib.MGHImage(pred_map, ref_affine, ref_header), pred_out)

print(f"Saved predicted maps to: {out_dir}")

#--------------------------
#Save Plot of predicted map
#--------------------------
from nilearn import plotting
from nibabel.freesurfer import read_label
import matplotlib.pyplot as plt

fs_path = op.join(bids_path, 'derivatives', 'freesurfer')

for h, hemi in enumerate(hemis):
    hemi_fs = "lh" if hemi == "L" else "rh"
    infl_surf = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.inflated")
    # ----------------------------
    # Load curvature map (sulci/gyri)
    # ----------------------------
    curv_file = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv")
    curv = nib.freesurfer.read_morph_data(curv_file)

    # normalize curvature for nicer background display
    curv_norm = (curv - np.percentile(curv, 5)) / (
        np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8
    )
    curv_norm = np.clip(curv_norm, 0, 1)

        # ----------------------------
    # Load Wang MT ROI
    # ----------------------------
    wang_hmt_path = op.join(
        '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
        f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
    )
    surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
    wang_hmt_vertices = np.where(surf_roi > 0)[0]

    for s, participant in enumerate(participants):
        # ----------------------------
        # Functional MT ROI (binary surface map)
        # ----------------------------
        label_file = op.join(bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi', participant,
            f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label")

        func_mt_vertices = read_label(label_file)

        func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
        func_mt_roi[func_mt_vertices] = 1

        # ----------------------------
        # Predicted Map visualization
        # ----------------------------
        img_out_dir = op.join(bids_path, "analysis", "surface_pngs", participant)
        os.makedirs(img_out_dir, exist_ok=True)

        vmin, vmax = -1.0, 1.0
        # -------------------------------------------------
        # Compute average functional map across runs FIRST
        # -------------------------------------------------

        # Build full-surface vector 
        surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
        surf_map[wang_hmt_vertices] = np.nanmean(predicted_maps[hemi][s, :, :], axis=0)

        # Output filename (no run index)
        out_png = op.join(
            img_out_dir,
            f"{participant}_hemi-{hemi}_desc-pred-motionXstationary_mean_inflated.png"
        )

        # -------------------------------------------------
        # Plot only once
        # -------------------------------------------------

        display = plotting.plot_surf_stat_map(
            surf_mesh=infl_surf,
            stat_map=surf_map,
            hemi="left" if hemi == "L" else "right",
            view="lateral",
            cmap="plasma",
            colorbar=True,
            vmin=vmin,
            vmax=vmax,
            threshold=None,
            bg_map=curv_norm,
            bg_on_data=True,
            darkness=0.6,
        )

        # ---- MT boundary overlay ----
        plotting.plot_surf_contours(
            surf_mesh=infl_surf,
            roi_map=func_mt_roi,
            levels=[1],
            colors=["lightgray"],
            linewidths=2.0,
            figure=display.figure,
            axes=display.axes[0]
        )

        # ---- save + close ----
        display.savefig(out_png, dpi=300)
        plt.close(display.figure)

#--------------------------------------------------------------
#----------------------------------------------
# Plot average predicted contrast map across participants
#----------------------------------------------

hemis = ["L", "R"]
groups = ["EB", "NS"]
for h, hemi in enumerate(hemis):
    for g, group in enumerate(groups):
        infl_surf = op.join(fs_path, "fsaverage", "surf", f"{'lh' if hemi == 'L' else 'rh'}.inflated")
        _, _, n_vertices  = density_data[hemi].shape
        #-----------------------------
        # Load func MT ROI
        #-----------------------------
        label_file = op.join(
            bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi',
            'group_averages',
            f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
        )

        func_mt_vertices = read_label(label_file)

        func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
        func_mt_roi[func_mt_vertices] = 1
        # ----------------------------
        # Load Wang MT ROI 
        # ----------------------------
        wang_hmt_path = op.join(
            '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
            f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
        )
        surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
        wang_hmt_vertices = np.where(surf_roi > 0)[0]
        print(f"{len(wang_hmt_vertices)} vertices in ROI ({hemi})")
        # ----------------------------
        # Load curvature map (sulci/gyri)
        # ----------------------------
        hemi_fs = "lh" if hemi == "L" else "rh"
        curv_file = op.join(fs_path, "fsaverage", "surf", f"{hemi_fs}.curv")
        curv = nib.freesurfer.read_morph_data(curv_file)

        # normalize curvature for nicer background display
        curv_norm = (curv - np.percentile(curv, 5)) / (
            np.percentile(curv, 95) - np.percentile(curv, 5) + 1e-8
        )
        curv_norm = np.clip(curv_norm, 0, 1)

        # ----------------------------
        # Functional Contrast Map visualization
        # ----------------------------
        img_out_dir = op.join(bids_path, "analysis", "surface_pngs", "mean")
        os.makedirs(img_out_dir, exist_ok=True)

        vmin, vmax = -.5, 0.5
        # -------------------------------------------------
        # Compute average functional map across runs FIRST
        # -------------------------------------------------
        hemi_maps = predicted_maps[hemi]
        sub_mean_maps = np.nanmean(hemi_maps[[group in p for p in participants],:,:], axis = 1)
        mean_vals = np.nanmean(sub_mean_maps, axis=0)

        # Build full-surface vector once
        surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
        surf_map[wang_hmt_vertices] = mean_vals

        # Output filename (no run index)
        out_png = op.join(
            img_out_dir,
            f"{group}-mean_hemi-{hemi}_desc-pred-motionXstationary_mean_inflated.png"
        )

        # -------------------------------------------------
        # Plot average participant map
        # -------------------------------------------------

        display = plotting.plot_surf_stat_map(
            surf_mesh=infl_surf,
            stat_map=surf_map,
            hemi="left" if hemi == "L" else "right",
            view="lateral",
            cmap="plasma",
            colorbar=True,
            vmin=vmin,
            vmax=vmax,
            threshold=None,
            bg_map=curv_norm,
            bg_on_data=True,
            darkness=0.6,
        )
        # ---- MT boundary overlay ----
        plotting.plot_surf_contours(
            surf_mesh=infl_surf,
            roi_map=func_mt_roi,
            levels=[1],
            colors=["lightgray"],
            linewidths=2.0,
            figure=display.figure,
            axes=display.axes[0]
        )

        # ---- save + close ----
        display.savefig(out_png, dpi=300)
        plt.close(display.figure)

#--------------------------------------------------------------
#Pearson's r
# =====================================
# SCATTER + JITTER PLOT BY GROUP × HEMI × TRACT
# =====================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

#First compute noise ceiling 95% CI
hemi_labels = ["L", "R"]
nc_rows = []

for h in range(2):     # left/right hemispheres
    for s, participant in enumerate(participants):
        gp = "EB" if "EB" in participant else "NS"

        nc_rows.append({
            "Subject": s,
            "Hemisphere": hemi_labels[h],
            "Group": gp,
            "Correlation": reliability[s,h] #r_all[s, h] #
        })

nc_df = pd.DataFrame(nc_rows)
nc_sem_df = (
    nc_df.groupby(["Group", "Hemisphere"])["Correlation"]
      .agg(["mean", sem])
      .reset_index()
      .rename(columns={"mean": "Mean", "sem": "SEM"})
)
nc_sem_df["CI95_upper"] = nc_sem_df["Mean"] + 1.96 * nc_sem_df["SEM"]
nc_sem_df["CI95_lower"] = nc_sem_df["Mean"] - 1.96 * nc_sem_df["SEM"]

# --------------------------------------
# Organize Pearson's r in table
# --------------------------------------

hemi_labels = ["L", "R"]
rows = []

for h in range(2):     # left/right hemispheres
    for s, participant in enumerate(participants):
        gp = "EB" if "EB" in participant else "NS"
        # pearson = rs[:, s, h]

        rows.append({
            "Subject": s,
            "Hemisphere": hemi_labels[h],
            "Group": gp,
            "Correlation": rs[s,h] #r_all[s, h] #
        })

df = pd.DataFrame(rows)

# ------------------------------------------------
# Compute SEM per Group × Hemisphere
# ------------------------------------------------
sem_df = (
    df.groupby(["Group", "Hemisphere"])["Correlation"]
      .agg(["mean", sem])
      .reset_index()
      .rename(columns={"mean": "Mean", "sem": "SEM"})
)

# ------------------------------------------------
# Color palette: EB = blue, NS = orange
# ------------------------------------------------
palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

# ------------------------------------------------
# Create 2 subplots — one per hemisphere
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, hemi in zip(axes, hemi_labels):
    # ---- Noise ceiling 95% CI upper bound ----
    nc_h = nc_sem_df[nc_sem_df["Hemisphere"] == hemi]

    df_h = df[df["Hemisphere"] == hemi]
    sem_h = sem_df[sem_df["Hemisphere"] == hemi]

    # Jittered dots
    sns.stripplot(
        data=df_h,
        x="Group",
        y="Correlation",
        hue="Group",
        dodge=False,
        jitter=0.15,
        alpha=0.7,
        palette=palette,
        ax=ax
    )

    # ---- Mean ± SEM (model performance) ----
    for _, row in sem_h.iterrows():
        group = row["Group"]
        mean = row["Mean"]
        se = row["SEM"]

        x_loc = 0 if group == "EB" else 1

        ax.errorbar(
            x=x_loc,
            y=mean,
            yerr=se,
            fmt="o",
            color="black",
            markersize=8,
            capsize=3,
            linewidth=2
        )
    # ---- Noise ceiling 95% CI upper/lower bound/mean ----
    for _, row in nc_h.iterrows():
        group = row["Group"]
        up_ci = row["CI95_upper"]
        low_ci = row["CI95_lower"]
        mean = row["Mean"]

        x_center = 0 if group == "EB" else 1

        ax.hlines(
            y=[low_ci, up_ci, mean],
            xmin=x_center - 0.25,
            xmax=x_center + 0.25,
            colors="gray",
            linestyles=["--","--", ":"],
            linewidth=3,
            alpha=0.9
        )
        ax.fill_between(
        [x_center - 0.25, x_center + 0.25],
        low_ci,
        up_ci,
        color="pink",
        alpha=0.2,
        linewidth=0
        )

    # Formatting
    ax.set_ylim(-0.2, 1)
    ax.set_title(f"{hemi}-Hemisphere", fontsize=16)
    ax.set_xlabel("Group", fontsize=14)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticklabels(["EB", "NS"], fontsize=13)

axes[0].set_ylabel("Mean Pearson's r", fontsize=14)
# axes[1].get_legend().remove()   # remove duplicated legend
sns.despine()
plt.tight_layout()

#Saving
saveDir = op.join(bids_path, "analysis", "plots")
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, "pearson_mean_ridgereg_loso_combined_tracts.png"), dpi=300, bbox_inches='tight')
plt.show()

#----------------------------
### Plot beta values
#----------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# --------------------------------------
# Convert trained_coefs into long format
# trained_coefs shape = (6 runs, n_tracts, n_subj, 2 hemis)
# Requires subject_group: list of "EB" or "NS"
# --------------------------------------

n_tracts, n_subj, n_hemi = trained_coefs.shape
hemi_labels = ["L", "R"]

rows = []
for h in range(n_hemi):
    for t in range(n_tracts):
        for s, participant in enumerate(participants):
            gp = "EB" if "EB" in participant else "NS"
            betas = trained_coefs[t, s, h]   # shape (6,)
            if np.isnan(betas).all():
                continue
            
            rows.append({
                "Tract": tract_order[t],
                "Participant": participant,
                "Hemisphere": hemi_labels[h],
                "Group": gp,    # EB or NS
                "Beta": betas
            }) # "Subject": s,

df = pd.DataFrame(rows)


# ------------------------------------------------
# Compute SEM per tract × hemisphere × group (EB/NS)
# ------------------------------------------------
sem_df = (
    df.groupby(["Group", "Tract", "Hemisphere"])["Beta"]
      .agg(["mean", sem])
      .reset_index()
      .rename(columns={"mean": "Mean", "sem": "SEM"})
)

# ------------------------------------------
# Create 2 subplots — one for each hemisphere
# ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
group_labels = ["EB", "NS"]
group_offset = {"EB": -0.2, "NS": 0.2}   # shift inside each tract

for ax, hemi in zip(axes, hemi_labels):

    # Filter for hemisphere
    df_h = df[df["Hemisphere"] == hemi]
    sem_h = sem_df[sem_df["Hemisphere"] == hemi]

    # Jitter dots per group (EB vs NS)
    sns.stripplot(
        data=df_h,
        x="Tract",
        y="Beta",
        hue="Group",
        dodge=True,
        jitter=0.15,
        alpha=0.7,
        ax=ax
    )

    # ---------------------------------------
    # Plot Mean ± SEM for EB and NS separately
    # ---------------------------------------
    for _, row in sem_h.iterrows():

        tract = row["Tract"]
        mean  = row["Mean"]
        se    = row["SEM"]
        group = row["Group"]

        # shift within tract index to match stripplot's dodge layout
        x_loc = tract_order.index(tract)  + group_offset[group]

        # Mean point
        ax.plot(x_loc, mean, "o", color="black", markersize=7)

        # SEM bar
        ax.errorbar(
            x=x_loc,
            y=mean,
            yerr=se,
            color="black",
            capsize=3,
            linewidth=2
        )

    # ------------------ Ax formatting ------------------
    ax.set_title(f"{hemi}-Hemisphere β-coefficients (Mean ± SEM within EB / NS)",fontsize=16)
    ax.set_xlabel("Tract",fontsize=14)
    ax.tick_params(axis="x", rotation=90)
    ax.set_xticks(np.arange(len(tract_order)))
    ax.set_xticklabels(tract_order, rotation=30, ha="right",fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
axes[0].set_ylabel("Beta", fontsize=14)
axes[1].legend(title="Group", labels=group_labels)
sns.despine()
plt.tight_layout()
saveDir = op.join(bids_path, 'analysis', 'plots')
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, "betas_ridgereg_group_loso_combined_tracts.png"),
            dpi=300, bbox_inches='tight')
plt.show()


## Bar plots /participant/hemisphere


# ---------------------------------------------------------
# BAR PLOTS: one figure per PARTICIPANT × HEMISPHERE
# ---------------------------------------------------------

saveDir = op.join(bids_path, 'analysis', 'plots')
os.makedirs(saveDir, exist_ok=True)

for s, participant in enumerate(participants):
    for hemi in hemi_labels:

        # subset to this subject & hemisphere
        df_sub = df[(df["Subject"] == s) & (df["Hemisphere"] == hemi)]

        # ensure tract order
        df_sub["Tract"] = pd.Categorical(
            df_sub["Tract"], categories=tract_order, ordered=True
        )
        df_sub = df_sub.sort_values("Tract")

        # create figure
        fig, ax = plt.subplots(figsize=(12, 5))

        # ------------------------
        # Bar plot (one bar per tract)
        # ------------------------
        sns.barplot(
            data=df_sub,
            x="Tract",
            y="MeanBeta",
            color="steelblue",
            ax=ax
        )

        # ------------------------
        # Formatting
        # ------------------------
        ax.set_title(
            f"{participant} | {hemi}-Hemisphere β-coefficients",
            fontsize=14
        )
        ax.set_xlabel("Tract", fontsize=12)
        ax.set_ylabel("Mean Beta", fontsize=12)
        ax.set_xticks(np.arange(len(tract_order)))
        ax.set_xticklabels(tract_order, rotation=30, ha="right", fontsize=10)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(-0.2, 0.3)
        sns.despine()

        plt.tight_layout()

        # ------------------------
        # Save separate file
        # ------------------------
        out_png = op.join(
            saveDir,
            f"{participant}_hemi-{hemi}_betas_barplot.png"
        )
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

#-----------------
# Plot MSE
#-----------------

# =====================================
# SCATTER + JITTER PLOT BY GROUP × HEMI × TRACT
# =====================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# --------------------------------------
# Build long-format dataframe
# r_all shape should be (n_subj, 2 hemis)
# --------------------------------------

hemi_labels = ["L", "R"]
rows = []

for h in range(2):     # left/right hemispheres
    for s, participant in enumerate(participants):
        gp = "EB" if "EB" in participant else "NS"
        mse_runs = mses[:, s, h]

        rows.append({
            "Subject": s,
            "Hemisphere": hemi_labels[h],
            "Group": gp,
            "meanMSE": mse_runs.mean() #r_all[s, h] #
        })

df = pd.DataFrame(rows)

# ------------------------------------------------
# Compute SEM per Group × Hemisphere
# ------------------------------------------------
sem_df = (
    df.groupby(["Group", "Hemisphere"])["meanMSE"]
      .agg(["mean", sem])
      .reset_index()
      .rename(columns={"mean": "Mean", "sem": "SEM"})
)

# ------------------------------------------------
# Color palette: EB = blue, NS = orange
# ------------------------------------------------
palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

# ------------------------------------------------
# Create 2 subplots — one per hemisphere
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, hemi in zip(axes, hemi_labels):

    df_h = df[df["Hemisphere"] == hemi]
    sem_h = sem_df[sem_df["Hemisphere"] == hemi]

    # Jittered dots
    sns.stripplot(
        data=df_h,
        x="Group",
        y="meanMSE",
        hue="Group",
        dodge=False,
        jitter=0.15,
        alpha=0.7,
        palette=palette,
        ax=ax
    )

    # ---- Mean ± SEM ----
    for _, row in sem_h.iterrows():
        group = row["Group"]
        mean = row["Mean"]
        se = row["SEM"]

        x_loc = 0 if group == "EB" else 1

        ax.errorbar(
            x=x_loc,
            y=mean,
            yerr=se,
            fmt="o",
            color="black",
            markersize=8,
            capsize=3,
            linewidth=2
        )

    # Formatting
    ax.set_title(f"{hemi}-Hemisphere", fontsize=16)
    ax.set_xlabel("Group", fontsize=14)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticklabels(["EB", "NS"], fontsize=13)

axes[0].set_ylabel("Mean MSE", fontsize=14)
# axes[1].get_legend().remove()   # remove duplicated legend
sns.despine()
plt.tight_layout()

# Saving
saveDir = op.join(bids_path, "analysis", "plots")
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, "mse_mean_ridgereg_loro_combined_tracts.png"), dpi=300, bbox_inches='tight')

plt.show()


#--------------
#Plot Rsquared
#--------------
# =====================================
# SCATTER + JITTER PLOT BY GROUP × HEMI × TRACT
# =====================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# --------------------------------------
# Build long-format dataframe
# r_all shape should be (n_subj, 2 hemis)
# --------------------------------------

hemi_labels = ["L", "R"]
rows = []

for h in range(2):     # left/right hemispheres
    for s, participant in enumerate(participants):
        gp = "EB" if "EB" in participant else "NS"

        rows.append({
            "Subject": s,
            "Hemisphere": hemi_labels[h],
            "Group": gp,
            "meanR2": rsquared[s,h]#r_all[s, h] #
        })

df = pd.DataFrame(rows)

# ------------------------------------------------
# Compute SEM per Group × Hemisphere
# ------------------------------------------------
sem_df = (
    df.groupby(["Group", "Hemisphere"])["meanR2"]
      .agg(["mean", sem])
      .reset_index()
      .rename(columns={"mean": "Mean", "sem": "SEM"})
)

# ------------------------------------------------
# Color palette: EB = blue, NS = orange
# ------------------------------------------------
palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

# ------------------------------------------------
# Create 2 subplots — one per hemisphere
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, hemi in zip(axes, hemi_labels):

    df_h = df[df["Hemisphere"] == hemi]
    sem_h = sem_df[sem_df["Hemisphere"] == hemi]

    # Jittered dots
    sns.stripplot(
        data=df_h,
        x="Group",
        y="meanR2",
        hue="Group",
        dodge=False,
        jitter=0.15,
        alpha=0.7,
        palette=palette,
        ax=ax
    )

    # ---- Mean ± SEM ----
    for _, row in sem_h.iterrows():
        group = row["Group"]
        mean = row["Mean"]
        se = row["SEM"]

        x_loc = 0 if group == "EB" else 1

        ax.errorbar(
            x=x_loc,
            y=mean,
            yerr=se,
            fmt="o",
            color="black",
            markersize=8,
            capsize=3,
            linewidth=2
        )

    # Formatting
    ax.set_title(f"{hemi}-Hemisphere", fontsize=16)
    ax.set_xlabel("Group", fontsize=14)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticklabels(["EB", "NS"], fontsize=13)

axes[0].set_ylabel("R2", fontsize=14)
# axes[1].get_legend().remove()   # remove duplicated legend
sns.despine()
plt.tight_layout()

# Saving
saveDir = op.join(bids_path, "analysis", "plots")
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, "r2_ridgereg_loso_combined_tracts.png"), dpi=300, bbox_inches='tight')

plt.show()

#--------- Noise normalized R -----------------------------------------


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# --------------------------------------
# Build long-format dataframe
# r_all shape should be (n_subj, 2 hemis)
# --------------------------------------

hemi_labels = ["L", "R"]
rows = []

for h in range(2):     # left/right hemispheres
    for s, participant in enumerate(participants):
        gp = "EB" if "EB" in participant else "NS"

        rows.append({
            "Subject": s,
            "Hemisphere": hemi_labels[h],
            "Group": gp,
            "noiseNormR": noise_norm_r[s,h]#reliability[s,h] 
        })

df = pd.DataFrame(rows)

# ------------------------------------------------
# Compute SEM per Group × Hemisphere
# ------------------------------------------------

sem_df = (
    df.groupby(["Group", "Hemisphere"])["noiseNormR"]
      .agg(
          Mean="mean",
          SEM=lambda x: sem(x, nan_policy="omit")
      )
      .reset_index()
      .rename(columns={"mean": "Mean", "sem": "SEM"})
)

# ------------------------------------------------
# Color palette: EB = blue, NS = orange
# ------------------------------------------------
palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

# ------------------------------------------------
# Create 2 subplots — one per hemisphere
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, hemi in zip(axes, hemi_labels):

    df_h = df[df["Hemisphere"] == hemi]
    sem_h = sem_df[sem_df["Hemisphere"] == hemi]

    # Jittered dots
    sns.stripplot(
        data=df_h,
        x="Group",
        y="noiseNormR",
        hue="Group",
        dodge=False,
        jitter=0.15,
        alpha=0.7,
        palette=palette,
        ax=ax
    )

    # ---- Mean ± SEM ----
    for _, row in sem_h.iterrows():
        group = row["Group"]
        mean = row["Mean"]
        se = row["SEM"]

        x_loc = 0 if group == "EB" else 1

        ax.errorbar(
            x=x_loc,
            y=mean,
            yerr=se,
            fmt="o",
            color="black",
            markersize=8,
            capsize=3,
            linewidth=2
        )

    # Formatting
    ax.set_title(f"{hemi}-Hemisphere", fontsize=16)
    ax.set_xlabel("Group", fontsize=14)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticklabels(["EB", "NS"], fontsize=13)

axes[0].set_ylabel("Noise Normalized r", fontsize=14)
# axes[1].get_legend().remove()   # remove duplicated legend
sns.despine()
plt.tight_layout()

# Saving
saveDir = op.join(bids_path, "analysis", "plots")
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, "noise_norm_r_ridgereg_group_loso_combined_tracts.png"), dpi=300, bbox_inches='tight')

plt.show()

#--------------------------------------------------------------------
# Nested Models
#--------------------------------------------------------------------


#--------------------------------------------------------------
# Delta MSE
#----------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# --------------------------------------
# Convert trained_coefs into long format
# trained_coefs shape = (6 runs, n_tracts, n_subj, 2 hemis)
# Requires subject_group: list of "EB" or "NS"
# --------------------------------------

n_tracts, n_subj, n_hemi = trained_coefs.shape
hemi_labels = ["L", "R"]

rows = []
for h in range(n_hemi):
    for t in range(n_tracts):
        for s, participant in enumerate(participants):
            gp = "EB" if "EB" in participant else "NS"
            
            rows.append({
                "Tract": tract_order[t],
                "Participant": participant,
                "Hemisphere": hemi_labels[h],
                "Group": gp,    # EB or NS
                "dMSE": delta_mse[t, s, h]
            }) # "Subject": s,

df = pd.DataFrame(rows)


# ------------------------------------------------
# Compute SEM per tract × hemisphere × group (EB/NS)
# ------------------------------------------------
std_df = (
    df.groupby(["Group", "Tract", "Hemisphere"])["dMSE"]
      .agg(["mean", "sem"])
      .reset_index()
      .rename(columns={"mean": "Mean", "sem": "SEM"})
)

# ------------------------------------------
# Create 2 subplots — one for each hemisphere
# ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
group_labels = ["EB", "NS"]
group_offset = {"EB": -0.2, "NS": 0.2}   # shift inside each tract

for ax, hemi in zip(axes, hemi_labels):

    # Filter for hemisphere
    df_h = df[df["Hemisphere"] == hemi]
    std_h = std_df[std_df["Hemisphere"] == hemi]

    # Jitter dots per group (EB vs NS)
    sns.stripplot(
        data=df_h,
        x="Tract",
        y="dMSE",
        hue="Group",
        dodge=True,
        jitter=0.15,
        alpha=0.7,
        ax=ax
    )

    # ---------------------------------------
    # Plot Mean ± SEM for EB and NS separately
    # ---------------------------------------
    for _, row in std_h.iterrows():

        tract = row["Tract"]
        mean  = row["Mean"]
        se    = row["SEM"]
        group = row["Group"]

        # shift within tract index to match stripplot's dodge layout
        x_loc = tract_order.index(tract)  + group_offset[group]

        # Mean point
        ax.plot(x_loc, mean, "o", color="black", markersize=7)

        # SEM bar
        ax.errorbar(
            x=x_loc,
            y=mean,
            yerr=se,
            color="black",
            capsize=3,
            linewidth=2
        )

    # ------------------ Ax formatting ------------------
    ax.set_title(f"{hemi}-Hemisphere delta-MSE (Mean ± SEM within EB / NS)",fontsize=16)
    ax.set_xlabel("Tract",fontsize=14)
    ax.tick_params(axis="x", rotation=90)
    ax.set_xticks(np.arange(len(tract_order)))
    ax.set_xticklabels(tract_order, rotation=30, ha="right",fontsize=12)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
axes[0].set_ylabel("delta-MSE", fontsize=14)
axes[1].legend(title="Group", labels=group_labels)
sns.despine()
plt.tight_layout()
saveDir = op.join(bids_path, 'analysis', 'plots')
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, "dMSE_ridgereg_group_LOSO_combined_tracts_nested.png"),
            dpi=300, bbox_inches='tight')
plt.show()

#-------- Correlations from training with Wang MT, within func MT only

rs   = np.full((3, n_subj, len(hemis)), np.nan)
for h, hemi in enumerate(hemis):
    #Load Wang MT
    wang_hmt_path = op.join(
        '/Users','ldaumail3','Documents','research','brain_atlases','Wang_2015','hmtplus',
        f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang_dilated.mgh"
    )
    surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
    wang_hmt_vertices = np.where(surf_roi > 0)[0]

    #Prepare predicted array
    predicted_full = np.full((n_subj, n_runs, n_vertices), np.nan)
    predicted_full[:, :, wang_hmt_vertices] = predicted_maps[hemi]

    for s, participant in enumerate(participants):
        #Load Func MT
        label_file = op.join( bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi', participant,
            f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label")

        func_mt_vertices = read_label(label_file)
        #Prepare true array
        subj_dict = subj_contrasts[s]
        C_full = subj_dict[contrast][np.astype(rnd_run_idx[s,:,h],int),:]  

        #Get predicted func MT
        predicted_funcMT = predicted_full[s,:, func_mt_vertices]
        #get true func MT
        C_funcMT = C_full[:, func_mt_vertices]

        #Calculate correlation for a given run
        for r in range(n_runs):

            y_true = C_funcMT[r, :].reshape(-1)
            if np.isnan(y_true).all():
                print(f"True Contrast Map: NaN, run {r}, participant: {participant}, hemisphere: {hemi}")
            y_pred = predicted_funcMT[:,r].reshape(-1)
            if np.isnan(y_pred).all():
                print(f"Predicted Contrast Map: NaN, run {r}, participant: {participant}, hemisphere: {hemi}")
            r_r, _ = pearsonr(y_true[~np.isnan(y_pred)], y_pred[~np.isnan(y_pred)])
            if np.isnan(r_r).all():
                print(f"Correlation: NaN, run {r}, participant: {participant}, hemisphere: {hemi}")
            rs[r, s, h] = r_r
            


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem

# --------------------------------------
# Build long-format dataframe
# r_all shape should be (n_subj, 2 hemis)
# --------------------------------------

hemi_labels = ["L", "R"]
rows = []

for h in range(2):     # left/right hemispheres
    for s, participant in enumerate(participants):
        gp = "EB" if "EB" in participant else "NS"
        pearson = rs[:, s, h]

        rows.append({
            "Subject": s,
            "Hemisphere": hemi_labels[h],
            "Group": gp,
            "Correlation": pearson.mean() #r_all[s, h] #
        })

df = pd.DataFrame(rows)

# ------------------------------------------------
# Compute SEM per Group × Hemisphere
# ------------------------------------------------
sem_df = (
    df.groupby(["Group", "Hemisphere"])["Correlation"]
      .agg(["mean", sem])
      .reset_index()
      .rename(columns={"mean": "Mean", "sem": "SEM"})
)

# ------------------------------------------------
# Color palette: EB = blue, NS = orange
# ------------------------------------------------
palette = {"EB": "#1f77b4", "NS": "#ff7f0e"}

# ------------------------------------------------
# Create 2 subplots — one per hemisphere
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, hemi in zip(axes, hemi_labels):

    df_h = df[df["Hemisphere"] == hemi]
    sem_h = sem_df[sem_df["Hemisphere"] == hemi]

    # Jittered dots
    sns.stripplot(
        data=df_h,
        x="Group",
        y="Correlation",
        hue="Group",
        dodge=False,
        jitter=0.15,
        alpha=0.7,
        palette=palette,
        ax=ax
    )

    # ---- Mean ± SEM ----
    for _, row in sem_h.iterrows():
        group = row["Group"]
        mean = row["Mean"]
        se = row["SEM"]

        x_loc = 0 if group == "EB" else 1

        ax.errorbar(
            x=x_loc,
            y=mean,
            yerr=se,
            fmt="o",
            color="black",
            markersize=8,
            capsize=3,
            linewidth=2
        )

    # Formatting
    ax.set_title(f"{hemi}-Hemisphere", fontsize=16)
    ax.set_xlabel("Group", fontsize=14)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticklabels(["EB", "NS"], fontsize=13)

axes[0].set_ylabel("Mean Pearson's r", fontsize=14)
# axes[1].get_legend().remove()   # remove duplicated legend
sns.despine()
plt.tight_layout()

# Saving
saveDir = op.join(bids_path, "analysis", "plots")
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, "funcMT_pearson_mean_ridgereg_loro_combined_tracts.png"), dpi=300, bbox_inches='tight')

plt.show()
