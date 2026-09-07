
# ✅ Fixed tract order (keep consistent across subjects!)
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
#tract_order = ['CallosumOccipital', 'VerticalOccipital', 'InferiorFrontooccipital', 'InferiorLongitudinal'] 

tract_order = [
    'PTR', 
    'InferiorLongitudinal', 
    'InferiorFrontooccipital', 
    'SuperiorLongitudinalI',
    'SuperiorLongitudinalII',
    'SuperiorLongitudinalIII',
    'AnteriorVerticalOccipital', 
    'PosteriorVerticalOccipital',
    'Arcuate',
    'PosteriorArcuate',
    'EarlyVisual',
    'OpticRadiation',
    'Temporoparietal'
    ] 
participants = sorted([p for p in os.listdir(density_dir) if p.startswith("sub-")])
# participants = ['sub-EBxLxQPx1957','sub-EBxLxTZx1956']
hemis = ["L", "R"]
projdist = '0' #Projection depth relative to wmgmi
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
        # for tract in ['MTmaskxLGN', 'MTmaskxPT', 'MTmaskxSTS1', 'MTmaskxPU', 'MTmaskxFEF', 'MTmaskxhIP', 'MTmaskxV1']:
        print(f"   🧩 Hemisphere: {hemi}")
        hemi_fs = "lh" if hemi == "L" else "rh"
        subj_dir = op.join(density_dir, participant, 'pyAFQ33_wb_red')
        subj_densities = []
        subj_present = []

        # Loop through *tracts in fixed order*
        for tract in tract_order:
            # Find file matching this tract and hemisphere
            matches = [f for f in os.listdir(subj_dir) if f"{tract}" in f and f"hemi-{hemi_fs}" in f and "fsaverage" in f and f.endswith(f"fsprojdensity{projdist}mm2.mgh")]

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
verbose = True

hemis = ["L", "R"]
contrast = contrast_order[0]   # e.g. "motionXstationary"

# get n_subj, n_tracts
n_subj = len(participants)
n_tracts = len(tract_order)
rs   = np.full((n_subj, n_tracts, len(hemis)), np.nan)
mses = np.full((n_subj,  n_tracts, len(hemis)), np.nan) #
rsquared = np.full(( n_subj, n_tracts, len(hemis)), np.nan) #goodness of fit
reliability = np.full((n_subj, n_tracts, len(hemis)), np.nan)
r_all = np.full((n_subj, len(hemis)), np.nan)
rnd_run_idx = np.full((n_subj, 3, len(hemis)), np.nan)
trained_coefs = np.zeros((n_tracts, n_subj, n_tracts, len(hemis)))  # scalar summary per tract/run
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
    predicted = np.full((n_subj, n_masked, n_tracts), np.nan)  # predicted maps per run
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

    for t in range(n_tracts):
        # -------------------------
        # main loop
        # -------------------------
        for test_idx in range(n_subj):
            #Prepare subject's Y = functional maps
            # get this subject's run maps of the chosen contrast
            
            # training participants indices
            train_idx = [i for i in range(n_subj) if i != test_idx]   

            if verbose:
                print(f" Left out participant {participants[test_idx]}")
            
            X_train = np.vstack([norm_density_data[hemi][i][t].reshape(-1, 1) for i in train_idx])   # (n_train*n_masked, n_tracts)
            y_train = np.hstack([C_mean[i,:] for i in train_idx])  # (n_train*n_masked,)


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
            linreg = LinearRegression()
            linreg.fit(X_train, y_train)

            trained_coefs[:, test_idx, t, h] = linreg.coef_.copy() #[0:n_tracts-1]
            # ridge.intercept_

            X_test = norm_density_data[hemi][test_idx][t].reshape(-1, 1) 
            y_pred_std = linreg.predict(X_test)
            # y_pred = (y_pred_std*np.std(C[test_idx,:]) + np.mean(C[test_idx,:])).ravel()

            predicted[test_idx, :, t] = y_pred_std
            reliability[test_idx, t, h] = vertex_bootstrap_reliability(all_C[test_idx,:,:])

            # Evaluate this test_participant if verbose
            y_participant_true = np.squeeze(C_mean[test_idx, :])
            r_participant, p_participant = pearsonr(np.squeeze(y_participant_true), y_pred_std)
            rs[test_idx,t,h] = r_participant
            rsquared[test_idx,t,h] = r2_score(y_participant_true, y_pred_std)
            mse_participant = mean_squared_error(np.squeeze(y_participant_true), y_pred_std)
            mses[test_idx,t,h] = mse_participant
            print(f"Participant r:{r_participant:.4f}, MSE={mse_participant:.4e}, p={p_participant:.4e}")


    #------------------
    #Performance metrics
    #------------------
    #overall correlation across all participants (concatenated)
    predicted_full = np.full((n_subj, n_vertices, n_tracts), np.nan)
    predicted_full[:, wang_hmt_vertices, :] = predicted

    true_full = np.full((n_subj, n_vertices), np.nan)
    true_full[:, wang_hmt_vertices] = C_mean

    # Coefficients already organized: (n_subj, n_tracts)
    coefs_arr = trained_coefs.copy()

    # Mean coefficient per tract
    mean_coefs = np.mean(coefs_arr, axis=0)

    print("\nDone.")

    predicted_maps[hemi] = predicted
    print(f"\nFinished hemisphere {hemi}")

#Plots and Figures
import sys
current_dir = op.dirname(op.abspath(__file__))
sys.path.append(current_dir)
from plots_utils_20_1 import pearsonr_plot, mse_plot

savefilename = f"wb_pearson_barplot_loso_separate_tracts_{projdist}mm.png"
pearsonr_plot(rs, reliability, participants, tract_order, bids_path, savefilename)

savefilename = f"wb_mse_barplot_loso_separate_tracts_{projdist}mm.png"
mse_plot(mses, participants, tract_order, bids_path, savefilename)