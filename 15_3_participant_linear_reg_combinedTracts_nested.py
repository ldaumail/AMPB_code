
import numpy as np
from sklearn.linear_model import LinearRegression
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

# get n_subj, n_tracts
n_subj = len(participants)
n_tracts = len(tract_order)
rs   = np.full((n_subj, len(hemis)), np.nan)
rsquared = np.full(( n_subj, len(hemis)), np.nan) #goodness of fit
reliability = np.full((n_subj, len(hemis)), np.nan)
rnd_run_idx = np.full((n_subj, 3, len(hemis)), np.nan)
trained_coefs = np.zeros((n_tracts, n_subj, len(hemis)))  # scalar summary per tract/run
delta_mse = np.full((n_tracts, n_subj, len(hemis)), np.nan) #
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

    for s, participant in enumerate(participants):

        if verbose:
            print(f"Fitting participant {participant}")

        # X: density maps for this participant
        # shape: (n_vertices_masked, n_tracts)
        X = norm_density_data[hemi][s].T

        # y: functional contrast map
        # shape: (n_vertices_masked,)
        y = C_mean[s, :]

        # Fit regression
        linreg = LinearRegression()
        linreg.fit(X, y)

        # Store coefficients
        trained_coefs[:, s, h] = linreg.coef_.copy()

        # Predict (optional)
        y_pred = linreg.predict(X)
        predicted[s, :] = y_pred
        

        # Reliability (if needed)
        reliability[s, h] = vertex_bootstrap_reliability(all_C[s,:,:])

        # Optional evaluation
        r, p = pearsonr(y, y_pred)
        rs[s,h] = r
        mse_full = mean_squared_error(y, y_pred)
        print(f"r={r:.4f}, MSE={mse_full:.4e}, p={p:.4e}")


        for t in range(n_tracts):

            X_red = np.delete(X, t, axis=1)
            linearreg_red = LinearRegression()
            linearreg_red.fit(X_red, y)

            y_train_pred_red = linearreg_red.predict(X_red).ravel()
            mse_red = mean_squared_error(y, y_train_pred_red)
            delta_mse[t, s, h] = mse_red - mse_full
    predicted_maps[hemi] = predicted


#--------------------------------------------------------------
# FIGURES
#--------------------------------------------------------------
#--------------------------
#Save Plot of predicted contrast map for each subject
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
    # s =1
    # participant= 'sub-EBxGxEYx1965'
    # # ----------------------------
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
        img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "participants_linearreg", "surface_pngs", participant)
        os.makedirs(img_out_dir, exist_ok=True)

        vmin, vmax = -1.0, 1.0

        # Build full-surface vector 
        surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
        surf_map[wang_hmt_vertices] = predicted_maps[hemi][s, :]

        # Output filename (no run index)
        out_png = op.join(
            img_out_dir,
            f"{participant}_hemi-{hemi}_desc-pred-motionXstationary_inflated.png"
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

#========================================
#Save plot of average predicted contrast map for each group
#========================================

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
        img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "participants_linearreg",  "surface_pngs", "mean")
        os.makedirs(img_out_dir, exist_ok=True)

        vmin, vmax = -.5, 0.5
        # -------------------------------------------------
        # Compute average functional map across runs FIRST
        # -------------------------------------------------
        hemi_maps = predicted_maps[hemi]
        mean_vals = np.nanmean(hemi_maps[[group in p for p in participants],:], axis = 0)
        # mean_vals = np.nanmean(sub_mean_maps, axis=0)

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

#-------------------------------------
## Plot density map
#-------------------------------------
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

    # for s, participant in enumerate(participants):
    s =1
    participant= 'sub-EBxGxEYx1965'
    # # ----------------------------
    # Functional MT ROI (binary surface map)
    # ----------------------------
    label_file = op.join(bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi', participant,
        f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label")

    func_mt_vertices = read_label(label_file)

    func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
    func_mt_roi[func_mt_vertices] = 1

    # ----------------------------
    # Density Map visualization
    # ----------------------------
    img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "participants_linearreg", "surface_pngs", participant)
    os.makedirs(img_out_dir, exist_ok=True)

    vmin, vmax = -5.0, 5.0

    for t, tract in enumerate(tract_order):
        # Build full-surface vector 
        surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
        surf_map[wang_hmt_vertices] = norm_density_data[hemi][s][t]#predicted_maps[hemi][s, :]

        # Output filename (no run index)
        out_png = op.join(
            img_out_dir,
            f"{participant}_hemi-{hemi}_desc-{tract}_inflated.png"
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
        # plotting.plot_surf_contours(
        #     surf_mesh=infl_surf,
        #     roi_map=func_mt_roi,
        #     levels=[1],
        #     colors=["lightgray"],
        #     linewidths=2.0,
        #     figure=display.figure,
        #     axes=display.axes[0]
        # )

        # ---- save + close ----
        display.savefig(out_png, dpi=300)
        plt.close(display.figure)


#-----------------------------------------------
# Plot average true functional map
#-----------------------------------------------
from nilearn import plotting
from nibabel.freesurfer import read_label
import matplotlib.pyplot as plt

fs_path = op.join(bids_path, 'derivatives', 'freesurfer')
groups = ["EB", "NS"]
hemis = ["L", "R"]
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

    #---------------------
    #Prepare func maps
    #---------------------
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
        infl_surf = op.join(fs_path, "fsaverage", "surf", f"{'lh' if hemi == 'L' else 'rh'}.inflated")
        _, _, n_vertices  = density_data[hemi].shape
        # #-----------------------------
        # # Load func MT ROI
        # #-----------------------------
        # label_file = op.join(
        #     bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi',
        #     'group_averages',
        #     f"group-{group}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
        # )

        # func_mt_vertices = read_label(label_file)

        # func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
        # func_mt_roi[func_mt_vertices] = 1
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
        img_out_dir = op.join(bids_path, "analysis", "diff2func_model_fits", "participants_linearreg",  "surface_pngs", "mean")
        os.makedirs(img_out_dir, exist_ok=True)

        vmin, vmax = -0.5, 0.5
        # -------------------------------------------------
        # Compute average functional map across runs FIRST
        # -------------------------------------------------
        hemi_maps = C_mean #predicted_maps[hemi]
        mean_vals = np.nanmean(hemi_maps[[group in p for p in participants],:], axis = 0)
        # mean_vals = np.nanmean(sub_mean_maps, axis=0)

        # Build full-surface vector once
        surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
        surf_map[wang_hmt_vertices] = mean_vals

        # Output filename (no run index)
        out_png = op.join(
            img_out_dir,
            f"{group}-mean_hemi-{hemi}_desc-true-motionXstationary_mean_inflated.png"
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
        # plotting.plot_surf_contours(
        #     surf_mesh=infl_surf,
        #     roi_map=func_mt_roi,
        #     levels=[1],
        #     colors=["lightgray"],
        #     linewidths=2.0,
        #     figure=display.figure,
        #     axes=display.axes[0]
        # )

        # ---- save + close ----
        display.savefig(out_png, dpi=300)
        plt.close(display.figure)


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
# plt.savefig(op.join(saveDir, "participants_dMSE_linearreg_combined_tracts_nested.png"),
#             dpi=300, bbox_inches='tight')
plt.show()


#----------------------------
### Plot beta values of full model
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
std_df = (
    df.groupby(["Group", "Tract", "Hemisphere"])["Beta"]
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

for i, (ax, hemi) in enumerate(zip(axes, hemi_labels)):

    # Filter for hemisphere
    df_h = df[df["Hemisphere"] == hemi]
    std_h = std_df[std_df["Hemisphere"] == hemi]

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
    ) #legend=(i == 1)

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
    ax.set_title(f"{hemi}-Hemisphere β-coefficients (Mean ± SEM within EB / NS)",fontsize=20)
    ax.set_xlabel("Tract",fontsize=20,fontweight='bold')
    ax.set_ylabel("Beta",fontsize=20,fontweight='bold')
    ax.tick_params(axis="x", rotation=90, width=2)
    ax.set_xticks(np.arange(len(tract_order)))
    ax.set_xticklabels(tract_order, rotation=30, ha="right",fontsize=18,fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.spines['left'].set_linewidth(2) #axis thickness
    ax.spines['bottom'].set_linewidth(2) #axis thickness
    plt.setp(ax.get_yticklabels(),fontsize=18,fontweight='bold')
    # if i == 1:
    leg = ax.legend(title="Group", fontsize=14, frameon=True,loc='upper right')
    # This specifically bolds the labels and the title
    plt.setp(leg.get_texts(), fontweight='bold')
    plt.setp(leg.get_title(), fontweight='bold')
# axes[0].set_ylabel("Beta", fontsize=14)
# plt.legend(title="Group", labels=group_labels)

sns.despine()
plt.tight_layout()
saveDir = op.join(bids_path, 'analysis', 'plots')
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, "participants_betas_linearreg_combined_tracts.png"),
            dpi=300, bbox_inches='tight')
plt.show()



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
text_plotted = False  # Flag to ensure we only plot the label once
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
        # ADD THIS BLOCK:
        if not text_plotted:
            ax.text(
                x=x_center, 
                y=mean + 0.02,          # Slightly above the mean line
                s="Noise ceiling", 
                ha='center',            # Center horizontally
                va='bottom',            # Align bottom of text to the Y coordinate
                fontsize=14, 
                fontweight='bold', 
                color='gray'
            )
            text_plotted = True         # Toggle flag so it doesn't repeat

    # Formatting
    ax.set_ylim(-0.2, 1)
    ax.set_title(f"{hemi}-Hemisphere", fontsize=20)
    ax.set_xlabel("Group", fontsize=18, fontweight = 'bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticklabels(["EB", "NS"], fontsize=18, fontweight = 'bold')
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, fontweight = 'bold')
    plt.setp(ax.get_yticklabels(),fontsize=18,fontweight='bold')

axes[0].set_ylabel("Pearson's r", fontsize=18, fontweight = 'bold')
# axes[1].get_legend().remove()   # remove duplicated legend
sns.despine()
plt.tight_layout()

#Saving
saveDir = op.join(bids_path, "analysis", "plots")
os.makedirs(saveDir, exist_ok=True)
plt.savefig(op.join(saveDir, "pearson_linreg_participants_combined_tracts.png"), dpi=300, bbox_inches='tight')
plt.show()