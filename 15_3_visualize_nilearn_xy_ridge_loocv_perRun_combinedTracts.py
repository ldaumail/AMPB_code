from nilearn import plotting, surface
from nibabel.freesurfer import read_geometry, read_label
import numpy as np
import nibabel as nib
import os
import os.path as op
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
tract_order = ['MTxLGNxPU', 'MTxPTxSTS1', 'MTxFEF'] #'MTxPT', 'MTxPU' 'MTxFEF','MTxhIP','MTxV1'
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


#---------------------------
## Fit linear model to data
#---------------------------
alphas = np.logspace(-4, 4, 25)      # Ridge alpha grid (adjust if needed)
inner_cv = 5                         # internal CV to choose alpha (could use LeaveOneOut if many subjects)
verbose = True

hemis = ["L", "R"]
contrast = contrast_order[0]   # e.g. "motionXstationary"

# get n_subj, n_tracts
n_subj, n_tracts, _  = density_data["L"].shape
# rs   = np.full((3, n_subj, len(hemis)), np.nan)
# mses = np.full((3, n_subj, len(hemis)), np.nan)
# r_all = np.full(( n_subj, len(hemis)), np.nan)
rnd_run_idx = np.full((n_subj, 3), np.nan)
trained_coefs = np.zeros((3, n_tracts, n_subj, len(hemis)))  # scalar summary per tract/run
for h, hemi in enumerate(hemis):
        #h = 0
        #hemi = "L"
        #s = 0
        #'sub-EBxGxCCx1986'
    #Load ref surface (here fsaverage)
    hemi_fs = "lh" if hemi == "L" else "rh"
    # wm_surf = op.join(fs_path, 'fsaverage', 'surf', f"{hemi_fs}.inflated")    # FreeSurfer surface
    # coords, faces = read_geometry(wm_surf)
    infl_surf = op.join(fs_path, "fsaverage", "surf", f"{'lh' if hemi == 'L' else 'rh'}.inflated")


    _, _, n_vertices  = density_data[hemi].shape
    densities = density_data[hemi]        # (subj, tract, vertices)
    subj_contrasts = contrast_data[hemi]  # list: one dict per participant

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

    # Masking densities (subj, tract, masked_vertices)
    densities_masked = densities[:, :, wang_hmt_vertices]
    n_masked = densities_masked.shape[2] #Number of wang MT vertices

    # -------------------------
    # main loop: subject -> tract -> run-LOOCV
    # -------------------------
    # Predicted and coef storage
    predicted = np.full((n_subj, 3, n_masked), np.nan)  # predicted maps per run
    for s, participant in enumerate(participants):

        # get this subject's run maps for the chosen contrast
        subj_dict = subj_contrasts[s]

        C_full = subj_dict[contrast]
        # mask ROI
        C = np.squeeze(C_full[:, wang_hmt_vertices] )         # (n_runs, n_masked)
        #zscore the runs for a given participant
        n_runs = C.shape[0]
        zscored_C = np.array([(C[r_num,:] - np.mean(C[r_num,:]))/np.std(C[r_num,:]) for r_num in range(n_runs)])
        if verbose:
            print(f"\nSubject {participant}: {n_runs} runs (hemi {hemi})")

        #Prepare zscored density maps for each tract
        zscored_densities = np.full((n_tracts, n_masked), np.nan)
        for tract_idx in range(n_tracts):

            if verbose:
                print(f" Tract {tract_idx+1}/{n_tracts}")

            # anatomical vector for this subject and tract (length = n_masked)
            # anat_vec = densities_masked[s, tract_idx, :]  # shape (n_masked,)
            #Zscore the density data of the tract of this participant
            if np.std(densities_masked[s, tract_idx, :]) == 0:
                zscored_densities[tract_idx,:] = 0
            else:
                zscored_densities[tract_idx,:] = (densities_masked[s, tract_idx, :] - np.mean(densities_masked[s, tract_idx, :]))/np.std(densities_masked[s, tract_idx, :])
            # np.sum(np.isnan(densities_masked))

        X_train = zscored_densities.transpose(1,0)
        y_train = np.squeeze(zscored_C).reshape(-1, 1)  # (n_train*n_masked,)
        

        # ----------------------------
        # Functional MT ROI (binary surface map)
        # ----------------------------
        label_file = op.join(
            bids_path, 'analysis', 'ROIs', 'func_roi', 'functional_surf_roi',
            participant,
            f"{participant}_hemi-{hemi}_space-fsaverage_label-MT_mask.label"
        )

        func_mt_vertices = read_label(label_file)

        func_mt_roi = np.zeros(n_vertices, dtype=np.float32)
        func_mt_roi[func_mt_vertices] = 1


        # ----------------------------
        # Functional Contrast Map visualization
        # ----------------------------
        img_out_dir = op.join(bids_path, "analysis", "surface_pngs", participant)
        os.makedirs(img_out_dir, exist_ok=True)

        vmin, vmax = -5.0, 5.0

        for run_idx in range(zscored_C.shape[0]):

            # full surface vector
            surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
            surf_map[wang_hmt_vertices] = zscored_C[run_idx, :]

            out_png = op.join(
                img_out_dir,
                f"{participant}_hemi-{hemi}_run-{run_idx+1}_desc-motionXstationary_inflated.png"
            )

            display = plotting.plot_surf_stat_map(
                surf_mesh=infl_surf,
                stat_map=surf_map,
                hemi="left" if hemi == "L" else "right",
                view="lateral",
                cmap="plasma",
                colorbar=True,
                vmin=vmin,
                vmax=vmax,
                bg_map=None,
                threshold=None,
            )
            # ---- MT boundary overlay ----
            plotting.plot_surf_contours(
                surf_mesh=infl_surf,
                roi_map=func_mt_roi,
                levels=[1],
                colors=["green"],
                linewidths=2.0,
                figure=display.figure,
                axes=display.axes[0]
            )
            # ---- save + close ----
            display.savefig(out_png, dpi=300)
            plt.close(display.figure)

        # ----------------------------
        # Density Map visualization
        # ----------------------------
        img_out_dir = op.join(bids_path, "analysis", "surface_pngs", participant)
        os.makedirs(img_out_dir, exist_ok=True)
        for tract_idx in range(n_tracts):
            # X_train_full = np.full((n_vertices), np.nan)
            # X_train_full[wang_hmt_vertices] = X_train[:,tract_idx]
            # full surface vector
            surf_map = np.full((n_vertices,), np.nan, dtype=np.float32)
            surf_map[wang_hmt_vertices] =  X_train[:,tract_idx]

            out_png = op.join(
                img_out_dir,
                f"{participant}_hemi-{hemi}_tract-{tract_order[tract_idx]}_desc-density_inflated.png"
            )

            display = plotting.plot_surf_stat_map(
                surf_mesh=infl_surf,
                stat_map=surf_map,
                hemi="left" if hemi == "L" else "right",
                view="lateral",
                cmap="plasma",
                colorbar=True,
                vmin=vmin,
                vmax=vmax,
                bg_map=None,
                threshold=None,
            )
            # ---- MT boundary overlay ----
            plotting.plot_surf_contours(
                surf_mesh=infl_surf,
                roi_map=func_mt_roi,
                levels=[1],
                colors=["green"],
                linewidths=2.0,
                figure=display.figure,
                axes=display.axes[0]
            )
            # ---- save + close ----
            display.savefig(out_png, dpi=300)
            plt.close(display.figure)