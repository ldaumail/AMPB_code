#Loic Daumail
#Started on 11/10/2025
#Trains a linear regression to predict functional activation based on tract end point densitiess


import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import nibabel as nib
import os
import os.path as op

# ----------------------------
# Inputs preparation
# ----------------------------

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
density_dir = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps')
func_dir = op.join(bids_path, 'analysis', 'fMRI_data')
# -----------------------
#1 Generate endpoint densities array
#-------------------------

# ✅ Fixed tract order (keep consistent across subjects!)
tract_order = ['MTxLGN', 'MTxPT', 'MTxSTS1', 'MTxPU', 'MTxFEF','MTxhIP','MTxV1'] #
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
        subj_dir = op.join(density_dir, participant)
        subj_densities = []

        # Loop through *tracts in fixed order*
        for tract in tract_order:
            
            # Find file matching this tract and hemisphere
            matches = [f for f in os.listdir(subj_dir) if f"wang{tract}" in f and f"hemi-{hemi_fs}" in f and "fsaverage" in f and f.endswith("fsprojdensity0mm.mgh")]

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
# Generate Beta contrast array
#-------------------------

contrast_order = ["motionXstationary", "motionXsilent"] # 
# Initialize storage dictionary
contrast_data = {hemi: [] for hemi in hemis}

for participant in participants:
    if not participant.startswith("sub-"):
        continue
    print(f"\n🔹 Participant: {participant}")
    contrasts_dir = op.join(func_dir, participant, 'glm', 'contrasts')
    # Loop by hemisphere
    for hemi in hemis:
        print(f"   🧩 Hemisphere: {hemi}")
        hemi_fs = "lh" if hemi == "L" else "rh"
        subj_contrasts = []

        for contrast in contrast_order:
            # Find file matching this tract and hemisphere
            if "EB" in participant:
                matches = [f for f in os.listdir(contrasts_dir) if f"hemi-{hemi}" in f and "fsaverage" in f and "ptlocal" in f and contrast in f and "tstat" in f]
            elif "NS" in participant: 
                matches = [f for f in os.listdir(contrasts_dir) if f"hemi-{hemi}" in f and "fsaverage" in f and "mtlocal" in f and contrast in f and "tstat" in f]
            # Load the file
            contrast_file = op.join(contrasts_dir, matches[0])
            img = nib.load(contrast_file)
            data = img.get_fdata().astype(np.float32)
            subj_contrasts.append(data)

        # Stack into one array: shape (n_tracts, n_vertices)
        subj_contrasts = np.stack(subj_contrasts, axis=0)  # (7, n_vertices)
        contrast_data[hemi].append(subj_contrasts)
        # for i, arr in enumerate(contrast_data[hemi]):
        #     print(f"{hemi} element {i}: shape = {arr.shape}")

# Convert to numpy arrays
for hemi in hemis:
    contrast_data[hemi] = np.squeeze(np.stack(contrast_data[hemi], axis=0))  # (n_subjects, n_contrasts, n_vertices)
    print(f"✅ {hemi}-hemisphere shape: {contrast_data[hemi].shape}")

#### Analyze data

# ----------------------------
# Parameters
# ----------------------------
alphas = np.logspace(-4, 4, 25)      # Ridge alpha grid (adjust if needed)
inner_cv = 5                         # internal CV to choose alpha (could use LeaveOneOut if many subjects)
verbose = True

# ----------------------------
# Basic checks
# ----------------------------
hemi = "L"
densities = density_data[hemi]
contrasts = contrast_data[hemi][:, 0,:]

assert densities.ndim == 3, "densities must be (n_subj, n_tracts, n_vertices)"
assert contrasts.ndim == 2, "contrasts must be (n_subj, n_vertices)"
n_subj, n_tracts, n_vertices  = densities.shape
assert contrasts.shape[0] == n_subj and contrasts.shape[1] == n_vertices

#---------------------
# Load MT ROI vertices
#---------------------
wang_hmt_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'brain_atlases','Wang_2015','hmtplus',  f"hemi-{hemi}_space-fsaverage_label-hMT_desc-wang.mgh")
surf_roi = nib.load(wang_hmt_path).get_fdata().squeeze()
# Get vertex indices where the ROI is nonzero (or above a threshold)
wang_hmt_vertices = np.where(surf_roi > 0)[0]
print(f"{len(wang_hmt_vertices)} vertices in ROI")

densities_masked = densities[:, :, wang_hmt_vertices]   # (n_subj, n_tracts, n_masked_vertices)
# for i in range(n_subj):
#     for t in range(n_tracts):
#         x = densities_masked[i, t, :]
#         std = np.nanstd(x)
#         if std == 0 and np.nanmean(x) == 0: #std is 0 if all vertices averaged are 0
#             densities_masked[i, t, :] = 0   # or keep original
#         else:
#             densities_masked[i, t, :] = (x - np.nanmean(x)) / std
        
        # nan_vals = np.isnan(densities_masked)
        # nan_count = nan_vals.sum()
        # coords = np.argwhere(nan_vals)

#-------------------------------------
#Convert t-stat map to a z-score map
#-------------------------------------
contrasts_masked  = contrasts[:,wang_hmt_vertices]     # (n_subj, n_masked_vertices)
# for i in range(n_subj):
#     contrasts_masked[i,:] = (contrasts_masked[i,:] - contrasts_masked[i,:].mean()) / contrasts_masked[i,:].std()
# nan_vals = np.isnan(contrasts_masked)
# nan_count = nan_vals.sum()
# coords = np.argwhere(nan_vals)

n_masked = densities_masked.shape[2]

# Storage
predicted = np.zeros_like(densities_masked)          # shape (n_subj, n_tracts, n_masked)
trained_coefs = np.zeros((n_subj, n_tracts))         # coef[test_subject, tract]

# ---- Train tract-specific models ----
for tract_idx in range(n_tracts):
# ---- Outer LOOCV across subjects ----
    for test_idx in range(n_subj):
        if verbose:
            print(f"\nLOOCV fold: leaving out subject {test_idx+1}/{n_subj}")

        train_idx = [i for i in range(n_subj) if i != test_idx]

        # ---- Build training matrix ----
        X_train = np.vstack([densities_masked[s,tract_idx].T for s in train_idx])   # (n_train*n_masked, n_tracts)
        y_train = np.hstack([contrasts_masked[s]    for s in train_idx])  # (n_train*n_masked,)

        # Standardize (per tract)
        scalerX = StandardScaler().fit(X_train.reshape(-1, 1))
        scaly   = StandardScaler().fit(y_train.reshape(-1, 1))

        Xtr = scalerX.transform(X_train.reshape(-1, 1))
        ytr = scaly.transform(y_train.reshape(-1, 1)).ravel()

        # Ridge CV for this tract
        ridge = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error", cv=inner_cv)
        ridge.fit(Xtr, ytr)

        # SAVE COEFFICIENT: 1 number per tract
        trained_coefs[test_idx, tract_idx] = ridge.coef_[0]

        # ---- Predict left-out subject ----
        X_test = densities_masked[test_idx, tract_idx, :].reshape(-1, 1)
        X_test_s = scalerX.transform(X_test)

        y_pred_std = ridge.predict(X_test_s)
        y_pred     = scaly.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()

        predicted[test_idx, tract_idx, :] = y_pred

        # ---- Evaluate ----
        if verbose:
            y_true = contrasts_masked[test_idx]
            r, p = pearsonr(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            print(f"  Tract {tract_idx}: r={r:.4f}, MSE={mse:.4e}, p={p:.4e}")

# ------------------------------------
# Collect results
# ------------------------------------
# Fill full vertex space
predicted_full = np.full((n_subj, n_tracts, n_vertices), np.nan)
predicted_full[:, :, wang_hmt_vertices] = predicted

true_full = np.full((n_subj, n_vertices), np.nan)
true_full[:, wang_hmt_vertices] = contrasts_masked

# Coefficients already organized: (n_subj, n_tracts)
coefs_arr = trained_coefs.copy()

# Mean coefficient per tract
mean_coefs = np.mean(coefs_arr, axis=0)

print("\nDone. Summary:")

# Store tract-wise Pearson r and MSE:
# rs[s, t] = correlation between predicted[s, t, :] and true[s]
rs   = np.full((n_subj, n_tracts), np.nan)
mses = np.full((n_subj, n_tracts), np.nan)

for s in range(n_subj):

    y_true = contrasts_masked[s]  # shape (n_masked,)

    for t in range(n_tracts):

        y_pred = predicted[s, t, :]  # shape (n_masked,)

        # Compute performance
        r, _ = pearsonr(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        rs[s, t] = r
        mses[s, t] = mse




# Optionally save predicted maps per subject using a reference image
out_dir = op.join(bids_path, 'analysis', 'ridgecv_predicted_maps')
os.makedirs(out_dir, exist_ok=True)
ref_img_for_save_path = wang_hmt_path 
ref_img_for_save = nib.load(ref_img_for_save_path)
for s in range(n_subj):
    participant = participants[s]
    os.makedirs(op.join(out_dir, participant), exist_ok=True)

    if "EB" in participant:
        task = "ptlocal"
    elif "NS" in participant: 
        task = "mtlocal"
    true_map_path = os.path.join(out_dir, participant, f"{participant}_task-{task}_hemi-{hemi}_space-fsaverage_label-{tract}_desc-{contrast_order[0]}_tstat_wangmask.mgz")
    true_map = true_full[s, :].reshape((1, 1, n_vertices)).astype(np.float32)
    nib.save(nib.MGHImage(true_map, ref_img_for_save.affine, ref_img_for_save.header), true_map_path)

    for t, tract in enumerate(tract_order):

        outpath = os.path.join(out_dir, participant, f"{participant}_hemi-{hemi}_label-{tract}_desc-predicted_contrast.mgz")
        # If ref_img is MGH and data must be shape (1,1,n_vertices) or so, adapt:
        data_to_save = predicted_full[s,t, :].reshape((1, 1, n_vertices)).astype(np.float32)
        # For surface mgz you may want a shape matching original — adapt as required
        nib.save(nib.MGHImage(data_to_save, ref_img_for_save.affine, ref_img_for_save.header), outpath)



#--------------------------------------------------------------

#Plotting results

#1. Heat map
import matplotlib.pyplot as plt
import numpy as np

# rs has shape (n_subj, n_tracts)
# rows = subjects
# columns = tracts

plt.figure(figsize=(10, 6))
plt.imshow(rs, aspect='auto', interpolation='nearest')
plt.colorbar(label='Pearson r')

plt.xlabel("Tracts")
plt.ylabel("Participants")
plt.title("Correlation Between Predicted and True Maps\n Left hemisphere")

# Tract labels
plt.xticks(np.arange(n_tracts), [f"{t}" for t in tract_order], rotation=45)

# Subject labels (optional)
plt.yticks(np.arange(n_subj), [f"P{s}" for s in range(n_subj)])

plt.tight_layout()
saveDir = op.join(bids_path, 'analysis', 'plots')
plt.savefig(op.join(saveDir,f"hemi-{hemi}_pearsonrs_ridgecv_heatmap.png"), dpi=300, bbox_inches='tight')

plt.show()





