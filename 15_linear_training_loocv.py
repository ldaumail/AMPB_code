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
# Inputs you must set
# ----------------------------
# densities: np.array shape (n_subj, n_tracts, n_vertices)
# contrasts: np.array shape (n_subj, n_vertices)
# ref_img_for_save: nibabel image (for affine/header when saving .mgz/.nii)
# out_dir: path to save predicted maps and weights

# Example placeholders (replace with your actual arrays)
# densities = np.load("densities.npy")
# contrasts = np.load("contrasts.npy")
# ref_img_for_save = nib.load("/path/to/example_hemi-L_beta.mgz")  # to copy affine/header
# out_dir = "/path/to/save"

bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
density_dir = op.join(bids_path, 'analysis', 'tdi_maps', 'dipy_wmgmi_tdi_maps')

# -----------------------
#1 Generate densities array
#-------------------------

# ✅ Fixed tract order (keep consistent across subjects!)
tract_order = ['MTmaskxLGN', 'MTmaskxPT', 'MTmaskxSTS1', 'MTmaskxPU', 'MTmaskxFEF', 'MTmaskxhIP','MTmaskxV1']
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

        subj_dir = op.join(density_dir, participant)
        subj_densities = []

        # Loop through *tracts in fixed order*
        for tract in tract_order:
            # Find file matching this tract and hemisphere
            matches = [f for f in os.listdir(subj_dir) if tract in f and f"hemi-{hemi}" in f and f.endswith("fsprojdensity0mm.mgh")]

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

for i, arr in enumerate(density_data[hemi]):
    print(f"{hemi} element {i}: shape = {arr.shape}")
    
# Convert to numpy arrays

for hemi in hemis:
    density_data[hemi] = np.stack(density_data[hemi], axis=0)  # (n_subjects, n_tracts, n_vertices)
    print(f"✅ {hemi}-hemisphere shape: {density_data[hemi].shape}")

# ----------------------------
# Parameters
# ----------------------------
alphas = np.logspace(-4, 4, 25)      # Ridge alpha grid (adjust if needed)
inner_cv = 5                         # internal CV to choose alpha (could use LeaveOneOut if many subjects)
verbose = True

# ----------------------------
# Basic checks
# ----------------------------
assert densities.ndim == 3, "densities must be (n_subj, n_tracts, n_vertices)"
assert contrasts.ndim == 2, "contrasts must be (n_subj, n_vertices)"
n_subj, n_tracts, n_vertices = densities.shape
assert contrasts.shape[0] == n_subj and contrasts.shape[1] == n_vertices

# Optional: mask vertices (e.g., exclude vertices with zero density across all subjects)
mask = np.any(np.any(densities > 0, axis=1), axis=0)   # True for vertex with any density somewhere
masked_idx = np.where(mask)[0]
print(f"Using {len(masked_idx)} / {n_vertices} vertices after mask")

# We'll work only on masked vertices to save time
densities_masked = densities[:, :, masked_idx]   # (n_subj, n_tracts, n_masked_vertices)
contrasts_masked  = contrasts[:, masked_idx]     # (n_subj, n_masked_vertices)
n_masked = densities_masked.shape[2]

# Prepare storage
predicted = np.zeros_like(contrasts_masked)  # predicted contrast maps for each subject on masked vertices
trained_coefs = []                           # store ridge.coef_ for each fold (global model weights across tracts)

# Outer LOOCV across subjects
for test_idx in range(n_subj):
    if verbose:
        print(f"\nLOOCV fold: leaving out subject {test_idx+1}/{n_subj}")

    # Train indices
    train_idx = [i for i in range(n_subj) if i != test_idx]

    # Build training dataset by pooling vertices across training subjects:
    # For each training subject s: create matrix (n_masked_vertices x n_tracts) = densities_masked[s].T
    # Stack them on top of one another -> X_train shape (n_train * n_masked, n_tracts)
    X_train_list = [densities_masked[s].T for s in train_idx]   # each is (n_masked, n_tracts)
    X_train = np.vstack(X_train_list)                           # (n_train*n_masked, n_tracts)

    # Y_train: stack contrast maps for same training subjects (shape (n_train*n_masked,))
    y_train_list = [contrasts_masked[s] for s in train_idx]     # each is (n_masked,)
    y_train = np.hstack(y_train_list)                           # (n_train*n_masked,)

    # Standardize predictors and target using training data
    scalerX = StandardScaler().fit(X_train)
    scaly  = StandardScaler().fit(y_train.reshape(-1, 1))

    Xtr = scalerX.transform(X_train)
    ytr = scaly.transform(y_train.reshape(-1, 1)).ravel()

    # Fit Ridge with built-in CV to choose alpha
    ridge = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=inner_cv, store_cv_values=False)
    ridge.fit(Xtr, ytr)   # multi-sample pooled model

    if verbose:
        print(f"  -> chosen alpha: {ridge.alpha_}")

    # Save coefficients (they map standardized X to standardized y)
    trained_coefs.append(ridge.coef_.copy())  # shape (n_tracts,)

    # Predict for left-out subject: X_test shape (n_masked, n_tracts)
    X_test = densities_masked[test_idx].T               # (n_masked, n_tracts)
    X_test_s = scalerX.transform(X_test)

    y_pred_s_std = ridge.predict(X_test_s)              # predicted standardized y (n_masked,)
    y_pred_s = scaly.inverse_transform(y_pred_s_std.reshape(-1,1)).ravel()

    predicted[test_idx] = y_pred_s

    # Evaluate per-subject
    y_true = contrasts_masked[test_idx]
    r, p = pearsonr(y_true, y_pred_s)
    mse = mean_squared_error(y_true, y_pred_s)
    if verbose:
        print(f"  Subject {test_idx}: Pearson r = {r:.4f}, MSE = {mse:.4e}, p = {p:.4e}")

# ----------------------------
# Collect results and save
# ----------------------------
# bring predictions back to full vertex space (fill unmasked with zeros or nan)
predicted_full = np.full((n_subj, n_vertices), np.nan)
predicted_full[:, masked_idx] = predicted

# Optionally save predicted maps per subject using a reference image
if 'ref_img_for_save' in globals() and out_dir is not None:
    os.makedirs(out_dir, exist_ok=True)
    for s in range(n_subj):
        outpath = os.path.join(out_dir, f"sub-{s:03d}_predicted_contrast.mgz")
        # If ref_img is MGH and data must be shape (1,1,n_vertices) or so, adapt:
        data_to_save = predicted_full[s].reshape((1, 1, n_vertices)).astype(np.float32)
        # For surface mgz you may want a shape matching original — adapt as required
        nib.save(nib.MGHImage(data_to_save, ref_img_for_save.affine, ref_img_for_save.header), outpath)

# Aggregate coefficient map across folds (mean)
coefs_arr = np.vstack(trained_coefs)  # (n_subj, n_tracts)
mean_coefs = np.mean(coefs_arr, axis=0)  # average weight per tract

print("\nDone. Summary:")
rs = []
mses = []
for s in range(n_subj):
    r, _ = pearsonr(contrasts_masked[s], predicted[s])
    rs.append(r)
    mses.append(mean_squared_error(contrasts_masked[s], predicted[s]))
print(f"  Mean Pearson r across subjects = {np.nanmean(rs):.4f} ± {np.nanstd(rs):.4f}")
print(f"  Mean MSE across subjects = {np.mean(mses):.4e}")

# Print mean weight per tract
for t in range(n_tracts):
    print(f"Tract {t}: mean weight = {mean_coefs[t]:.4e}")
