#Perform contrasts on the beta data

import nibabel as nib
import os.path as op
import os
import pandas as pd
import numpy as np

# -----------------
# 1. Load event files and compute contrasts per hemisphere
# -----------------
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research',
                    'ampb_mt_tractometry_analysis', 'ampb')
func_dir = op.join(bids_path, 'analysis', 'fMRI_data')

# --- Define condition codes ---
conditions = {"motion": 1, "silent": 2, "stationary": 3}

# --- Define contrasts (positive - negative) ---
contrasts = {
    "motion_vs_stationary": ("motion", "stationary"),
    "motion_vs_silent": ("motion", "silent")
}

# --- Loop through participants ---
for participant in sorted(os.listdir(func_dir)):
    if not participant.startswith("sub-"):
        continue

    print(f"\n🔹 Processing {participant}")
    event_dir = op.join(func_dir, participant, "eventfiles")
    glm_dir = op.join(func_dir, participant, "glm")

    if not os.path.isdir(event_dir) or not os.path.isdir(glm_dir):
        print(f"⚠️ Skipping {participant} (missing eventfiles or glm folder)")
        continue

    # -----------------
    # Loop by hemisphere
    # -----------------
    for hemi in ["L", "R"]:
        print(f"   🧩 Hemisphere: {hemi}")

        # --- Load beta maps for this hemisphere ---
        beta_files = sorted(
            f for f in os.listdir(glm_dir)
            if f.endswith("_beta.mgz") and f"hemi-{hemi}" in f and "ptlocal" in f
        )

        if not beta_files:
            print(f"   ⚠️ No beta files found for hemi-{hemi}")
            continue

        betas = []
        for bf in beta_files:
            img = nib.load(op.join(glm_dir, bf))
            betas.append(img.get_fdata())
        betas = np.stack(betas, axis=-1)  # (X, Y, Z, n_betas)
        print(f"      📊 Loaded {betas.shape[-1]} beta maps for hemi-{hemi}")

        # --- Load corresponding event files for this hemisphere ---
        subject_runs = []
        for file in sorted(
            f for f in os.listdir(event_dir)
            if f"hemi-{hemi}" in f and "ptlocal" in f and f.endswith("_events.tsv")
        ):
            df = pd.read_csv(op.join(event_dir, file), sep="\t")
            run_conditions = df["trial_type"].map(conditions).fillna(0).astype(int).values
            subject_runs.append(run_conditions)

        if not subject_runs:
            print(f"   ⚠️ No event files found for hemi-{hemi}")
            continue

        # Flatten all condition codes across runs
        condition_codes = np.concatenate(subject_runs)
        print(f"      🧠 {len(condition_codes)} total conditions across runs")

        # --- Sanity check ---
        if betas.shape[-1] != len(condition_codes):
            print(f"   ⚠️ Mismatch: {betas.shape[-1]} betas but {len(condition_codes)} conditions.")
            print("      → Check GLM design or event file alignment.")
            continue

        # -----------------
        # 2. Compute contrasts for this hemisphere
        # -----------------
        for contrast_name, (pos_cond, neg_cond) in contrasts.items():
            pos_code = conditions[pos_cond]
            neg_code = conditions[neg_cond]

            pos_idx = np.where(condition_codes == pos_code)[0]
            neg_idx = np.where(condition_codes == neg_code)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                print(f"   ⚠️ Missing {pos_cond} or {neg_cond} trials — skipping {contrast_name}")
                continue

            # Mean beta per condition
            mean_pos = np.mean(betas[..., pos_idx], axis=-1)
            mean_neg = np.mean(betas[..., neg_idx], axis=-1)

            contrast_map = mean_pos - mean_neg

            # -----------------
            # 3. Save contrast map
            # -----------------
            contrast_img = nib.Nifti1Image(contrast_map, img.affine, img.header)
            out_dir = op.join(glm_dir, "contrasts")
            os.makedirs(out_dir, exist_ok=True)
            out_path = op.join(out_dir, f"{participant}_hemi-{hemi}_{contrast_name}.nii.gz")

            nib.save(contrast_img, out_path)
            print(f"      💾 Saved: {out_path}")

print("\n✅ All contrasts computed and saved per hemisphere.")

