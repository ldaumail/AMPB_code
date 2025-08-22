import os
import re

def rename_file(old_name):
    """
    Convert filename from:
    sub-XXX_ses-concat_space-T1w_desc-lhROI03SyN_mask.nii.gz
    to:
    sub-XXX_hemi-L_space-T1w_label-ROI_mask.nii.gz
    """
    subj_match = re.match(r"(sub-[^_]+)_.*_space-T1w", old_name)
    if not subj_match:
        raise ValueError(f"Could not parse subject from {old_name}")
    subj = subj_match.group(1)

    if "desc-lh" in old_name:
        hemi = "L"
        roi_part = re.search(r"desc-lh([^_]+)", old_name).group(1)
    elif "desc-rh" in old_name:
        hemi = "R"
        roi_part = re.search(r"desc-rh([^_]+)", old_name).group(1)
    else:
        raise ValueError(f"No lh/rh hemisphere found in {old_name}")

    # Clean ROI name (remove trailing digits and 'SyN')
    roi = re.sub(r"\d+|SyN", "", roi_part)

    new_name = f"{subj}_hemi-{hemi}_space-T1w_label-{roi}_mask.nii.gz"
    return new_name

def batch_rename_inplace(folder):
    """
    Rename all .nii.gz files inside `folder` in place.
    """
    for fname in os.listdir(folder):
        if fname.endswith(".nii.gz"):
            try:
                new_name = rename_file(fname)
                old_path = os.path.join(folder, fname)
                new_path = os.path.join(folder, new_name)

                os.rename(old_path, new_path)  # renames in place
                print(f"Renamed: {fname} → {new_name}")
            except Exception as e:
                print(f"Skipping {fname}: {e}")

# ---- Example usage ----
import os.path as op
participants=['sub-EBxLxTZx1956', 'sub-NSxGxBAx1970', 'sub-EBxGxCCx1986', 'sub-EBxLxHHx1949', 'sub-EBxGxZAx1990', 
'sub-NSxLxATx1954', 'sub-EBxGxEYx1965', 'sub-NSxGxHNx1952', 'sub-NSxLxYKx1964', 
'sub-EBxLxQPx1957', 'sub-EBxLxTZx1956', 'sub-NSxGxIFx1991', 
'sub-EBxGxPEx1959', 'sub-NSxGxHKx1965', 'sub-NSxLxQUx1953', 'sub-NSxLxPQx1973']

for participant in participants:
    folder = op.join("/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/analysis/julich_space-ACPC_rois",participant, "ses-concat", "anat")
    batch_rename_inplace(folder)