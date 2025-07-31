
import os
import os.path as op
from dipy.io.streamline import load_tractogram, save_tractogram

data_path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/afq/gpu-afq_MT-V1_nseeds20_0mm_nowm_dist3/sub-EBxGxEYx1965/'

trx_data = load_tractogram(op.join(data_path,'bundles', 'sub-EBxGxEYx1965_ses-04_acq-HCPdir99_desc-V1xMTR_tractography.trx'), op.join(data_path,'tractography','sub-EBxGxEYx1965_ses-04_acq-HCPdir99_desc-stop_mask.nii.gz'))

save_tractogram(trx_data, op.join(data_path,'bundles', 'sub-EBxGxEYx1965_ses-04_acq-HCPdir99_desc-V1xMTR_tractography.trk'))

#On bash:
#dipy_convert_tractogram your_file.trx --out_tractogram converted_tractogram.trk