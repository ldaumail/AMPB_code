import subprocess
import os
import os.path as op
from dipy.io.streamline import load_tractogram, save_tractogram
import nibabel as nib
import numpy as np

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = current_dir  # main_script.py is inside project/
sys.path.append(project_dir)
from utils.streamlines_utils import convert_streamlines, tckmap_to_image, load_mif



participant = 'sub-EBxGxEYx1965'
bids_path = op.join('/Users','ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
freesurfer_path = op.join(bids_path, 'derivatives', 'freesurfer')
tdi_path = op.join(bids_path, 'analysis', 'tdi_maps', 'mrtrix3_tdi_maps', participant)
os.makedirs(tdi_path, exist_ok=True)

pyAFQ_path = op.join('/Volumes','cos-lab-wpark78','LoicDaumail','ampb','derivatives','pyafq', 'gpu-afq_MT-STS1_nseeds20_0mm_nowm_dist3',participant)
bundle_path = op.join(pyAFQ_path,'bundles')

# First perpare files to run tckmap
# #1 convert T1.mgz to .nii
# fs_t1 = os.path.join(freesurfer_path, participant, 'mri', 'T1.mgz')
# op.exists(fs_t1)
# fs_t1_nii = os.path.join(tdi_path, 'T1.nii')
# # Run mri_convert
# freesurferCommand = f'mri_convert {fs_t1} {fs_t1_nii}'
# utils = op.join(bids_path, 'code', 'utils')
# sys.path.append(utils)
# os.system(f'bash {utils}/callFreesurferFunction.sh -s "{freesurferCommand}"')

#2 Convert .trx files to .tck format
sts1_mtL_trx_path = op.join(bundle_path, participant+'_ses-04_acq-HCPdir99_desc-STS1xMTL_tractography.trx')
sts1_mtR_trx_path = op.join(bundle_path, participant+'_ses-04_acq-HCPdir99_desc-STS1xMTR_tractography.trx')

trx_template = op.join(pyAFQ_path, participant+'_ses-04_acq-HCPdir99_b0ref.nii.gz')
sts1_mtL_tck_path = op.join(bundle_path, participant+'_ses-04_acq-HCPdir99_desc-STS1xMTL_tractography.tck')
sts1_mtR_tck_path = op.join(bundle_path, participant+'_ses-04_acq-HCPdir99_desc-STS1xMTR_tractography.tck')

convert_streamlines(sts1_mtL_trx_path, trx_template, sts1_mtL_tck_path)
convert_streamlines(sts1_mtR_trx_path, trx_template, sts1_mtR_tck_path)

# in_data = load_tractogram(sts1_mtL_tck_path, trx_template)
# print(in_data.is_bbox_in_vox_valid())
# streamlines =in_data.streamlines
# in_data = load_tractogram(sts1_mtL_trx_path, trx_template)
# print(in_data.is_bbox_in_vox_valid())
# streamlines =in_data.streamlines
# print(streamlines)
# in_data.space

# Run tckmap
sts1_mtL_tdi_map = op.join(tdi_path, participant+'_ses-04_acq-HCPdir99_desc-STS1xMTL_tdi_map.mif')
sts1_mtR_tdi_map = op.join(tdi_path, participant+'_ses-04_acq-HCPdir99_desc-STS1xMTR_tdi_map.mif')

tckmap_to_image(sts1_mtL_tck_path, sts1_mtL_tdi_map, template_img=trx_template,  contrast='tdi', vox_size=0.5)
tckmap_to_image(sts1_mtR_tck_path, sts1_mtR_tdi_map, template_img=trx_template,  contrast='tdi', vox_size=0.5)




# Load the .mif image
#First convert into .nii
# Convert .mif to .nii.gz using MRtrix command
sts1_mtL_tdi_nii = op.join(tdi_path, participant+'_ses-04_acq-HCPdir99_desc-STS1xMTL_tdi_map.nii')
sts1_mtR_tdi_nii = op.join(tdi_path, participant+'_ses-04_acq-HCPdir99_desc-STS1xMTR_tdi_map.nii')

subprocess.run(['mrconvert', sts1_mtL_tdi_map, sts1_mtL_tdi_nii], check=True)
subprocess.run(['mrconvert', sts1_mtR_tdi_map, sts1_mtR_tdi_nii], check=True)


# data = load_mif(sts1_mtL_tdi_map)

# import matplotlib.pyplot as plt
# plt.imshow(data[:, :, data.shape[2]//2].T, cmap="hot", origin="lower")
# plt.show()

# --------  visualize tracts ----------- #
from fury import actor, window
from fury.colormap import create_colormap

import AFQ.data.fetch as afd
from AFQ.viz.utils import PanelFigure
from dipy.tracking.streamline import transform_streamlines

trx_template_img = nib.load(trx_template)
sts1_mtL = load_tractogram(sts1_mtL_tck_path, trx_template)
# sts1_mtL.space
sts1_mtL_ref = transform_streamlines(sts1_mtL.streamlines,
                                np.linalg.inv(trx_template_img.affine))

#Visualizing bundles with principal direction coloring

def lines_as_tubes(sl, line_width, **kwargs):
    line_actor = actor.line(sl, **kwargs)
    line_actor.GetProperty().SetRenderLinesAsTubes(1)
    line_actor.GetProperty().SetLineWidth(line_width)
    return line_actor


sts1_mtL_actor = lines_as_tubes(sts1_mtL_ref, 8)
# sts1_mtR_actor = lines_as_tubes(sts1_mtR_t1w, 8)

#Slice anatomical image using slicer actors

def slice_volume(data, x=None, y=None, z=None):
    slicer_actors = []
    slicer_actor_z = actor.slicer(data)
    if z is not None:
        slicer_actor_z.display_extent(
            0, data.shape[0] - 1,
            0, data.shape[1] - 1,
            z, z)
        slicer_actors.append(slicer_actor_z)
    if y is not None:
        slicer_actor_y = slicer_actor_z.copy()
        slicer_actor_y.display_extent(
            0, data.shape[0] - 1,
            y, y,
            0, data.shape[2] - 1)
        slicer_actors.append(slicer_actor_y)
    if x is not None:
        slicer_actor_x = slicer_actor_z.copy()
        slicer_actor_x.display_extent(
            x, x,
            0, data.shape[1] - 1,
            0, data.shape[2] - 1)
        slicer_actors.append(slicer_actor_x)

    return slicer_actors

qsiprep_path = op.join(bids_path,'derivatives', 'qsiprep', participant, 'anat')
# read in the bundles, trasnform into RASMM coordinates and then into anatomical coordinates
t1w_img = nib.load(op.join(qsiprep_path,
                           participant+'_space-ACPC_desc-preproc_T1w.nii.gz'))
t1w = t1w_img.get_fdata()
slicers = slice_volume(t1w, x=t1w.shape[0] // 2, z=t1w.shape[-1] // 3)

# template_img = nib.load(trx_template)
# template_b0 = template_img.get_fdata()
# slicers = slice_volume(template_b0, x=template_b0.shape[0] // 2, z=template_b0.shape[-1] // 3)

#Add actors to 3D window scen object
scene = window.Scene()

scene.add(sts1_mtL_actor)
# scene.add(sts1_mtR_actor)

mtL_actor = actor.line(sts1_mtL_ref, colors=(1, 0, 0), linewidth=0.5)
#mtR_actor = actor.contour_from_roi(mtR_dat, color=(1, 0, 0), opacity=0.5)

# Show in interactive window
scene.add(mtL_actor)
# scene.add(mtR_actor)


for slicer in slicers:
    scene.add(slicer)

#Now visualize:
window.show(scene, size=(1200, 1200), reset_camera=False)







########### Visualize Bundles #############
img = nib.load(sts1_mtL_tdi_nii)
data = img.get_fdata()
data_norm = data / np.max(data)  # normalize to [0, 1]

# ---------------------------
# Load streamlines (convert to anatomical space)
# ---------------------------
t1_img = nib.load(fs_t1_nii)
tck = load_tractogram(tck_file, t1_img)
tck.to_rasmm()

streamlines_t1w = transform_streamlines(
    tck.streamlines,
    np.linalg.inv(t1_img.affine)
)

# ---------------------------
# Create scene
# ---------------------------
scene = window.Scene()

# Volume slicer (TDI map)
slicer_actor = actor.slicer(data_norm)
scene.add(slicer_actor)

# Streamline tubes
def lines_as_tubes(sl, line_width=3, color=(1, 0, 0)):
    line_actor = actor.line(sl, colors=color)
    line_actor.GetProperty().SetRenderLinesAsTubes(1)
    line_actor.GetProperty().SetLineWidth(line_width)
    return line_actor

stream_actor = lines_as_tubes(streamlines_t1w, line_width=4, color=(0, 1, 0))
scene.add(stream_actor)

# ---------------------------
# Show
# ---------------------------
window.show(scene)