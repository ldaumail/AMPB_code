import os
import os.path as op
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.tracking.streamline import transform_streamlines
from dipy.tracking.utils import density_map
import nibabel as nib
import subprocess
import numpy as np

def streamline2dipy_density(tract_file, template_img, out_file, voxel_size=None):
    # Load streamlines
    sft = load_tractogram(tract_file, template_img, bbox_valid_check=False)

    # Match space to template (AC-PC RASmm)
    template = nib.load(template_img)
    affine = template.affine
    shape = template.shape[:3]
    # Compute density map
    density = density_map(sft.streamlines, affine, shape)

    # Save output
    nib.save(nib.Nifti1Image(density.astype(np.float32), affine), out_file)

def convert_streamlines(streamline_path, template_path, out_path):
    """
    Parameters
    ----------
    streamline_path: path to the streamline file you aim to convert 
    Ex:  
    data_path = '/Users/ldaumail3/Documents/research/ampb_mt_tractometry_analysis/ampb/derivatives/afq/gpu-afq_MT-V1_nseeds20_0mm_nowm_dist3/sub-EBxGxEYx1965/'
    op.join(data_path,'bundles', 'sub-EBxGxEYx1965_ses-04_acq-HCPdir99_desc-V1xMTR_tractography.trx')
    
    template_path: template you want to load the streamlines file with the correct spatial attribute
    Ex: op.join(data_path,'tractography','sub-EBxGxEYx1965_ses-04_acq-HCPdir99_desc-stop_mask.nii.gz'))
    
    out_path: streamline file in the format you want. 
    Ex: convert to .trk format: op.join(data_path,'bundles', 'sub-EBxGxEYx1965_ses-04_acq-HCPdir99_desc-V1xMTR_tractography.trk')
    """
    # streamline_path = op.join(bundle_path, participant+'_ses-04_acq-HCPdir99_desc-STS1xMTL_tractography.trx')
    # template_path = trx_template
    # out_path = op.join(bundle_path, participant+'_ses-04_acq-HCPdir99_desc-STS1xMTL_tractography.tck')


    in_data = load_tractogram(streamline_path, template_path)
    # in_data.space
    # in_data.is_bbox_in_vox_valid()
    # orig_streamlines = in_data.streamlines
    # reference_anatomy = nib.load(template_path)
    # new_streamlines = StatefulTractogram(orig_streamlines, reference_anatomy, Space.RASMM)
    template_img = nib.load(template_path)
    in_data_reg = transform_streamlines(in_data.streamlines,
    np.linalg.inv(template_img.affine))
    in_data_reg_tract = StatefulTractogram(in_data_reg, template_img, Space.VOX)
    # in_data_reg_tract.is_bbox_in_vox_valid()
    # in_data_reg_tract.space
    save_tractogram(in_data_reg_tract, out_path)

def tckmap_to_image(tractogram, tdi_map, template_img, contrast, vox_size=None):
    """
    Creates a Track Density Image (TDI) from a tractogram.
    
    Parameters
    ----------
    tractogram : str
        Path to the input tractogram (.tck) file.
    tdi_map : str
        Path for the output TDI map (.mif) file.
    template_img : str
        Path to the template image file (.mgz) to use for voxel space.
    contrast : str
        The contrast to use for the TDI map (e.g., 'tdi').
    vox_size : float, optional
        The voxel size of the output image. Defaults to None.
    """
    cmd = ['tckmap', tractogram, '-template', template_img, '-contrast', contrast, tdi_map]
    if vox_size:
        cmd += ['-vox', str(vox_size)]
    
    print(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def load_mif(path):
    """
    Loads super resolution TDI maps from an .mif file.
    
    This function correctly handles the file header and binary data
    to prevent 'I/O operation on closed file' errors.
    
    Parameters
    ----------
    path : str
        The path to the .mif file.

    Returns
    -------
    numpy.ndarray
        A 3D numpy array containing the image data.
    """
    with open(path, "rb") as f:
        # Read the header
        header = []
        while True:
            line = f.readline().decode("utf-8")
            if line.strip() == "END":
                break
            header.append(line)
        
        # At this point, the file cursor is at the start of the binary data
        # We can read the data immediately before the file closes
        
        # Parse header for dimensions and data type
        dim_str = [x.split(":")[1] for x in header if x.startswith("dim")][0]
        dim = tuple(map(int, dim_str.split(",")))

        # Define a map for data types
        dtype_map = {
            "Float32LE": "<f4", "Float32BE": ">f4",
            "UInt16LE": "<u2", "UInt16BE": ">u2",
            "UInt32LE": "<u4", "UInt32BE": ">u4"
        }
        datatype = [x.split(":")[1].strip() for x in header if x.startswith("datatype")][0]
        dtype = np.dtype(dtype_map[datatype])
        
        # Calculate the total number of elements to read
        num_elements = np.prod(dim)
        
        # Read binary data with a specific count while the file is still open
        data = np.fromfile(f, dtype=dtype, count=num_elements)

    # Now that the file is closed, we can safely reshape the data
    return data.reshape(dim)

#On bash:
#dipy_convert_tractogram your_file.trx --out_tractogram converted_tractogram.trk