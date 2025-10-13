import argparse
import os
from pathlib import Path

import nibabel as nib


def extract_voxel_data(nii_path, mask_path, output_dir, index_s, index_e):
    """
    Extract voxel data from NIfTI files across all time frames based on a brain mask.

    Args:
        nii_path (str): Path to the 4D NIfTI file.
        bvec_path (str): Path to the bvec file.
        bval_path (str): Path to the bval file.
        mask_path (str): Path to the mask NIfTI file.
        num_voxels (int): Number of voxels to sample.
    """
    # Load NIfTI files
    nii_img = nib.load(nii_path)
    mask_img = nib.load(mask_path)

    # Get data arrays
    nii_data = nii_img.get_fdata()
    mask_data = mask_img.get_fdata()


    new_data = nii_data[index_s:index_e, index_s:index_e, :, :]  # Shape: (width, height, channels, time)
    new_mask = mask_data[index_s:index_e, index_s:index_e, :]  # Shape: (width, height, channels)

    # Save extracted data
    data_nii_out_path = os.path.join(output_dir, f"voxels_{index_s}_{index_e}_data.nii")
    mask_nii_out_path = os.path.join(output_dir, f"voxels_{index_s}_{index_e}_mask.nii")

    # Save the new data to a NIfTI file
    new_data_img = nib.Nifti1Image(new_data, nii_img.affine)
    nib.save(new_data_img, data_nii_out_path)
    new_mask_img = nib.Nifti1Image(new_mask, mask_img.affine)
    nib.save(new_mask_img, mask_nii_out_path)

    print(f"Extraction complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a subset of volumes for testing purposes.')

    # Define command-line arguments
    parser.add_argument('--input-dir', type=Path, help='Path to the input folder', required=True)
    parser.add_argument('--data', type=Path, help='data file nmae', required=True)
    parser.add_argument('--mask', type=Path, help='mask file name', required=True)
    parser.add_argument('--output-dir', type=Path, help='Path to the output directory', required=True)
    parser.add_argument('--index-s', type=int, help='start index to extract', required=True)
    parser.add_argument('--index-e', type=int, help='end index to extract', required=True)

    # Parse the command-line arguments
    args = parser.parse_args()

    input_files = [os.path.join(args.input_dir, file_path) for file_path in [args.data, args.mask]]

    # Check if each file exists
    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    # print(int(args.index_e))
    extract_voxel_data(*input_files, args.output_dir, args.index_s, args.index_e)

"""
Run command: (run from tests folder)
====================================

s=50;e=70;python extract_voxels.py --input-dir '.' --mask '11001_mask.nii' --data '11001_drc144images.nii' --output-dir '.' --index-s $s --index-e $e
"""
