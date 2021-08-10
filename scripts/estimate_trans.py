import json

import mne
from mne.coreg import fit_matched_points
from mne.io.constants import FIFF
import numpy as np
import nibabel as nib
from mne.transforms import apply_trans


def _extract_landmarks(dig):
    """Extract NAS, LPA, and RPA from raw.info['dig'].

    NB: copied verbatim from mne_bids.utils
    """
    coords = dict()
    landmarks = {d['ident']: d for d in dig
                 if d['kind'] == FIFF.FIFFV_POINT_CARDINAL}
    if landmarks:
        if FIFF.FIFFV_POINT_NASION in landmarks:
            coords['NAS'] = landmarks[FIFF.FIFFV_POINT_NASION]['r'].tolist()
        if FIFF.FIFFV_POINT_LPA in landmarks:
            coords['LPA'] = landmarks[FIFF.FIFFV_POINT_LPA]['r'].tolist()
        if FIFF.FIFFV_POINT_RPA in landmarks:
            coords['RPA'] = landmarks[FIFF.FIFFV_POINT_RPA]['r'].tolist()
    return coords


def _get_t1_sidecar_landmarks(bids_t1_sidecar_path):

    with open(bids_t1_sidecar_path, 'r', encoding='utf-8') as f:
        contents = json.load(f)
    mri_coords_dict = contents.get('AnatomicalLandmarkCoordinates', dict())

    # landmarks array: rows: [LPA, NAS, RPA]; columns: [x, y, z]
    mri_landmarks = np.full((3, 3), np.nan)
    for landmark_name, coords in mri_coords_dict.items():
        if landmark_name.upper() == 'LPA':
            mri_landmarks[0, :] = coords
        elif landmark_name.upper() == 'RPA':
            mri_landmarks[2, :] = coords
        elif (landmark_name.upper() == 'NAS' or
              landmark_name.lower() == 'nasion'):
            mri_landmarks[1, :] = coords
        else:
            continue

    return mri_landmarks


def _get_mri_landmarks(bids_t1_sidecar_path, bids_t1_path, freesurfer_t1_path):

    # Load t1 images from BIDS and Freesurfer
    bids_nifti = nib.load(str(bids_t1_path))
    bids_mgh = nib.MGHImage(bids_nifti.dataobj, bids_nifti.affine)
    fs_mgh = nib.load(str(freesurfer_t1_path))

    # Transformation matrices
    bids_vox2ras = bids_mgh.header.get_vox2ras()
    fs_ras2vox = fs_mgh.header.get_ras2vox()
    fs_vox2ras_tkr = fs_mgh.header.get_vox2ras_tkr()

    # Get MRI landmarks from the JSON sidecar
    mri_landmarks = _get_t1_sidecar_landmarks(bids_t1_sidecar_path)

    # Apply transformations
    mri_landmarks = apply_trans(bids_vox2ras, mri_landmarks)
    mri_landmarks = apply_trans(fs_ras2vox, mri_landmarks)
    mri_landmarks = apply_trans(fs_vox2ras_tkr, mri_landmarks)
    mri_landmarks = mri_landmarks * 1e-3

    return mri_landmarks


def _get_meg_landmarks(raw_path):
    raw = mne.io.read_raw_fif(raw_path, allow_maxshield=False)
    meg_coords_dict = _extract_landmarks(raw.info['dig'])
    return np.asarray((meg_coords_dict['LPA'],
                       meg_coords_dict['NAS'],
                       meg_coords_dict['RPA']))


def estimate_trans(bids_t1_path, bids_t1_sidecar_path, freesurfer_t1_path, bids_meg_path):
    """
    Estimates transformation matrix from MEG and freesurfer MRI landmark points.

    NB: The code is an adapted version of mne_bids.get_head_mri_trans which we couldn't use because it relied on the
    development version of mne-python

    returns : mne.transforms.Transform - The data transformation matrix from head to MRI coordinates.
    """
    mri_landmarks = _get_mri_landmarks(
        bids_t1_sidecar_path=bids_t1_sidecar_path,
        bids_t1_path=bids_t1_path,
        freesurfer_t1_path=freesurfer_t1_path)
    meg_landmarks = _get_meg_landmarks(bids_meg_path)

    # Given the two sets of points, fit the transform
    trans_fitted = fit_matched_points(src_pts=meg_landmarks,
                                      tgt_pts=mri_landmarks)
    return mne.transforms.Transform(fro='head', to='mri', trans=trans_fitted)
