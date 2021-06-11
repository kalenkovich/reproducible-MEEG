from pathlib import Path
import os
from zipfile import ZipFile, Path as ZipPath
import requests
from urllib.parse import urljoin

import mne


# Helper functions
def download_file_from_url(url, save_to):
    response = requests.get(url)
    # Raise an error if there was a problem
    response.raise_for_status()

    with open(save_to, 'wb') as file:
        file.write(response.content)


def filter_bids_dir(bids_root, subject=None, session=None, data_type=None):
    """
    Returns a list of paths to folders/directories in bids_root filtered by subject, session, and/or data_type
    :param bids_root: Path object (can be zipfile.Path as well as pathlib.Path)
    :param subject: subject id (e.g., '01') or None
    :param session: session type ('meg', 'mri') or None
    :param data_type: data type ('anat', 'beh', etc.) or None
    :return: list of objects of the same type as bids_root
    """
    # The archive structure is "sub-<xx>/ses-<yyy>/<data_type>/...".
    # "ses-meg" in addition to "meg" and "beh" contains a json file which will be skipped if data_type is set.

    # All subjects if subject is None else a single session folder
    subject_folders = [bids_root / f'sub-{subject}'] if subject else list(zip_path.iterdir())

    # All sessions if session is None else a single session folder
    session_folders = [session_folder
                       for subject_folder in subject_folders
                       for session_folder in
                       ([subject_folder / f'ses-{session}'] if session else list(subject_folder.iterdir()))]

    # All folders and files in the session folders if data_type is None else a single data type folder
    to_unpack = [item
                 for session_folder in session_folders
                 for item in ([session_folder / f'{data_type}'] if data_type else list(session_folder.iterdir()))                 ]

    return to_unpack


def unzip_bids_archive(archive_path, bids_root, subject=None, session=None, data_type=None):
    with ZipFile(archive_path,'r') as zip_file:
        # All the archives contain a single folder in the root called 'ds000117_R1.0.0/' which we will skip.
        zip_path = ZipPath(zip_file, 'ds000117_R1.0.0/')
        # Filter the contents by subject, session, and/or data type.
        to_unpack = filter_bids_dir(zip_path, subject=subject, session=session, data_type=data_type)
        # Recurse the list of folders/files and copy the files to bids_dir
        while to_unpack:
            item = to_unpack.pop()
            if item.is_file():
                relative_path = Path(str(item)).relative_to(Path(str(zip_path)))
                output_path = bids_root / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(item.read_bytes())
            else:
                to_unpack += list(item.iterdir())

# Configuration constants
L_FREQS = (None, 1)

# Folders
data_dir = Path(os.environ['reproduction-data'])
downloads_dir = data_dir / 'downloads'
bids_dir = data_dir / 'bids'
derivatives_dir = bids_dir / 'derivatives'
preprocessing_dir = derivatives_dir / '01_preprocessing'

# Templates
subject_json_template = (bids_dir / 'sub-{subject_number}' / 'ses-meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_proc-tsss_meg.json')
run_template = (bids_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_meg.fif')
events_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_eve.fif')
filtered_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                     'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_filteredHighPass{l_freq}.fif')

# Other file-related variables
openfmri_url_prefix = 'https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/'
openfmri_zip_files = [
    'ds000117_R1.0.0_sub01-04.zip',
    'ds000117_R1.0.0_sub05-08.zip',
    'ds000117_R1.0.0_sub09-12.zip',
    'ds000117_R1.0.0_sub13-16.zip',
]


rule all:
    input:
         events=expand(events_template,
                       subject_number=[f'{i:02d}' for i in range(1, 16 + 1)],
                       run_id=[f'{i:02d}' for i in range(1, 6 + 1)]),

         filtered=expand(filtered_template,
                         subject_number=[f'{i:02d}' for i in range(1, 16 + 1)],
                         run_id=[f'{i:02d}' for i in range(1, 6 + 1)],
                         l_freq=L_FREQS)


def linear_filter(run_path, output_path, l_freq):
    raw = mne.io.read_raw_fif(run_path, preload=True, verbose='error')
    raw.set_channel_types({'EEG061': 'eog',
                           'EEG062': 'eog',
                           'EEG063': 'ecg',
                           'EEG064': 'misc'})  # EEG064 free-floating el.
    raw.rename_channels({'EEG061': 'EOG061',
                         'EEG062': 'EOG062',
                         'EEG063': 'ECG063'})

    # Band-pass the data channels (MEG and EEG)
    raw.filter(
        l_freq=l_freq, h_freq=40, l_trans_bandwidth='auto', h_trans_bandwidth='auto',
        filter_length='auto', phase='zero', fir_window='hamming',
        fir_design='firwin')

    # High-pass EOG to get reasonable thresholds in autoreject
    picks_eog = mne.pick_types(raw.info, meg=False, eog=True)
    raw.filter(
        l_freq=1., h_freq=None, picks=picks_eog, l_trans_bandwidth='auto',
        filter_length='auto', phase='zero', fir_window='hann',
        fir_design='firwin')

    # Save
    raw.save(output_path)

rule apply_linear_filter:
    input:
        run = run_template
    output:
        filtered = filtered_template
    run:
        l_freq = None if wildcards.l_freq == 'None' else float(wildcards.l_freq)
        linear_filter(input.run, output.filtered, l_freq)


def extract_events(run_path, events_path):
    raw = mne.io.read_raw_fif(str(run_path))
    mask = 4096 + 256  # mask for excluding high order bits
    events = mne.find_events(raw, stim_channel='STI101',
                             consecutive='increasing', mask=mask,
                             mask_type='not_and', min_duration=0.003)
    mne.write_events(str(events_path), events)

rule extract_events:
    input:
        run = run_template
    output:
        events = events_template
    run:
        extract_events(input.run, output.events)

# Pseudo-rule to connect run files to the json file common to all runs
rule get_run:
    input:
        subject_json_template
    output:
        run_template

def find_archive_with_subject(wildcards):
    k = int(wildcards.subject_number)
    start = (k - 1) // 4  * 4 + 1
    end = start + 3
    return downloads_dir / f'ds000117_R1.0.0_sub{start:02d}-{end:02}.zip'

rule unpack_single_subject_meg_data:
    input:
        archive = find_archive_with_subject
    output:
        json = subject_json_template
    run:
        unzip_bids_archive(archive_path=input.archive, bids_root=bids_dir, subject=wildcards.subject_number,
                           session='meg')

rule download_single_file_from_openfmri:
    output:
        file_path = downloads_dir / '{filename}'
    run:
        url = urljoin(openfmri_url_prefix, wildcards.filename)
        download_file_from_url(url=url, save_to=output.file_path)
