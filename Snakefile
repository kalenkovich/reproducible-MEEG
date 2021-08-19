from pathlib import Path
import os
import re
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


# Configuration constants
L_FREQS = (None, 1)
# [from the original script at scripts/processing/05-run_ica.py]
# SSS reduces the data rank and the noise levels, so let's include
# components based on a higher proportion of variance explained (0.999)
# than we would otherwise do for non-Maxwell-filtered raw data (0.98)
ICA_N_COMPONENTS = 0.999
RANDOM_STATE = 42

# Folders
data_dir = Path(os.environ['reproduction-data'])
downloads_dir = data_dir / 'downloads'
bids_dir = data_dir / 'bids'
derivatives_dir = bids_dir / 'derivatives'
preprocessing_dir = derivatives_dir / '01_preprocessing'

openneuro_maxfiltered_dir = derivatives_dir / 'meg_derivatives'

# Templates
run_template = (openneuro_maxfiltered_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_proc-sss_meg.fif')
events_template = (bids_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_events.tsv')
filtered_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                     'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_filteredHighPass{l_freq}.fif')
ica_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_filtered.fif')

# Other file-related variables
openneuro_url_prefix = 'https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.4/files/'


# Helper variables
subject_numbers = [f'{i:02d}' for i in range(1,16 + 1)]
run_ids = [f'{i:02d}' for i in range(1,6 + 1)]


# Rules and functions that execute them

rule all:
    input:
        events = expand(events_template, subject_number=subject_numbers, run_id=run_ids),
        filtered = expand(filtered_template, subject_number=subject_numbers, run_id=run_ids, l_freq=L_FREQS),
        icas = expand(ica_template, subject_number=subject_numbers)


def calculate_ica(run_paths, output_path):
    raw = mne.concatenate_raws([mne.io.read_raw_fif(run_path) for run_path in run_paths])
    ica = mne.preprocessing.ICA(method='fastica',random_state=RANDOM_STATE, n_components=ICA_N_COMPONENTS)
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
    ica.fit(raw, picks=picks, reject=dict(grad=4000e-13, mag=4e-12), decim=11)
    ica.save(output_path)


rule ica:
    input:
        runs = expand(filtered_template, run_id=run_ids, l_freq=1, allow_missing=True)
    output:
        ica = ica_template
    run:
        calculate_ica(input.runs, output.ica)


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


# We need to distinguish files from openneuro from files that we create here. We need maxfiltered data, so we will
# download some of the derivatives from openneuro as well. In order for snakemake to understand that it shouldn't try to
# download # files that are created by our rules, we need to add constraints on the files that *can* be downloaded. For
# now, these are:
# - events files in the `sub-**` folders
# - maxfiltered data in the derivatives/meg_derivatives

dir_separator = re.escape(str(Path('/')))
file_in_subject_folder = fr'sub-\d+{dir_separator}.*'
maxfiltered_file = fr'derivatives{dir_separator}meg_derivatives{dir_separator}.*'

openneuro_filepath_regex = fr'({file_in_subject_folder}|{maxfiltered_file})'


rule download_from_openneuro:
    output:
        file_path = bids_dir / '{openneuro_filepath}'
    wildcard_constraints:
        openneuro_filepath = openneuro_filepath_regex
    run:
        relative_path = Path(output.file_path).relative_to(bids_dir)
        # The file urls on openneuro look like the paths, just with ':' instead of '/'.
        # To prevent urljoin from interpreting the part before the first colon as a scheme name, we need to add './'
        # (see https://stackoverflow.com/q/55202875/)
        url = urljoin(openneuro_url_prefix, './' + ':'.join(relative_path.parts))
        download_file_from_url(url=url, save_to=output.file_path)
