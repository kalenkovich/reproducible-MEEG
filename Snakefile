import json
from distutils.version import LooseVersion
from pathlib import Path
import os
import re

import pandas as pd
import requests
from urllib.parse import urljoin
import numpy as np
import mne


# Helper functions
from autoreject import get_rejection_threshold
from matplotlib import pyplot as plt
from mne.minimum_norm import make_inverse_operator, write_inverse_operator, apply_inverse, read_inverse_operator
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from sklearn.model_selection import KFold

from scripts.estimate_trans import estimate_trans


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
REJECT_TMAX = 0.8  # duration we really care about
# Minimal distance for the forward model
MINDIST = 5
# Spacing of the source space used in the forward model
SOURCE_SPACE_SPACING = 'oct6'
CONDITIONS = ['scrambled', 'unfamiliar', 'famous', 'faces', 'contrast', 'faces_eq', 'scrambled_eq']


# Folders
data_dir = Path(os.environ['reproduction_data'])
downloads_dir = data_dir / 'downloads'
bids_dir = data_dir / 'bids'
derivatives_dir = bids_dir / 'derivatives'
preprocessing_dir = derivatives_dir / '01_preprocessing'
# TODO: rename both the variable and the directory later
processing_dir = derivatives_dir / '02_processing'
source_modeling_dir = derivatives_dir / '03_source_modeling'
plots_dir = derivatives_dir / '04_plots'
# This is the folder with *our* freesurfer outputs, not the one from openneuro
freesurfer_dir = derivatives_dir / 'freesurfer_lk'

openneuro_maxfiltered_dir = derivatives_dir / 'meg_derivatives'


# Templates
run_template = (openneuro_maxfiltered_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_proc-sss_meg.fif')
events_template = (bids_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_events.tsv')
filtered_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                     'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_filteredHighPass{l_freq}_meg.fif')
ica_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_ica.fif')
maxfilter_log_template = (openneuro_maxfiltered_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                          'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_proc-sss_log.txt')
bad_channels_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                         'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_bads.fif')
concatenated_raw_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                             'sub-{subject_number}_ses-meg_task-facerecognition_proc-sss_concatenated_meg.fif')
concatenated_events_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                                'sub-{subject_number}_ses-meg_task-facerecognition_proc-sss_concatenated-eve.fif')
epoched_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                         'sub-{subject_number}_ses-meg_task-facerecognition_epo.fif')
ecg_epochs_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                       'sub-{subject_number}_ses-meg_task-facerecognition_ecg_epo.fif')
eog_epochs_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                       'sub-{subject_number}_ses-meg_task-facerecognition_eog_epo.fif')
artifact_components_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                       'sub-{subject_number}_ses-meg_task-facerecognition_artifactComponents.npz')
epochs_cleaned_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                           'sub-{subject_number}_ses-meg_task-facerecognition_cleaned_epo.fif')
evoked_template = (processing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                   'sub-{subject_number}_ses-meg_task-facerecognition_ave.fif')
covariance_template = (processing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                       'sub-{subject_number}_ses-meg_task-facerecognition_cov.fif')
tfr_template = (processing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_{measure}-{condition}.fif')
group_average_evokeds_path = processing_dir / 'ses-meg' / 'meg' / 'ses-meg_task-facerecognition_grand-ave.fif'
bids_t1_sidecar_template = (bids_dir / 'sub-{subject_number}' / 'ses-mri' / 'anat' /
                            'sub-{subject_number}_ses-mri_acq-mprage_T1w.json')
bids_t1_template = bids_t1_sidecar_template.with_suffix('.nii.gz')
freesurfer_t1_template = freesurfer_dir / 'sub-{subject_number}' / 'ses-mri' / 'anat' / 'mri' / 'T1.mgz'
freesurfer_lh_reg_template = freesurfer_dir / 'sub-{subject_number}' / 'ses-mri' / 'anat' / 'surf' / 'lh.sphere.reg'
freesurfer_rh_reg_template = freesurfer_dir / 'sub-{subject_number}' / 'ses-mri' / 'anat' / 'surf' / 'rh.sphere.reg'
transformation_template = source_modeling_dir / 'sub-{subject_number}' / 'sub-{subject_number}-trans.fif'
bem_src_template = (freesurfer_dir / 'sub-{subject_number}' / 'ses-mri' / 'anat' / 'bem' /
                    f'sub-{{subject_number}}-{SOURCE_SPACE_SPACING}-src.fif')
bem_sol_template = (freesurfer_dir / 'sub-{subject_number}' / 'ses-mri' / 'anat' / 'bem' /
                    'sub-{subject_number}-5120-bem-sol.fif')
forward_model_template = (source_modeling_dir / 'sub-{subject_number}' /
                          f'sub-{{subject_number}}_spacing-{SOURCE_SPACE_SPACING}-fwd.fif')
inverse_model_template = (source_modeling_dir / 'sub-{subject_number}' /
                          f'sub-{{subject_number}}_spacing-{SOURCE_SPACE_SPACING}-inv.fif')
morph_matrix_template = source_modeling_dir / 'sub-{subject_number}' / 'sub-{subject_number}-morph.h5'
dspm_stc_template = (source_modeling_dir / 'sub-{subject_number}' /
                     'sub-{subject_number}_condition-{condition}_algorithm-dSPM-stc.h5')
dspm_stc_morphed_template = (source_modeling_dir / 'sub-{subject_number}' /
                             'sub-{subject_number}_condition-{condition}_algorithm-dSPM-stcMorphed.h5')
dspm_stc_averaged_template = source_modeling_dir / 'condition-{condition}_algorithm-dSPM.h5'
lcmv_stc_template = source_modeling_dir / 'sub-{subject_number}' / ('sub-{subject_number}_condition-contrast'
                                                                    '_algorithm-LCMV-{hemisphere}.stc')
lcmv_stc_morphed_template = (source_modeling_dir / 'sub-{subject_number}' /
                             'sub-{subject_number}_condition-contrast_algorithm-LCMV_morphed-{hemisphere}.stc')
lcmv_stc_averaged_template = source_modeling_dir / 'condition-contrast_algorithm-LCMV-{hemisphere}.stc'


wildcard_constraints:
    subject_number="\d+",
    run_id="\d+"


# Other file-related variables
openneuro_url_prefix = 'https://openneuro.org/crn/datasets/ds000117/snapshots/1.0.4/files/'


# Helper variables
subject_numbers = [f'{i:02d}' for i in range(1,16 + 1)]
run_ids = [f'{i:02d}' for i in range(1,6 + 1)]


# Experiment-specific variables
EVENTS_ID = {
    'face/famous/first': 5,
    'face/famous/immediate': 6,
    'face/famous/long': 7,
    'face/unfamiliar/first': 13,
    'face/unfamiliar/immediate': 14,
    'face/unfamiliar/long': 15,
    'scrambled/first': 17,
    'scrambled/immediate': 18,
    'scrambled/long': 19,
}
TMIN = -0.2
TMAX = 2.9  # min duration between onsets: (400 fix + 800 stim + 1700 ISI) ms
REJECT_TMAX = 0.8  # duration we really care about


# Hemispheres form LCMV source estimates that are saved into two files
HEMISPHERES = ['lh', 'rh']


# Rules and functions that execute them

rule all:
    input:
        events = expand(events_template, subject_number=subject_numbers, run_id=run_ids),
        filtered = expand(filtered_template, subject_number=subject_numbers, run_id=run_ids, l_freq=L_FREQS),
        icas = expand(ica_template, subject_number=subject_numbers),
        bad_channels = expand(bad_channels_template, subject_number=subject_numbers, run_id=run_ids),
        epoched = expand(epoched_template, subject_number=subject_numbers),
        ecg_epochs = expand(ecg_epochs_template, subject_number=subject_numbers),
        eog_epochs = expand(eog_epochs_template, subject_number=subject_numbers),
        artifact_components = expand(artifact_components_template, subject_number=subject_numbers),
        clean_epochs = expand(epochs_cleaned_template, subject_number=subject_numbers),
        evoked = expand(evoked_template, subject_number=subject_numbers),
        prestimulus_covariance = expand(covariance_template, subject_number=subject_numbers),
        tfr = expand(tfr_template, subject_number=subject_numbers, measure=('itc', 'power'),
                     condition=('face', 'scrambled')),
        group_average_evokeds = group_average_evokeds_path,
        # TODO: run for all subjects once we have run FreeSurfer on all of them
        transformation = expand(transformation_template, subject_number=subject_numbers),
        forward_model = expand(forward_model_template, subject_number=subject_numbers),
        inverse_model= expand(inverse_model_template, subject_number=subject_numbers),
        dspm_stc = expand(dspm_stc_template, subject_number=subject_numbers, condition=CONDITIONS),
        dspm_stc_morphed = expand(dspm_stc_morphed_template, subject_number=subject_numbers, condition=CONDITIONS),
        dspm_stc_morphed_average = expand(dspm_stc_averaged_template, condition='contrast')[0],
        lcmv_stc = expand(lcmv_stc_template, subject_number=subject_numbers, hemisphere=HEMISPHERES),
        lcmv_stc_morphed = expand(lcmv_stc_morphed_template, subject_number=subject_numbers, hemisphere=HEMISPHERES),
        lcmv_stc_morphed_average = expand(lcmv_stc_averaged_template, hemisphere=HEMISPHERES),
        erp_figure = plots_dir / 'erp.png',
        erp_properties = plots_dir / 'erp.json',
        dspm_figure = plots_dir /'dspm.png',
        manuscript_html = 'report.html'


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


# The bad channels are the same as the ones used during applying MaxFilterint the data by the dataset authors
rule extract_bad_channels:
    input:
        maxfilter_log = maxfilter_log_template
    output:
        bad_channels = bad_channels_template
    run:
        # code adapted from 03-maxwell_filtering.py
        bads = []
        with open(input.maxfilter_log, mode='r', encoding='utf-8') as fid:
            for line in fid:
                if line.startswith('Static bad channels'):
                    chs = line.split(':')[-1].split()
                    bads = ['MEG%04d' % int(ch) for ch in chs]
                    break

        with open(output.bad_channels, 'w', encoding='utf=8') as f:
            f.writelines('\n'.join(bads))


def _read_bads(bads_path):
    bads = list()
    with open(bads_path,encoding='utf-8') as f:
        for line in f:
            bads.append(line.strip())
    return bads


def _read_events(events_path, first_samp):
    events_df = pd.read_csv(events_path, delimiter='\t')
    events = events_df[['onset_sample', 'duration', 'trigger']].values
    # In FIF files and mne-python, the first sample is not counted as the first sample for reasons.
    # See https://mne.tools/dev/glossary.html#term-first_samp
    events[:, 0] += first_samp
    return events


def concatenate_runs(filtered_paths, bad_paths, events_paths, concatenated_raw_path, concatenated_events_path):
    # Load all runs, all events, set bad channels
    raw_list = list()
    events_list = list()
    for run_path, bads_path, events_path in zip(filtered_paths, bad_paths, events_paths):
        bads = _read_bads(bads_path)
        raw = mne.io.read_raw_fif(run_path, preload=True)
        events = _read_events(events_path, raw.first_samp)

        # Data in events.tsv BIDS files already accounts for the trigger-stimulus delay so we don't need to.
        # delay = int(round(0.0345 * raw.info['sfreq']))
        # events[:, 0] = events[:, 0] + delay

        events_list.append(events)

        raw.info['bads'] = bads
        raw.interpolate_bads()
        raw_list.append(raw)

    # Concatenate the runs
    raw, events = mne.concatenate_raws(raw_list, events_list=events_list)
    raw.set_eeg_reference(projection=True)
    del raw_list

    raw.save(concatenated_raw_path)
    mne.write_events(concatenated_events_path, events)


# Epoching and artifact searching is done on the non-highpassed data
EPOCHS_L_FREQ = None


rule concatenate_runs:
    input:
        filtered = expand(filtered_template, run_id=run_ids, l_freq=EPOCHS_L_FREQ, allow_missing=True),
        bads = expand(bad_channels_template, run_id=run_ids, allow_missing=True),
        events = expand(events_template, run_id=run_ids, allow_missing=True)
    output:
        raw = temp(concatenated_raw_template),
        events = temp(concatenated_events_template)
    run:
        concatenate_runs(filtered_paths=input.filtered, bad_paths=input.bads, events_paths=input.events,
                         concatenated_raw_path=output.raw, concatenated_events_path=output.events)


def make_epochs(raw_path, events_path, l_freq, epoched_path):
    raw = mne.io.read_raw(raw_path)
    events = mne.read_events(events_path)

    # `exclude` is empty so that the bad channels are not excluded
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True, eog=True, exclude=[])

    # Epoch the data
    baseline = (None, 0) if l_freq is None else None
    epochs = mne.Epochs(raw, events, event_id=EVENTS_ID, tmin=TMIN, tmax=TMAX, proj=True,
                        picks=picks, baseline=baseline, preload=False,
                        decim=5, reject=None, reject_tmax=REJECT_TMAX)
    epochs.save(epoched_path)


rule make_epochs:
    input:
        raw = concatenated_raw_template,
        events = concatenated_events_template
    output:
        epoched = epoched_template
    run:
        make_epochs(raw_path=input.raw, events_path=input.events, l_freq=EPOCHS_L_FREQ, epoched_path=output.epoched)


rule make_artifact_epochs:
    input:
        concatenated_raw = concatenated_raw_template
    output:
        ecg = ecg_epochs_template,
        eog = eog_epochs_template
    run:
        raw = mne.io.read_raw(input.concatenated_raw)

        ecg_epochs = create_ecg_epochs(raw, tmin=-.3, tmax=.3,preload=False)
        ecg_epochs.save(output.ecg)

        eog_epochs = create_eog_epochs(raw, tmin=-.5, tmax=.5,preload=False)
        eog_epochs.save(output.eog)


def select_artifact_components(ica_path, ecg_epochs_path, eog_epochs_path, artifact_components_path):
    ica = mne.preprocessing.read_ica(ica_path)

    # ECG
    ecg_epochs = mne.read_epochs(ecg_epochs_path)
    ecg_epochs.decimate(5)
    ecg_epochs.load_data()
    ecg_epochs.apply_baseline((None, None))
    ecg_inds, scores_ecg = ica.find_bads_ecg(ecg_epochs, method='ctps', threshold=0.8)

    # EOG
    eog_epochs = mne.read_epochs(eog_epochs_path)
    eog_epochs.decimate(5)
    eog_epochs.load_data()
    eog_epochs.apply_baseline((None, None))
    eog_inds, scores_eog = ica.find_bads_eog(eog_epochs)

    # save
    np.savez(artifact_components_path, ecg_inds=ecg_inds, scores_ecg=scores_ecg, eog_inds=eog_inds,
             scores_eog=scores_eog)


rule select_artifact_components:
    input:
        ica = ica_template,
        ecg_epochs = ecg_epochs_template,
        eog_epochs = eog_epochs_template
    output:
        artifact_components = artifact_components_template
    run:
        select_artifact_components(ica_path=input.ica, ecg_epochs_path=input.ecg_epochs,
            eog_epochs_path=input.eog_epochs, artifact_components_path=output.artifact_components)


def clean_epochs(ica_path, artifact_components_path, epochs_path, epochs_cleaned_path):
    # Load ica and bad components
    ica = mne.preprocessing.read_ica(ica_path)
    artifact_components = np.load(artifact_components_path)
    ecg_inds, eog_inds = artifact_components['ecg_inds'], artifact_components['eog_inds']

    # Set components to exclude
    n_max_ecg = 3  # use max 3 ECG components
    n_max_eog = 3  # use max 2 (sic) EOG components
    ica.exclude = list(ecg_inds[:n_max_ecg]) + list(eog_inds[:n_max_eog])

    # Remove artifact ICA components
    epochs = mne.read_epochs(epochs_path)
    epochs.load_data()
    ica.apply(epochs)

    # Use autoreject to remove bad epochs
    reject = get_rejection_threshold(epochs.copy().crop(None, REJECT_TMAX),
                                     random_state=RANDOM_STATE)
    epochs.drop_bad(reject=reject)

    # Save
    epochs.save(epochs_cleaned_path)


rule clean_epochs:
    input:
        ica = ica_template,
        artifact_components = artifact_components_template,
        epochs = epoched_template
    output:
        clean_epochs = epochs_cleaned_template
    run:
        clean_epochs(ica_path=input.ica, artifact_components_path=input.artifact_components,
                     epochs_path=input.epochs, epochs_cleaned_path=output.clean_epochs)


def make_evoked(clean_epochs_path, evoked_path):
    epochs = mne.read_epochs(clean_epochs_path, preload=True)

    # Evoked
    evoked_famous = epochs['face/famous'].average()
    evoked_famous.comment = 'famous'

    evoked_scrambled = epochs['scrambled'].average()
    evoked_scrambled.comment = 'scrambled'

    evoked_unfamiliar = epochs['face/unfamiliar'].average()
    evoked_unfamiliar.comment = 'unfamiliar'

    # Faces vs. scrambled
    contrast = mne.combine_evoked([evoked_famous, evoked_unfamiliar, evoked_scrambled],
                                   weights=[0.5, 0.5, -1.])
    contrast.comment = 'contrast'

    # All faces
    faces = mne.combine_evoked([evoked_famous, evoked_unfamiliar], 'nave')
    faces.comment = 'faces'

    # let's make trial-count-normalized ones for group statistics
    epochs_eq = epochs.copy().equalize_event_counts(['face', 'scrambled'])[0]
    evoked_faces_eq = epochs_eq['face'].average()
    evoked_scrambled_eq = epochs_eq['scrambled'].average()
    assert evoked_faces_eq.nave == evoked_scrambled_eq.nave
    evoked_faces_eq.comment = 'faces_eq'
    evoked_scrambled_eq.comment = 'scrambled_eq'

    # Save all to one file
    mne.evoked.write_evokeds(evoked_path, [evoked_famous, evoked_scrambled,
                                           evoked_unfamiliar, contrast, faces,
                                           evoked_faces_eq, evoked_scrambled_eq])


rule make_evoked:
    input:
        clean_epochs = epochs_cleaned_template
    output:
        evoked = evoked_template
    run:
        make_evoked(clean_epochs_path=input.clean_epochs, evoked_path=output.evoked)


rule calculate_prestimulus_covariance:
    input:
        clean_epochs = epochs_cleaned_template
    output:
        covariance = covariance_template
    run:
        epochs = mne.read_epochs(input.clean_epochs, preload=True)
        cv = KFold(3, random_state=RANDOM_STATE)  # make sure cv is deterministic
        cov = mne.compute_covariance(epochs, tmax=0, method='shrunk', cv=cv)
        cov.save(output.covariance)


rule calculate_tfr:
    input:
        clean_epochs = epochs_cleaned_template
    output:
        **{measure: expand(tfr_template, measure=(measure,), allow_missing=True)[0]
           for measure in ('power', 'itc')}
    run:
        condition = wildcards.condition  # faces/scrambled
        epochs_subset = mne.read_epochs(input.clean_epochs)[wildcards.condition]

        freqs = np.arange(6,40)
        n_cycles = freqs / 2.
        idx = [epochs_subset.ch_names.index('EEG065')]
        power, itc = mne.time_frequency.tfr_morlet(epochs_subset, freqs=freqs, return_itc=True, n_cycles=n_cycles,
                                                   picks=idx)

        power.save(output.power)
        itc.save(output.itc)


def group_average_evokeds(evoked_paths, group_average_path):
    # Load evokeds. One element - one subject.
    all_evokeds = [mne.read_evokeds(evoked_path) for evoked_path in evoked_paths]

    # Check for consistency of categories across subjects
    assert len({tuple(evoked.comment for evoked in evokeds) for evokeds in all_evokeds}) == 1

    # Combine evokeds from different subjects. One element - one category.
    combined_evokeds = [mne.combine_evoked(same_category_evokeds, 'equal')
                        for same_category_evokeds in zip(*all_evokeds)]

    # Save
    mne.evoked.write_evokeds(group_average_path, combined_evokeds)


rule group_average_evokeds:
    input:
        evokeds = expand(evoked_template, subject_number=subject_numbers)
    output:
        averaged_evokeds = group_average_evokeds_path
    run:
        group_average_evokeds(evoked_paths=input.evokeds, group_average_path=output.averaged_evokeds)


rule estimate_transformation_matrix:
    input:
        run01 = expand(run_template, run_id='01', allow_missing=True)[0],
        bids_t1 = bids_t1_template,
        bids_t1_sidecar = bids_t1_sidecar_template,
        freesurfer_t1 = freesurfer_t1_template
    output:
        trans = transformation_template
    run:
        trans = estimate_trans(bids_t1_path=input.bids_t1, bids_t1_sidecar_path=input.bids_t1_sidecar,
            freesurfer_t1_path=input.freesurfer_t1, bids_meg_path=input.run01)
        trans.save(output.trans)


# Mapping openneuro subject codes to the openfmri ones. See section "RELATIONSHIP OF SUBJECT NUMBERING RELATIVE TO OTHER
# VERSIONS OF DATASET" at https://openneuro.org/datasets/ds000117/versions/1.0.4
subject_code_map = {
    'sub002': 'sub-01',
    'sub003': 'sub-02',
    'sub004': 'sub-03',
    'sub011': 'sub-04',
    'sub006': 'sub-05',
    'sub007': 'sub-06',
    'sub008': 'sub-07',
    'sub009': 'sub-08',
    'sub010': 'sub-09',
    'sub012': 'sub-10',
    'sub013': 'sub-11',
    'sub014': 'sub-12',
    'sub015': 'sub-13',
    'sub017': 'sub-14',
    'sub018': 'sub-15',
    'sub019': 'sub-16'
}


def make_forward_model(evoked_path, trans_path, src_path, bem_path, forward_model_path):
    info = mne.io.read_info(evoked_path)
    # Because we use a 1-layer BEM, we do MEG only
    fwd = mne.make_forward_solution(info, trans_path, src_path, bem_path,
                                    meg=True, eeg=False, mindist=MINDIST)

    # We ran FreeSurfer on the OpenfMRI version of the data which has different subject codes than openneuro does. Here,
    # we change the codes to the openneuro codes for consistency with all the other files.
    for src_ in fwd['src']:
        src_['subject_his_id'] = subject_code_map[src_['subject_his_id']]

    mne.write_forward_solution(forward_model_path, fwd, overwrite=True)


rule run_forward:
    input:
        evoked = evoked_template,
        transformation = transformation_template,
        src = bem_src_template,
        bem = bem_sol_template
    output:
        forward_model = forward_model_template
    run:
        make_forward_model(evoked_path=input.evoked, trans_path=input.transformation, src_path=input.src,
            bem_path=input.bem, forward_model_path=output.forward_model)


rule make_inverse_model:
    input:
        evoked = evoked_template,
        cov = covariance_template,
        forward_model = forward_model_template,
    output:
        inverse_model = inverse_model_template
    run:

        cov = mne.read_cov(input.cov)
        forward = mne.read_forward_solution(input.forward_model)

        # This will be an MEG-only inverse because the 3-layer BEMs are not
        # reliable, so our forward only has MEG channels.
        info = mne.read_evokeds(input.evoked)[0].info
        inverse_operator = make_inverse_operator(info, forward, cov, loose=0.2, depth=0.8)
        write_inverse_operator(output.inverse_model, inverse_operator)


rule apply_dspm:
    input:
        evoked = evoked_template,
        inverse_model = inverse_model_template
    output:
        stcs = expand(dspm_stc_template, condition=CONDITIONS, allow_missing=True)
    run:
        # Load
        evokeds = mne.read_evokeds(input.evoked, condition=CONDITIONS)
        inverse_operator = read_inverse_operator(input.inverse_model)

        # Apply inverse
        snr = 3.0
        lambda2 = 1.0 / snr ** 2

        for evoked, stc_path in zip(evokeds, output.stcs):
            stc = apply_inverse(evoked, inverse_operator, lambda2, "dSPM", pick_ori='vector')
            stc.save(stc_path)


SMOOTH = 10


rule compute_morph_matrix:
    input:
        random_stc = expand(dspm_stc_template, condition=CONDITIONS, allow_missing=True)[0],
        t1 = freesurfer_t1_template,
        lh_reg = freesurfer_lh_reg_template,
        rh_reg = freesurfer_rh_reg_template,
    output:
        morph_matrix = morph_matrix_template
    run:
        # mne expects freesurfer output folder to have the fressurfer layout, not the bids-like one which is two levels
        # deeper. Here, we trick mne by including the additional "ses-mri/anat/" folders into the subject "names".
        subject_from = str(Path(f'sub-{wildcards.subject_number}/ses-mri/anat'))
        subject_to = str(Path('fsaverage/ses-mri/anat'))
        # mne saves a morphing map to <freesurfer_dir>/morph-maps/<sub-to>-<sub-from>-morph.fif
        # Due to the additional folders in the subject names, this becomes
        # <sub-to>/ses-mri/anat-<sub-from>/ses-mri/anat-morph.fif
        # It still works but we need to create the folder fot this file.
        morph_path = (freesurfer_dir.joinpath('morph-maps').joinpath(f'{subject_to}-{subject_from}')
                      .with_name('anat-morph.fif'))
        morph_path.parent.mkdir(parents=True, exist_ok=True)

        stc = mne.read_source_estimate(input.random_stc)
        stc.subject = subject_from

        morph = mne.compute_source_morph(
            subject_from=subject_from,
            src=stc,
            subject_to=subject_to,
            subjects_dir=freesurfer_dir,
            smooth=SMOOTH)

        # Restore the original subject names
        morph.subject_from = f'sub-{wildcards.subject_number}'
        morph.subject_to = 'fsaverage'

        morph.save(output.morph_matrix)


rule morph_dspm:
    input:
        stc = dspm_stc_template,
        morph_matrix = morph_matrix_template
    output:
        stc_morphed = dspm_stc_morphed_template
    run:
        morph = mne.read_source_morph(input.morph_matrix)
        stc = mne.read_source_estimate(input.stc)
        morphed = morph.apply(stc)
        morphed.save(output.stc_morphed)


def group_average_stcs(stc_paths, output_path):
    """
    :param stc_paths: list of file paths of stcs or stems in case of files split into hemisphere-specific files
    """
    stcs = [mne.read_source_estimate(stc_path) for stc_path in stc_paths]
    data = np.average([s.data for s in stcs],axis=0)
    random_stc = stcs[0]
    StcClass = type(random_stc)
    stc = StcClass(data, random_stc.vertices, random_stc.tmin, random_stc.tstep, random_stc.subject)
    stc.save(output_path)


rule group_average_dspm_sources:
    input:
        morphed_contrasts = expand(dspm_stc_morphed_template, condition='contrast', subject_number=subject_numbers)
    output:
        averaged_sources = dspm_stc_averaged_template
    run:
        group_average_stcs(stc_paths=input.morphed_contrasts, output_path=output.averaged_sources)


def _get_stem(two_hemisphere_files):
    suffix = '-lh.stc'
    n_to_remove = len(suffix)
    assert two_hemisphere_files[0][-n_to_remove:] == suffix
    stem = two_hemisphere_files[0][:-n_to_remove]
    assert two_hemisphere_files[1][:-n_to_remove] == stem
    return stem


def run_lcmv(fname_epo, fname_ave, fname_cov, fname_fwd, fnames_output):
    """
    Runs mne.beamformer.make_lcmv and mne.beamformer.apply_lcmv to get the LCMV solution to the inverse problem.
    :param fname_epo: epochs
    :param fname_ave: evoked data
    :param fname_cov: covariance
    :param fname_fwd: forward model
    :param fnames_output: list of two paths where solutions for the left and right hemisphere respectivelye will be
     stored. See mne.SourceEstimate.save for details.
    :return: None
    """
    epochs = mne.read_epochs(fname_epo, preload=False)
    data_cov = mne.compute_covariance(
        epochs[['face', 'scrambled']], tmin=0.03, tmax=0.3, method='shrunk')
    evoked = mne.read_evokeds(fname_ave, condition='contrast')
    noise_cov = mne.read_cov(fname_cov)
    forward = mne.read_forward_solution(fname_fwd)
    forward = mne.convert_forward_solution(forward, surf_ori=True)
    beamformer = mne.beamformer.make_lcmv(
        evoked.info, forward=forward, noise_cov=noise_cov, data_cov=data_cov,
        pick_ori='max-power', weight_norm='unit-noise-gain', rank=None)
    stc = mne.beamformer.apply_lcmv(evoked, filters=beamformer, max_ori_out='signed')

    # Parse out the common stem of the output file paths and save
    stem = _get_stem(fnames_output)
    stc.save(stem)


rule apply_lcmv:
    input:
        epochs= epochs_cleaned_template,
        evoked= evoked_template,
        covariance= covariance_template,
        forward_model= forward_model_template
    output:
        stc = expand(lcmv_stc_template, hemisphere=HEMISPHERES, allow_missing=True)
    run:
        run_lcmv(fname_epo=input.epochs, fname_ave=input.evoked, fname_cov=input.covariance,
                 fname_fwd=input.forward_model, fnames_output=output.stc)


rule morph_lcmv:
    input:
        stcs = expand(lcmv_stc_template, hemisphere=HEMISPHERES, allow_missing=True),
        morph_matrix = morph_matrix_template
    output:
        stcs_morphed = expand(lcmv_stc_morphed_template, hemisphere=HEMISPHERES, allow_missing=True)
    run:
        morph = mne.read_source_morph(input.morph_matrix)
        stc = mne.read_source_estimate(_get_stem(input.stcs))
        morphed = morph.apply(stc)
        morphed.save(_get_stem(output.stcs_morphed))


def _get_stems(list_of_hemisphere_pairs):
    lhs, rhs = list_of_hemisphere_pairs[::2], list_of_hemisphere_pairs[1::2]
    stems = [_get_stem([lh, rh]) for (lh, rh) in zip(lhs,rhs)]
    return stems


rule group_average_lcmv_sources:
    input:
        morphed_contrasts = expand(lcmv_stc_morphed_template, subject_number=subject_numbers, hemisphere=HEMISPHERES)
    output:
        averaged_sources = expand(lcmv_stc_averaged_template, hemisphere=HEMISPHERES)
    run:
        group_average_stcs(stc_paths=_get_stems(input.morphed_contrasts),
                           output_path=_get_stem(output.averaged_sources))


def _set_matplotlib_defaults():
    import matplotlib.pyplot as plt
    fontsize = 8
    params = {'axes.labelsize': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'axes.titlesize': fontsize + 2,
              'figure.max_open_warning': 200,
              'axes.spines.top': False,
              'axes.spines.right': False,
              'axes.grid': True,
              'lines.linewidth': 1,
              }
    import matplotlib
    if LooseVersion(matplotlib.__version__) >= '2':
        params['font.size'] = fontsize
    else:
        params['text.fontsize'] = fontsize
    plt.rcParams.update(params)


ERP_EEG_CHANNEL = 'EEG065'
ANNOT_KWARGS = dict(fontsize=12, fontweight='bold',
                    xycoords="axes fraction", ha='right', va='center')
TMAX = 2.9  # min duration between onsets: (400 fix + 800 stim + 1700 ISI) ms


def plot_erp(evokeds_path, png_path, properties_path):
    l_freq = EPOCHS_L_FREQ

    evokeds = mne.read_evokeds(evokeds_path)
    idx = evokeds[0].ch_names.index(ERP_EEG_CHANNEL)
    assert evokeds[1].ch_names[idx] == ERP_EEG_CHANNEL
    assert evokeds[2].ch_names[idx] == ERP_EEG_CHANNEL
    mapping = {'Famous': evokeds[0], 'Scrambled': evokeds[1],
               'Unfamiliar': evokeds[2]}
    colors =  {'Famous': 'blue', 'Scrambled': 'red',
               'Unfamiliar': 'green'}

    _set_matplotlib_defaults()

    fig, ax = plt.subplots(1, figsize=(3.3, 2.3))
    scale = 1e6
    times = evokeds[0].times * 1000
    for condition in ('Scrambled', 'Unfamiliar', 'Famous'):
        ax.plot(times, mapping[condition].data[idx] * scale,
                colors[condition], label=condition)
    ax.grid(True)
    ax.set(xlim=[-100, 1000 * TMAX], xlabel='Time (in ms after stimulus onset)',
           ylim=[-12.5, 5], ylabel=u'Potential difference (μV)')
    ax.axvline(800, ls='--', color='k')
    if l_freq == 1:
        ax.legend(loc='lower right')
    ax.annotate('A' if l_freq is None else 'B', (-0.2, 1), **ANNOT_KWARGS)
    fig.tight_layout(pad=0.5)
    # plt.show()

    fig.savefig(png_path)

    baseline = tuple(np.round(evokeds[0].baseline, 3))
    properties = dict(
        sensor=ERP_EEG_CHANNEL,
        baseline=baseline,
        baseline_units='s',
        colors=colors
    )
    with open(properties_path, 'w', encoding='utf-8') as f:
        # code from https://stackoverflow.com/a/12309296
        json.dump(properties, f, ensure_ascii=False, indent=4)


rule plot_erp:
    input:
        evokeds = rules.group_average_evokeds.output.averaged_evokeds
    output:
        png = plots_dir / 'erp.png',
        properties = plots_dir / 'erp.json'
    run:
        plot_erp(input.evokeds, output.png, output.properties)


def plot_dspm(dspm_path, png_path):
    stc : mne.SourceEstimate = mne.read_source_estimate(dspm_path, subject='fsaverage').magnitude()
    lims = (1, 3, 5)  # if l_freq is None else (0.5, 1.5, 2.5)
    stc.subject = str(Path('fsaverage/ses-mri/anat'))
    brain_dspm = stc.plot(
        views='ven',
        hemi='both', 
		backend='pyvista',
        brain_kwargs=dict(show=False),
        subjects_dir=freesurfer_dir,
        initial_time=0.17, time_unit='s', background='w', figure=1,
        clim=dict(kind='value',lims=lims), foreground='k', time_viewer=False)

    brain_dspm.save_image(png_path)
    brain_dspm.close()


rule plot_dspm:
    input:
        dspm = expand(rules.group_average_dspm_sources.output.averaged_sources, condition='contrast')[0]
    output:
        png = rules.all.input.dspm_figure
    run:
        plot_dspm(dspm_path=input.dspm, png_path=output.png)


rule make_report:
    input:
        rmd = 'report.Rmd',
        # Converting to posix-style paths is necessary on Windows when path become part of the code as below
        erp = Path(rules.plot_erp.output.png).as_posix(),
        erp_properties = Path(rules.plot_erp.output.properties).as_posix(),
    output:
        'report.html'
    shell:
        ('Rscript -e "rmarkdown::render(\'{input.rmd}\', output_file = \'{output}\', params = list('
         'erp = \'{input.erp}\','
         'erp_properties = \'{input.erp_properties}\''
         '))"')
