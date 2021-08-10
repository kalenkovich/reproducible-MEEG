from pathlib import Path
import os
import re
from zipfile import ZipFile, Path as ZipPath

import pandas as pd
import requests
from urllib.parse import urljoin
import numpy as np
import mne


# Helper functions
from autoreject import get_rejection_threshold
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
ICA_N_COMPONENTS = 0.999
RANDOM_STATE = 42
REJECT_TMAX = 0.8  # duration we really care about
# Minimal distance for the forward model
MINDIST = 5
# Spacing of the source space used in the forward model
SOURCE_SPACE_SPACING = 'oct6'

# Mapping openneuro subject codes to the openfmri ones. See section "RELATIONSHIP OF SUBJECT NUMBERING RELATIVE TO OTHER
# VERSIONS OF DATASET" at https://openneuro.org/datasets/ds000117/versions/1.0.4
OPENNEURO_TO_OPENFMRI_SUBJECT_NUMBER = {
    '01': '002',
    '02': '003',
    '03': '004',
    '04': '011',
    '05': '006',
    '06': '007',
    '07': '008',
    '08': '009',
    '09': '010',
    '10': '012',
    '11': '013',
    '12': '014',
    '13': '015',
    '14': '017',
    '15': '018',
    '16': '019'
}


def openfmri_input(openneuro_subject_number, openfmri_template):
    openfmri_subject_number = OPENNEURO_TO_OPENFMRI_SUBJECT_NUMBER[openneuro_subject_number]
    return str(openfmri_template).format(openfmri_subject_number=openfmri_subject_number)



# Folders
data_dir = Path(os.environ['reproduction-data'])
downloads_dir = data_dir / 'downloads'
bids_dir = data_dir / 'bids'
derivatives_dir = bids_dir / 'derivatives'
preprocessing_dir = derivatives_dir / '01_preprocessing'
# TODO: rename both the variable and the directory later
processing_dir = derivatives_dir / '02_processing'
source_modeling_dir = derivatives_dir / '03_source_modeling'

openneuro_maxfiltered_dir = derivatives_dir / 'meg_derivatives'
freesurfer_dir = data_dir / 'reconned'

# Templates
run_template = (openneuro_maxfiltered_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_proc-sss_meg.fif')
events_template = (bids_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_events.tsv')
filtered_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                     'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_filteredHighPass{l_freq}.fif')
ica_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_filtered.fif')
maxfilter_log_template = (openneuro_maxfiltered_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                          'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_proc-sss_log.txt')
bad_channels_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                         'sub-{subject_number}_ses-meg_task-facerecognition_run-{run_id}_bads.fif')
concatenated_raw_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                             'sub-{subject_number}_ses-meg_task-facerecognition_proc-sss_megConcatenated.fif')
concatenated_events_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                                'sub-{subject_number}_ses-meg_task-facerecognition_proc-sss_eventsConcatenated.fif')
epoched_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                         'sub-{subject_number}_ses-meg_task-facerecognition_epo.fif')
ecg_epochs_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                       'sub-{subject_number}_ses-meg_task-facerecognition_ecgEpochs.fif')
eog_epochs_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                       'sub-{subject_number}_ses-meg_task-facerecognition_eogEpochs.fif')
artifact_components_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                       'sub-{subject_number}_ses-meg_task-facerecognition_artifactComponents.npz')
epochs_cleaned_template = (preprocessing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                           'sub-{subject_number}_ses-meg_task-facerecognition_epoCleaned.fif')
evoked_template = (processing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                   'sub-{subject_number}_ses-meg_task-facerecognition_evo.fif')
covariance_template = (processing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                       'sub-{subject_number}_ses-meg_task-facerecognition_cov.fif')
tfr_template = (processing_dir / 'sub-{subject_number}' / 'ses-meg' / 'meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_{measure}-{condition}.fif')
group_average_evokeds_path = processing_dir / 'ses-meg' / 'meg' / 'ses-meg_task-facerecognition_evo-ave.fif'
bids_t1_sidecar_template = (bids_dir / 'sub-{subject_number}' / 'ses-mri' / 'anat' /
                            'sub-{subject_number}_ses-mri_acq-mprage_T1w.json')
bids_t1_template = bids_t1_sidecar_template.with_suffix('.nii.gz')
freesurfer_t1_template = freesurfer_dir / 'sub{openfmri_subject_number}' / 'mri' / 'T1.mgz'
transformation_template = source_modeling_dir / 'sub-{subject_number}' / 'sub-{subject_number}-trans.fif'
bem_src_template = (freesurfer_dir / 'sub{openfmri_subject_number}' / 'bem' /
                f'sub{{openfmri_subject_number}}-{SOURCE_SPACE_SPACING}-src.fif')
bem_sol_template = (freesurfer_dir / 'sub{openfmri_subject_number}' / 'bem' /
                    'sub{openfmri_subject_number}-5120-bem-sol.fif')
forward_model_template = (source_modeling_dir / 'sub-{subject_number}' /
                          f'sub-{{subject_number}}_spacing-{SOURCE_SPACE_SPACING}-fwd.fif')


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
        transformation = expand(transformation_template, subject_number=['01']),
        forward_model = expand(forward_model_template, subject_number=['01']),


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
freesurfer_file = fr'derivatives{dir_separator}freesurfer{dir_separator}.*'

openneuro_filepath_regex = fr'({file_in_subject_folder}|{maxfiltered_file}|{freesurfer_file})'


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
        freesurfer_t1 = lambda wildcards: openfmri_input(wildcards.subject_number, freesurfer_t1_template)
    output:
        trans = transformation_template
    run:
        trans = estimate_trans(bids_t1_path=input.bids_t1, bids_t1_sidecar_path=input.bids_t1_sidecar,
            freesurfer_t1_path=input.freesurfer_t1, bids_meg_path=input.run01)
        trans.save(output.trans)


def make_forward_model(evoked_path, trans_path, src_path, bem_path, forward_model_path):
    info = mne.io.read_info(evoked_path)
    # Because we use a 1-layer BEM, we do MEG only
    fwd = mne.make_forward_solution(info, trans_path, src_path, bem_path,
                                    meg=True, eeg=False, mindist=MINDIST)
    mne.write_forward_solution(forward_model_path, fwd, overwrite=True)


def freesurfer_inputs(wildcards):
    openfmri_subject_number = OPENNEURO_TO_OPENFMRI_SUBJECT_NUMBER[wildcards.subject_number]
    return str(freesurfer_t1_template).format(openfmri_subject_number=openfmri_subject_number)


rule run_forward:
    input:
        evoked = evoked_template,
        transformation = transformation_template,
        src = lambda wildcards: openfmri_input(wildcards.subject_number, bem_src_template),
        bem = lambda wildcards: openfmri_input(wildcards.subject_number, bem_sol_template)
    output:
        forward_model = forward_model_template
    run:
        make_forward_model(evoked_path=input.evoked, trans_path=input.transformation, src_path=input.src,
            bem_path=input.bem, forward_model_path=output.forward_model)
