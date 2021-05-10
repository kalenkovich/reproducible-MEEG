from pathlib import Path
import os
from zipfile import ZipFile

import requests
from urllib.parse import urljoin


def download_file_from_url(url, save_to):
    response = requests.get(url)
    # Raise an error if there was a problem
    response.raise_for_status()

    with open(save_to, 'wb') as file:
        file.write(response.content)


# Folders
data_dir = Path(os.environ['reproduction-data'])
downloads_dir = data_dir / 'downloads'
bids_dir = data_dir / 'bids'

# Templates
subject_json_template = (bids_dir / 'sub-{subject_number}' / 'ses-meg' /
                'sub-{subject_number}_ses-meg_task-facerecognition_proc-tsss_meg.json')

# Other file-related variables
openfmri_url_prefix = 'https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/'
openfmri_zip_files = [
    'ds000117_R1.0.0_sub01-04.zip',
    'ds000117_R1.0.0_sub05-08.zip',
    'ds000117_R1.0.0_sub09-12.zip',
    'ds000117_R1.0.0_sub13-16.zip',
]


checkpoint unpack_subject_folders:
    input:
        archive = downloads_dir / 'ds000117_R1.0.0_sub{start}-{end}.zip'
    output:
        'dummy_output_{start}-{end}'

        {start}.json
        {start + 1}.json
        {start + 1}.json
    run:
        with ZipFile(input.archive, 'r') as zip_object:
            for member in zip_object.namelist():
                # We only want the MEG files
                if 'ses-meg' in member and not member.endswith('/') and subject_kotoryj_nuzhen:
                    # Skip the root folder in the archive
                    relative_path = Path(*Path(member).parts[1:])
                    output_path = bids_dir / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(zip_object.read(member))


def get_subject_data_from_archive(wildcards):
    k = int(wildcards.subject_number)
    start = (k - 1) // 4  * 4 + 1
    end = start + 3
    checkpoints.unpack_subject_folders.get(start=f'{start:02d}', end=f'{end:02d}')


rule get_single_subject_data:
    input:
        get_subject_data_from_archive
    output:
        subject_json_template

rule get_all_subject_data:
    input:
        expand(subject_json_template, subject_number=range(1, 16 + 1))


rule download_data_from_openfmri:
    input:
        [downloads_dir / filename for filename in openfmri_zip_files]

rule download_single_file_from_openfmri:
    output:
        file_path = downloads_dir / '{filename}'
    run:
        url = urljoin(openfmri_url_prefix, wildcards.filename)
        download_file_from_url(url=url, save_to=output.file_path)
