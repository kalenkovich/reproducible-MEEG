from pathlib import Path
import os
from zipfile import ZipFile, Path as ZipPath

import requests
from urllib.parse import urljoin


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


# Folders
data_dir = Path(os.environ['reproduction-data'])
downloads_dir = data_dir / 'downloads'
bids_dir = data_dir / 'bids-2'

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


rule get_all_subjects_meg_data:
    input:
        expand(subject_json_template, subject_number=[f'{i:02d}' for i in range(1, 16 + 1)])

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
