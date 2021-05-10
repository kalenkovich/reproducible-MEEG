from pathlib import Path
import os
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


# Folders
data_dir = Path(os.environ['reproduction-data'])
downloads_dir = data_dir / 'downloads'

# Templates

# Other file-related variables
openfmri_url_prefix = 'https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/'
openfmri_zip_files = [
    'ds000117_R1.0.0_sub01-04.zip',
    'ds000117_R1.0.0_sub05-08.zip',
    'ds000117_R1.0.0_sub09-12.zip',
    'ds000117_R1.0.0_sub13-16.zip',
]

rule download_data_from_openfmri:
    input:
        [downloads_dir / filename for filename in openfmri_zip_files]

rule download_single_file_from_openfmri:
    output:
        file_path = downloads_dir / '{filename}'
    run:
        url = urljoin(openfmri_url_prefix, wildcards.filename)
        download_file_from_url(url=url, save_to=output.file_path)
