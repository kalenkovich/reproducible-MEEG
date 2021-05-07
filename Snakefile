from pathlib import Path
import os
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
