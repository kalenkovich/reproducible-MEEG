# Reproduced MEEG Group Study

Code that attempts to reproduce "A Reproducible MEG/EEG Group Study With the MNE Software" by Jas et al., 2018 ([link](https://www.frontiersin.org/articles/10.3389/fnins.2018.00530/full))

# To re-run the analysis:

1. Clone/download this repository.
2. Select a folder where you will store the data and set an environment variable `reproduction-data` to its path.
3. In a terminal, navigate to the repository root.
4. Create and activate the conda environment:
    ```sh
    conda env create -n reproduction -f environment.yml
    conda activate reproduction
    ```
5. Download the archive files from openfmri:
```sh
snakemake download_data_from_openfmri --cores <n>
```
