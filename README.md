# Reproduced MEEG Group Study

Code that attempts to reproduce "A Reproducible MEG/EEG Group Study With the MNE Software" by Jas et al., 2018 ([link](https://www.frontiersin.org/articles/10.3389/fnins.2018.00530/full))

# To run the analysis:

1. Download the files in this project.
2. Select a folder where you will store the data and set an environment variable `reproduction_data` to its path.
3. Create a folder `bids/derivatives/freesurfer_lk/` and download the contents of the `freesurfer_lk` [component](https://osf.io/jy2g7/) into it.
4. In a terminal, navigate to the repository root.
5. Create and activate the conda environment:
    ```sh
    conda env create -n reproduction -f environment.yml
    conda activate reproduction
    ```
6. Run the full analysis with
    ```sh
    snakemake all --cores <n_cores> --keep-going
    ```
   Where `n_cores` is the number of processing cores you have available for the analysis.
   
