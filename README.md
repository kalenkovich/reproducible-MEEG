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
   
## Troubleshooting

In case of errors when running Snakemake, look through the output in the terminal to see which rule caused the error.

## Rule `download_from_openneuro`

Sometimes downloading data from openeuro does not work on the first attempt and then we have to try again.
There are two workaround:

1. Download the data from the openneuro [dataset](https://openneuro.org/datasets/ds000117/versions/1.0.4) manually and put it in the `bids` folder you created earlier. You will need the following files/folders:
   1. All `sub-*` folder in the root.
   2. The `derivatives/meg_derivatives/` folder.
2. Run Snakemake telling it to try again if a job fails. There is currently no way to do this at the rule level, so we will run only the `download_from_openneuro` rule:
   ```sh
   snakemake --until download_from_openneuro --cores <n_cores> --restart-times 5
   ```
   Once done, run the full workflow again (the downloaded files will not be downloaded again).


## Rules `apply_linear_filter` or `make_artifact_epochs`

These rules run linear filtering at some point which may create issues when another job is running at the same time.
In Snakefile, we told Snakemake that if the `filtering_process` resource is set during Snakemake invocation, then those
two rules will require as much of this resource as there are cores available and thus these rules can't be run in
parallel with each other job (see `resources` nodes of these rules).
This setting does not change anything unless we run Snakemake with the `--resources` flag and set `filtering_process`
resource capacity to the total number of cores.
To avoid these rules running in parallel with other rules we also set the default amount of this resource to `1` for all
the rules that do not have this resource specified with the `default-resources` flag.

```sh
snakemake --cores <n_cores> --resources filtering_process=<n_cores> --default-resources filtering_process=1 --keep-going
```
