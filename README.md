<h1>Code for "Rethinking Backdoor Attacks"</h1>

Presented at ICML 2023. Cite paper as:

```
@inproceedings{khaddaj2023rethinking,
    title = {Rethinking Backdoor Attacks},
    author = {Alaa Khaddaj and Guillaume Leclerc and Aleksandar Makelov and Kristian Georgiev and Hadi Salman and Andrew Ilyas and Aleksander Madry},
    booktitle = {ICML},
    year = {2023},
}
```

## Getting started

This repository implements the maximum-sum submatrix subroutine from our backdoor defense. To use it:

1. Clone the repo

2. Install our code dependencies
    ```
        conda env create -f env.yml -y
        conda activate poisenv
    ```

3. Copy your datamodel matrix in the folder (or specify its path using `DM_PATH` variable in `run.sh` script). To compute the datamodel matrix, you can check the [datamodel repo](https://github.com/MadryLab/datamodels).

4. Run the bash script `run.sh`. The resulting output to analyze will be saved in `./results/scores/sample_scores.npy` file. Each index value in the array is the score of the target example returned by our algorithm. The inputs with the highest scores will be flagged as backdoored.
