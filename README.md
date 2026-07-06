# AF2_resTrain: Injecting distance information into Alphafold-like structure predictors

## On Google Colab

A simplified version of resTrain is available to run as notebook on Google Colab. Try it by clicking on the button below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clami66/resTrain/blob/notebook/notebook/resTrain.ipynb)

## Installing resTrain

### Stand-alone installation

If you don't wish to use a Docker image, the installation and setup procedure is as follows:

1. [Install Anaconda/Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Set up conda environment, install dependencies:

```bash
# clone this repository
git clone https://github.com/clami66/resTrain.git
cd resTrain

conda env create --file=environment.yaml

conda activate resTrain

python -m pip install -r requirements.txt
```

### With Docker

1. Pull the image from Docker Hub. 
```
docker image pull clami66/restrain:test
```

Alternatively, use the provided Dockerfile to build the image from scratch.

2. Run from the docker image while mapping the necessary directories with `-v` and the GPU devices with `--gpus`, e.g.:
```
docker run -v /home:/home --gpus device=0  clami66/restrain:test run_alphafold.py ...
```

### Setting up the databases

[optional] Download and set up the AF parameters and sequence databases. Reduced databases (`reduced_dbs` option) should suffice as evolutionary inputs are not as important when a good template is provided. If the full databases are needed, run the following by omitting `reduced_dbs`:

```bash
cd scripts
chmod +x download_all_data.sh

./download_all_data.sh ../AF_data/ reduced_dbs
```

If you have databases and parameters from a precedent AlphaFold installation, it is not necessary to repeat this step, just make sure that the paths inside `databases.flag` point to the right directories.

## Quick start

### Restraint file format

Restraints can be written as `.tsv` files, each line representing a separate AA pair restraint:

```
chain_id1  res_number1    chain_id2  res_number2    d
...
```

For example, the file `restraints.tsv`:

```
A   5   A   15  5.0
A   10  B   20  8.0
```

defines two restraints:

* intra-chain restraint at 5Å for residues 5 and 15 in chain A;
* inter-chain restraint at 8Å between residue 10 in chain A and residue 20 in chain B.

Restraints can also be distogram-like binned distance probabilities. In this case, the distance field is a comma-separated list of 64 `float` numbers, representing probabilities for each distance bin in AlphaFold's distogram (from [2.31, 2.62) Å in the first bin to [22, ∞) in the last bin):

```
A   5   A   15  0.0,0.0,0.1,0.2,0.3,0.2,0.1,0.1,0.0, ... ,0.0
...
```

## Running resTrain

Restraint files are passed with the flag `--restraints`:

`run_alphafold.py --fasta_paths examples/H1142/H1142.fasta --restraints example/H1142/restraints.tsv`

### Other flags

* `--approximate`: treats all restraints as maximum-distance restraints rather than exact distances (default: `False`)
* `--kl`: use KL divergence as loss, for example when restraining against distance distributions (default: `False`)
* `--optimization_steps`: number of gradient descent steps to optimize distogram loss (default: `10`)
* `--increase_seed_during_optimization`: changes AlphaFold's random seed after each gradient descent steps (default: `False`)
* `--pae_w`: include pAE as loss term during search to encourage better quality predictions (default: `0.0`)

## OF3_resTrain

The OpenFold3 implementation of resTrain will be available soon.

