# AI for Non-aqueous Electrolyte Design
This software package implements the SCAN (Shaping Conductivity Atlas for Non-aqueous electrolytes) that takes Li-salts, solvents, and conditions to predict the ionic conductivity.

The package provides two major functions:
* Calculate the descriptors based on Li-salts, solvents, and conditions.
* Train a SCAN model with the entire data

## Prerequisites
* Pytorch
* rdkit
* scikit-learn
* pysr

The easiest way of installing the prerequisites is via `conda`. After installing `conda`, run the following command to create a new environment named `scan` and install all prerequisites:

    conda upgrade conda
    conda create -n scan python=3.12 scikit-learn pytorch rdkit pysr

This creates a `conda` environment for running `SCAN`. Before using `SCAN`, activate the environment by:
    
    source activate scan

Then, in directory `model`, you can test if all the prerequisites are installed properly by running:

    python train.py

After you finished using `SCAN`, exit the environment by:

    source deactivate

## Tools
We provide the practical tools in `utils` directory for calculating the ionic conductivity, constructing simulation box, and calculating molecular properties.

* Molecular property calculation
If you want to re-calcualte or re-design the molecular properties, just provide the `SMILES` for a given molecule, and run:

    python molecular_property.py

* Simulation box construction
This tool is useful to calculate the number of molecules for Li-salts and two solvents in the simulation box, according to parameters: `box_size`, `density`, `salt_concentration`, `salt_molar_mass`, `solvent1_molar_mass`,` solvent2_molar_mass`, `solvent_mass_ratio`, by running:

    python box_construction.py

* Conductivity calculation
After obtaining the diffusion coefficent from the MD simultions, you can use this tool to calculate the ionic conductivity based on Arrhenius equation:

    python conductivity_calculation.py


## Data
To reproduce our paper, you can download the corresponding datasets in `data` directory.

## Authors
This software was primarily written by `Dr. Zhilong Wang` who was advised by `Prof. Fengqi You`.

## How to cite
Please cite the following work if you want to use SCAN:

    Zhilong Wang, Fengqi You*. Submitted (2025).





