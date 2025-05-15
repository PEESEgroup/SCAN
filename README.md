# AI for Non-aqueous Electrolyte Design
This software package implements the SCAN (Shaping Conductivity Atlas for Non-aqueous electrolytes) that takes Li-salts, solvents, and conditions to predict the ionic conductivity.

The package provides two major functions:
* Calculate the descriptors based on Li-salts, solvents, and conditions.
* Train a SCAN model with the entire data
![web-1](https://github.com/user-attachments/assets/fce81555-1663-422a-81d3-bdf4ca1915da)

## Prerequisites
* torch==2.2.1
* rdkit==2024.3.6
* scikit-learn==1.5.1
* pysr==1.5.2

The easiest way of installing the prerequisites is via `conda`. After installing `conda`, run the following command to create a new environment named `scan` and install all prerequisites:

    conda upgrade conda
    conda create -n scan python=3.12 scikit-learn pytorch rdkit pysr

This creates a `conda` environment for running `SCAN`. Before using `SCAN`, activate the environment by:
    
    source activate scan

Alternatively, `environment.yaml` provides the dependencies for creating running environment. Then, in directory `model`, you can test if all the prerequisites are installed properly by running:

    python train.py

After you finished using `SCAN`, exit the environment by:

    source deactivate

## Models
We provide the model files in `model` and  `sampling` directories.
* `sampling`: the original multi-feature network (MFNet) is provided, which is one of the baseline model. Additionally, three strategies (over-sampling, under-sampling, hybrid-sampling) with various hyper-parameters were adopted for training baseline models. To run these models:

        python over_sampling_train.py
        python under_sampling_train.py
        python over_under_sampling_train.py

* `model`: MFNet with dynamic routing strategy was implemented for predicting the conductivity, to run the model:
        python train.py

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
* `calisol`: lists the compiled data, including k values, temperature, concentration/unti, salt, solvent. The temperature was scaled by a factor of 100.
* `salt_feature.npy`: feature matrix of salt molecules based on the designed descriptor.
* `solvent_feature.npy`: feature matrix of solvent molecules based on the designed descriptor.
* `condition_feature.npy`: feature matrix of conditions.
* `conductivity_target.txt`: collected k values.

## Author contributions
This software was primarily written by `Dr. Zhilong Wang` who is advised by `Prof. Fengqi You`.

## Web app
A online platform was estabilished that allows researchers to query our non-aqueous electrolyte database and predict conductivity properties using our deep learning models. It accelerates the discovery of non-aqueous electrolyte for battery and energy storage applications.

        https://peese-scan.streamlit.app/


https://github.com/user-attachments/assets/21823485-5af6-4877-bcac-aabc342348c5



## How to cite
Please cite the following work if you want to use SCAN:

    Zhilong Wang, Fengqi You*. Submitted (2025).





