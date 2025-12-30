# ICU Multiclass Classifier

This project develops and externally validates a multiclass machine-learning
meta-classifier for diagnosis-based stratification of ICU patients that supports
scalable, robust, and sustainable AI deployment.

For this project we use routinely collected demographic and physiological variables, we trained and evaluated multiple machine-learning models on the MIMIC-IV database and performed external validation on the eICU database.

## Installation

This project uses **Poetry** for dependency management and it recommends using **Conda** to manage the Python environment.

### Step 1: Clone the repository

Clone the repository using ssh cloning:

    git clone git@github.com:santiago-salas-iic/ICU_Multiclass_Classifier.git
    cd ICU_Multiclass_Classifier

### Step 2: Create and activate a Conda environment

Create a new Conda environment with the required Python version (it can be 3.10, 3.11 or 3.12)

    conda create -n icu_multiclass_classifier python=3.10

Activate the environment:

    conda activate icu_multiclass_classifier

### Step 3: Install Poetry

Install **Poetry** inside the **Conda** environment:

    conda install -c conda-forge poetry

### Step 4: Install required project dependencies

Use **Poetry** to install all required dependencies:

    poetry install


## Download data

For this project we use **MIMIC-IV** version 2.2 and **eICU** version 2.0.

### Download MIMIC-IV 2.2

From the root of the project run the following command

    wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimiciv/2.2/

### Download eICU 2.0

From the root of the project run the following command

    wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/eicu-crd/2.0/

## Usage

To use this project we provide a jupyter notebook <code>src/main.ipynb</code>. This notebook will use our pipeline to load the **MIMIC-IV** version 2.2 and **eICU** version 2.0 datasets as tabular data, preprocess it for ML, train models and run experiments.

