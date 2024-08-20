PyDeconv
==========

PyDeconv is an open-source Python package for empowering neuroimaging with ERP deconvolution for EEG and MEG data. It includes modules for creating linear models based on experimental features. You can define interactions and non-linear contributions modeled via B-Splines into the models. Additionally, PyDeconv provides tools to estimate the collinearity between features using the VIF module.

## Documentation

[Documentation](#) for PyDeconv tutorials and examples.

## Installation

Follow the installation guide in the documentation to get started with PyDeconv.

## Dependencies

The minimum required dependencies to run PyDeconv examples are:

- [Python](https://www.python.org) ≥ 3.9
- [MNE](https://mne.tools/stable/index.html) ≥ 1.3.1  
- [NumPy](https://numpy.org) ≥ 1.24.2
- [SciPy](https://scipy.org) ≥ 1.10.1
- [Matplotlib](https://matplotlib.org) ≥ 3.6
- [Pandas](https://pandas.pydata.org) ≥ 2.1.0
- [scikit-learn](https://scikit-learn.org) ≥ 1.2.2  

## Use

The main class `PyDeconv` takes three arguments: 
1. The parsed configurations from `config.py`, 
2. A DataFrame with columns labeled with event types to model and the features to use as predictors, 
3. The EEG data as a `mne` raw object.

## Example Scripts

### `example_script.py`
The `example_script.py` script is designed to show some of the functionalities of the tool and runs a simple model over the exaple data. 

It also can be use the example_notebook that applied a simple model to a sample dataset. This notebook will run through the process of setting up the deconvolution model, applying it to the data, and printing out the results of the analysis.

