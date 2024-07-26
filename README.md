# PyDeconv
.. -*- mode: rst -*-


PyDeconv
==========

PyDeconv is an open-source Python package for empowering neuroimaging with ERP deconvolution for EEG and MEG data.
It includes modules for create linear models based on experimental features. It is possible to deffine interactions
and non-linear contributions moled via B-Splines into the models. It is possible to estimate the collinearity between 
features with the VIF module.


Documentation
^^^^^^^^^^^^^

`Documentation`_ for PyDeconv tutorials,
and examples.

Installation
^^^^^^^^^^^^




Dependencies
^^^^^^^^^^^^

The minimum required dependencies to run MNE-Python are:

- `Python <https://www.python.org>`__ ≥ 3.9
- `NumPy <https://numpy.org>`__ ≥ 1.24.2
- `SciPy <https://scipy.org>`__ ≥ 1.10.1
- `Matplotlib <https://matplotlib.org>`__ ≥ 3.6
- `Pandas <https://pandas.pydata.org>`__ ≥ 2.1.0
- `scikit-learn <https://scikit-learn.org>`__ ≥ 1.2.2  

Use
^^^
The main function takes only two arguments first a Pandas DataFrame keyworded as features indicating the kind 
of event to be modeledand the features taking part in the model formula as colums. It should look like this:


![features](https://github.com/user-attachments/assets/38463a3e-358d-4341-8e08-1acfa841b33b)


