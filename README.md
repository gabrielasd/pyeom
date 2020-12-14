# eomee

Developer(s) Name:

Date of project start:

This project provides a common framework for implementing and solving Equation-of-motion approximations based on reduced density matrices (RDMs).

The folders and files for this project are as follows:

docs - Documentation for the project  
refs - Reference material used for the project, including papers  
src - Source code  
test - Test cases  

## Dependencies
- [Python](http://python.org/) >= 3.6
- [NumPy](http://numpy.org/) >= 1.13
- [SciPy](http://docs.scipy.org/doc/scipy/reference/) >= 1.0

### Optional dependencies
- [Pytest](https://docs.pytest.org/en/latest/) >=

## Installation
### Conda environment
It is recommended to do the installation inside a virtual environment. The following instructions are for a conda virtual environment. You can obtain conda thorugh a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installation.

Create a conda environment and install the dependencies:

```
$ conda create -n eomenv python=3.6 numpy scipy
```

To activate the conda environment (e.g. eomenv) do:

```
$ conda activate eomenv
```

### EOM source code download and installation
Download eomee and move to the package directory:

```
$ git clone -b cas741 https://github.com/gabrielasd/eomee.git && cd eomee
```

To install in "editable" mode do:

```
$ pip install -e .
```

## Uninstallation
To unistall eomee run:

```
$ pip uninstall eomee
```

## Running tests
Please see the README.md file under the _test_ file.


## Running an example
Please see the README.md file under the _examples_ file.


