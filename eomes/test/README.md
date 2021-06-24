## Tests


The folders for this directory are as follows:

data - Data files for tests 

### Dependencies  
- [Pytest](https://docs.pytest.org/en/latest/) >=

### Instructions for running tests  
The instructions below continue from the installation instructions in the root directory's README.md file.

Activate the virtual environment if the package was installed inside one. For a conda environment (e.g. eomenv) this would be:  

```
$ conda activate eomenv
```

Istall the dependencies and run all tests:  

```
$ conda install pytest
```
```
$ pytest test/
```

To run a specific test module (e.g. test_solver.py) do:  

```
$ pytest test/test_solver.py
```




