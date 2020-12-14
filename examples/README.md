## Examples

The folders and files for this directory are as follows:

example_exceom.in - Example Input file for an excited states Equation-of-motion calculation      
be\_sto3g\_oneint_spino.npy - One-electron integrals in the spinorbital basis   
be\_sto3g\_twoint_spino.npy - Two-electron integrals in the spinorbital basis   
be\_sto3g\_onedm\_spino.npy - One-electron reduced density matrix    
be\_sto3g\_twodm\_spino.npy - One-electron reduced density matrix  


### Running a calculation with EOMEE
The following instructions assume that EOMEE package has been installed as indicated in the root directory's README.md file.  

The _examples_ folder contains the required files to run an excited states calculation for Be atom using the Equation-of-motion formalism.  

Move to _examples_ and copy to it the scipt `main.py`, located under the package folder (_eomes_/_main.py_):  

`cd examples/`
`cp ../eomes/main.py .`

If EOMEE was installed inside a virtual environment, activate it:  

```
$ conda activate eomenv
```

To run the calculation do:  

`python main.py example_exceom.txt`

This should produce two output files:  
example_exceom_excen_coeffs.npz - A NumPy .npz file containing the computed excitation energies and coefficients.  
example_exceom.out - All input parameters used by the program.  

Alternatively, the code can be run from an external folder. In that case, copy the content of _examples_ and the running script to the working directory (e.g. my\_dir):  

`cd ~`  
`mkdir my_dir `  
`cp <path_to_eomee>/eomee/examples/* my_dir/.`  
`cp <path_to_eomee>/eomee/eomes/main.py my_dir/.`  
`cd my_dir`  

and then activate the virtual environment and run EOMEE.  


### Comments on the input options
The file _example\_exceom.in_ contains the following required input parameters:   
**nelec** - Number of electrons in the system    
**oneint\_file** -  Path to the one-electron integrlas file (Must be a NumPy .npy)    
**twoint\_file** -  Path to the one-electron integrlas file (Must be a NumPy .npy)    
**dm1\_file** -  Path to the one-electron reduced density matrix file (Must be a NumPy .npy)  
**dm2\_file** -  Path to the two-electron reduced density matrix file (Must be a NumPy .npy)  
**orthog** - Orthogonalization method used in the solution of a generalized eigenvalue problem (symmetric or asymmetric)  
**eom** - Selection of an Equation-of-motion method (one of `exc`, `ip`, `ea`, `dip` and `dea`). See Section 4.2.5 in the [SRS.pdf](https://github.com/gabrielasd/eomee/tree/cas741/docs/SRS)  

Optional parameters incude   
**tol** - Factor used during orthogonalization to control numerical instabilities. Defaults to 1.0e-7 if not specified.  
**get_tdm** - Request to evaluate the transition density matrix. Defaults to False if not specified.  
**roots** - Request to print the specified number of solutions to the output file (.out). Defaults to None.  

Aditional input files (.in) can be found under the project's _test/data_ folder.


