"""Utility functions."""


from os import path
import numpy as np
from eomee.ionization import EOMIP, EOMIPDoubleCommutator, EOMIPAntiCommutator
from eomee.electronaff import EOMEA, EOMEADoubleCommutator, EOMEAAntiCommutator
from eomee.doubleionization import EOMDIP
from eomee.excitation import EOMExc
from eomee.doubleelectronaff import EOMDEA
from eomee.tools import hartreefock_rdms


EOMSTATES = {
    "ip": EOMIP,
    "ipc": EOMIPDoubleCommutator,
    "ipa": EOMIPAntiCommutator,
    "ea": EOMEA,
    "eac": EOMEADoubleCommutator,
    "eaa": EOMEAAntiCommutator,
    "exc": EOMExc,
    "dip": EOMDIP,
    "dea": EOMDEA,
}


def _check_inputs(
    state,
    int_files,
    dm_files,
    occs,
    solver,
    nvals,
    filename,
    save_lhs,
    save_rhs,
    save_eigvals,
    save_coeffs,
    save_nexcs,
    **kwargs,
):
    """Verify `main_eom` script input parameters.

    Parameters
    ----------
    state : str
        Type of wavefuntion ansatze for excited states.
    int_files : list/tuple
        Electron integrals files paths.
    dm_files : list/tuple
        Density matrices files paths.
    occs : list/tuple, optional
        Number of alpha and beta electrons.
    solver : str, optional
        Solver used to solve the EOM generalized eigenvalue problem.
        Valid options are `dense` and `sparse` solvers.
    nvals : int, optional
        Number of transition energies to output.
    filename : str, optional
        Prefix to be used in the name of the output files.
    save_lhs : bool, optional
        Whether to save the left-hand-side matrix of the EOM problem.
    save_rhs : bool, optional
        Whether to save the right-hand-side matrix of the EOM problem.
    save_eigvals : bool, optional
        Whether to save the solution eigenvalues (transition energies).
    save_coeffs : bool, optional
        Whether to save the solution eigenvectors (excited state wavefunction parameters).
    save_nexcs : bool, optional
        Whether to save the `nvals` energies in an output file.

    Raises
    ------
    TypeError
        [description]
    ValueError
        [description]
    AssertionError
        [description]
    """
    # Check electron integrals
    if not isinstance(int_files, (list, tuple)):
        raise TypeError("Electron integral files must be passed as a list/tuple.")
    if len(int_files) == 1:
        if not isinstance(int_files[0], str):
            raise TypeError(
                f"The path to the FCIDUMP file must be a string, got {type(int_files[0])} instead."
            )
        if not path.isfile(int_files[0]):
            raise AssertionError(f"File {int_files[0]} not found.")
    elif len(int_files) == 2:
        oneint_file, twoint_file = int_files
        if not all([isinstance(oneint_file, str), isinstance(twoint_file, str)]):
            raise TypeError("Electron integrals must be numpy npy files.")
        for intf in int_files:
            if not path.isfile(intf):
                raise AssertionError(f"File {intf} not found.")
    else:
        raise TypeError(
            f"Only one or two files paths should be provided, {len(int_files)} were given."
        )
    # Check density matrices
    if dm_files is not None:
        if not (isinstance(dm_files, (list, tuple)) and len(dm_files) == 2):
            raise TypeError(f"Only two files paths should be provided, {len(dm_files)} were given.")
        if not all(isinstance(dm_files[0], str), isinstance(dm_files[1], str)):
            raise TypeError("The density matrices must be numpy npy files.")
        for dmf in dm_files:
            if not path.isfile(dmf):
                raise AssertionError(f"File {dmf} not found.")
    # Check EOM state
    if not state.lower() in EOMSTATES.keys():
        raise ValueError(
            f"Invalid EOM method. State must be one of {list(EOMSTATES.keys())}. {state} given."
        )
    # Check occupations
    # if "occs" in kwargs.keys():
    #     occs = kwargs["occs"]
    if occs is not None:
        if not (isinstance(occs, (list, tuple)) and all(isinstance(i, int) for i in occs)):
            raise TypeError("Number of electrons must be given as list/tuple of integers.")
        elif not len(occs) == 2:
            raise ValueError("`occs` must have only two elemets, the alpha and beta occupations.")
    # Check eigenvalue problem solver
    # if "solver" in kwargs.keys():
    #     solver = kwargs["solver"]
    if not solver.lower() in ["dense", "sparse"]:
        raise ValueError(f"Solver must be one of `dense` or `sparse`. {solver} given.")
    # Check number of transitions to output
    if not isinstance(nvals, int):
        raise TypeError("Number of transitions must be an integer.")
    # Check output files
    if filename is not None:
        if not isinstance(filename, str):
            raise TypeError(f"The file's name must be a string, {type(filename)} given.")
    # out_files = ["save_lhs", "save_rhs", "save_eigvals", "save_coeffs", "save_nexcs"]
    out_files = [save_lhs, save_rhs, save_eigvals, save_coeffs, save_nexcs]
    for outf in out_files:
        if not isinstance(outf, bool):
            raise TypeError(f"Parameter {outf} must be of boolean type.")


def _load_integrals(int_files):
    """Read the electron integrals from numpy files or a FCIDUMP.

    Parameters
    ----------
    int_files : list/tuple
        Electron integrals file paths (stored a numpy files or FCIDUMP).

    Returns
    -------
    np.ndarray
        The one- and two-electron integrals as numpy arrays. These are expected
        to be in the spatial molecular orbital basis.

    Raises
    ------
    TypeError
        If the file is not a FCIDUMP.
        Only one (FCIDUMP) or two (.npy) files paths should be provided.
    NotImplementedError
        FCIDUMP input is not suported yet.
    """
    if len(int_files) == 2:
        oneint_file, twoint_file = int_files
        oneint = np.load(oneint_file)
        twoint = np.load(twoint_file)
        if oneint.ndim > twoint.ndim:
            oneint, twoint = twoint, oneint
    elif len(int_files) == 1:
        _, fext = path.splitext(int_files[0])
        if not (fext in [".fcidump", ".FCIDUMP"]):
            raise TypeError(f"FCIDUMP file expected but got {int_files[0]} instead.")
        raise NotImplementedError("FCIDUMP input is not suported yet.")
    else:
        raise TypeError(
            f"Only one or two files paths should be provided, {len(int_files)} were given."
        )
    return oneint, twoint


def _load_dms(dm_files, nbasis=None, occs=None):
    """Read the one- and two-particle reduced density matrices from numpy files.

    Parameters
    ----------
    dm_files : list/tuple
        Density matrices file paths (stored a numpy files).
    nbasis: int, optional
        Number of spatial molecular orbitals.
    occs: list/tuple of ints, optional
        Number of alpha and beta electrons.

    Returns
    -------
    np.ndarray
        The one- and two-electron reduced density matrices. These are expected
        to be in the generalized spin-orbital basis.

    Raises
    ------
    ValueError
        If the `dm_files` is not None or a list/tuple with two file paths.
    """
    if dm_files is None:
        assert nbasis is not None
        assert occs is not None
        rdm1, rdm2 = hartreefock_rdms(nbasis, *(occs))
    elif isinstance(dm_files, (list, tuple)) and len(dm_files) == 2:
        rdm1_file, rdm2_file = dm_files
        rdm1 = np.load(rdm1_file)
        rdm2 = np.load(rdm2_file)
        if rdm1.ndim > rdm2.ndim:
            rdm1, rdm2 = rdm2, rdm1
    else:
        raise ValueError(f"`dm_files` must be None or a list with two file paths.")
    return rdm1, rdm2
