"""Excitation energies control module."""


import argparse
import numpy as np
from eomee.tools import spinize, pickeig
from eomee.scripts.utils import EOMSTATES, _check_inputs, _load_integrals, _load_dms



def run_eom(
    state,
    int_files,
    dm_files=None,
    occs=None,
    solver="dense",
    nvals=3,
    filename=None,
    save_lhs=False,
    save_rhs=False,
    save_eigvals=False,
    save_coeffs=False,
    save_nexcs=False,
    **kwargs,
):
    """Compute transition energies and excited state wavefunctions through the Equation-of-motion formalism.

    Parameters
    ----------
    state : str
        Type of wavefuntion ansatze for excited states. Must be one of `ip`, `ipc`, `ipa`, `ea`,
        `eac`, `eaa`, `exc`, `dip`, `dea`.
    int_files : list/tuple
        Electron integrals files paths.
    dm_files : list/tuple, optional
        Density matrices files paths, by default None.
    occs : list/tuple, optional
        Number of alpha and beta electrons, by default None. Must be specified if `dm_files` is
        None.
    solver : str, optional
        Solver used to solve the EOM generalized eigenvalue problem.
        Valid options are `dense` and `sparse` solvers, by default `dense`.
    nvals : int, optional
        Number of transition energies to output. Default is 3.
    filename : str, optional
        Prefix to be used in the name of the output files, by default None.
    save_lhs : bool, optional
        Whether to save the left-hand-side matrix of the EOM problem. Default to False.
    save_rhs : bool, optional
        Whether to save the right-hand-side matrix of the EOM problem. Default to False.
    save_eigvals : bool, optional
        Whether to save the solution eigenvalues (transition energies). Default to False.
    save_coeffs : bool, optional
        Whether to save the solution eigenvectors (excited state wavefunction parameters). Default to False.
    save_nexcs : bool, optional
        Whether to save the `nvals` energies in an output file. Default to False.

    Returns
    -------
    np.ndarray
        A selected`nvals` transition energies.

    Raises
    ------
    TypeError
        [description]
    NotImplementedError
        [description]
    """
    # Verify input parameters
    _check_inputs(
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
    )

    # Load the electron integrals and density matrices
    # When loaded from a numpy file the integrals are expected
    # to be in the spatial molecular orbital basis and the density
    # matrices in the generalized spin-orbital basis.
    # The electron integrals will be changed to the generalized basis
    # when passed to the EOM.
    oneint, twoint = _load_integrals(int_files)
    if dm_files is None:
        if occs is None:
            raise TypeError("The number of alpha and beta occupations (`occs`) must be provided.")
        nbasis = oneint.shape[0]
        rdm1, rdm2 = _load_dms(dm_files, nbasis=nbasis, occs=occs)
    else:
        rdm1, rdm2 = _load_dms(dm_files)

    #
    # Select an EOM and solve the generalized eigenvalue problem
    #
    eom = EOMSTATES[state](spinize(oneint), spinize(twoint), rdm1, rdm2)

    if solver == "dense":
        if "tol" not in kwargs.keys():
            tol = 1.0e-10
        else:
            tol = kwargs["tol"]
        if "orthog" not in kwargs.keys():
            orthog = "asymmetric"
        else:
            orthog = kwargs["orthog"]
        ev, cv = eom.solve_dense(tol=tol, orthog=orthog, **kwargs)
    else:
        raise NotImplementedError(f"{solver} solver not available yet.")
    #
    # Output the lowest `nvals` energies
    #
    nexcs = pickeig(ev)[:nvals]

    #
    # Output results
    #
    output = f"{state}" if filename is None else f"{filename}_{state}"
    out_npy = {
        save_lhs: ("lhs", eom.lhs),
        save_rhs: ("rhs", eom.rhs),
        save_eigvals: ("eigvals", ev),
        save_coeffs: ("coeffs", cv),
    }
    for k_arg, vals in out_npy.items():
        if k_arg:
            np.save(f"{output}_{vals[0]}.npy", vals[1])
    if save_nexcs:
        np.savetxt(f"{output}_nexcs.csv", nexcs, delimiter=",")
    return nexcs


def parser_add_eom_args(parser):
    """Parse arguments for `run_eom.py` script."""
    parser.add_argument(
        "state",
        type=str,
        choices=["ip", "ipc", "ipa", "ea", "eac", "eaa", "exc", "dip", "dea"],
        help="Type of wavefuntion ansatze for the excited states. Must be one of `ip`, `ipc`, `ipa`, "
        "`ea`, `eac`, `eaa`, `exc`, `dip`, `dea`.",
    )
    parser.add_argument(
        "--int_files",
        nargs="+",
        type=str,
        required=True,
        help="Name of the files that contain the one- and two-electron integrlas. Numpy or FCIDUMP formats are allowed.",
    )
    parser.add_argument(
        "--dm_files",
        nargs=2,
        type=str,
        default=None,
        help="Name of the numpy files that contain the one- and two-particle reduced density matrices."
        "[default=%(default)s]",
    )
    parser.add_argument(
        "--occs",
        nargs=2,
        type=int,
        default=None,
        help="Number of alpha and beta electrons, e.g. `--occs 2 1` for two alpha and one beta electrons.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="dense",
        help="Solver used for the EOM generalized eigenvalue problem. Valid options are `dense` and `sparse` solvers."
        "[default=%(default)s]",
    )
    parser.add_argument(
        "--nvals",
        type=int,
        default=3,
        help="Number of transition energies to output." "[default=%(default)s]",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Prefix to be used in naming the output files." "[default=%(default)s]",
    )
    parser.add_argument(
        "--save_lhs",
        type=bool,
        default=False,
        help="Whether to save the left-hand-side matrix of the EOM problem."
        "[default=%(default)s]",
    )
    parser.add_argument(
        "--save_rhs",
        type=bool,
        default=False,
        help="Whether to save the right-hand-side matrix of the EOM problem."
        "[default=%(default)s]",
    )
    parser.add_argument(
        "--save_eigvals",
        type=bool,
        default=False,
        help="Whether to save the solution eigenvalues (transition energies)."
        "[default=%(default)s]",
    )
    parser.add_argument(
        "--save_coeffs",
        type=bool,
        default=False,
        help="Whether to save the solution eigenvectors (excited state wavefunction parameters)."
        "[default=%(default)s]",
    )
    parser.add_argument(
        "--save_nexcs",
        type=bool,
        default=False,
        help="Whether to save the `nvals` energies in an output file." "[default=%(default)s]",
    )


def main():
    """Entry point function for excitation energies evaluated through EOM."""
    parser = argparse.ArgumentParser(prog='eom')
    parser.description = "Compute excited states."

    parser_add_eom_args(parser)
    args = parser.parse_args()
    run_eom(
        args.state,
        args.int_files,
        dm_files=args.dm_files,
        occs=args.occs,
        solver=args.solver,
        nvals=args.nvals,
        filename=args.filename,
        save_lhs=args.save_lhs,
        save_rhs=args.save_rhs,
        save_eigvals=args.save_eigvals,
        save_coeffs=args.save_coeffs,
        save_nexcs=args.save_nexcs,
    )


if __name__ == "__main__":
    main()