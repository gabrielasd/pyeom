"""Control module."""


import sys
import load
import integrals
import density
import eom
import solver
import output


def main():
    """[summary]

    Raises:
        ValueError: [description]
    """
    args = sys.argv[1:]
    if len(args) == 0:
        print('Expecting one argument: "input_file.txt"', file=sys.stderr)
        sys.exit(-1)
    inputfile = args.pop(0)
    params = load.parse_inputfile(inputfile)
    load.check_inputs(params)

    ham = integrals.ElectronIntegrals(params.oneint_file, params.twoint_file)
    wfn = density.WfnRDMs(params.nparts, params.dm1_file, params.dm2_file)

    # FIXME: Consider adding a check for number of spinorbitlas
    # in integrlas an density matrices

    if params.eom == "ip":
        eommethod = eom.EOMIP
    elif params.eom == "ea":
        eommethod = eom.EOMEA
    elif params.eom == "exc":
        eommethod = eom.EOMExc
    elif params.eom == "dip":
        eommethod = eom.EOMDIP
    elif params.eom == "dea":
        eommethod = eom.EOMDEA
    else:
        raise ValueError("Invalid EOM method: {}".format(params.eom))

    eomstate = eommethod(ham.h, ham.v, wfn.dm1, wfn.dm2)
    print("Start EOM calcularion")
    exce, coeffs = solver.dense(eomstate.lhs, eomstate.rhs, params.tol, params.orthog)
    print("Done")

    if params.get_tdm:
        tdms = eomstate.compute_tdm(coeffs)
        output.dump(inputfile, params, exce, coeffs, tdms=tdms)
    else:
        output.dump(inputfile, params, exce, coeffs)


if __name__ == "__main__":
    main()
