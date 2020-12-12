"""Control module."""


import sys
from eomes.load import parse_inputfile, check_inputs
from eomes.integrals import ElectronIntegrals
from eomes.density import WfnRDMs
from eomes.eom import EOMIP, EOMEA, EOMExc, EOMDIP, EOMDEA
from eomes.solver import dense
from eomes.output import dump


def main():
    """[summary]

    Raises:
        ValueError: [description]
    """
    args = sys.argv[1:]
    if len(args) == 0:
        print('Expecting one argument: "input_file.in"', file=sys.stderr)
        sys.exit(-1)
    inputfile = args.pop(0)
    params = parse_inputfile(inputfile)
    check_inputs(params)

    ham = ElectronIntegrals(params.oneint_file, params.twoint_file)
    wfn = WfnRDMs(params.nparts, params.dm1_file, params.dm2_file)

    # FIXME: Consider adding a check for number of spinorbitlas
    # in integrlas an density matrices

    if params.eom == "ip":
        eommethod = EOMIP
    elif params.eom == "ea":
        eommethod = EOMEA
    elif params.eom == "exc":
        eommethod = EOMExc
    elif params.eom == "dip":
        eommethod = EOMDIP
    elif params.eom == "dea":
        eommethod = EOMDEA
    else:
        raise ValueError("Invalid EOM method: {}".format(params.eom))

    eomstate = eommethod(ham.h, ham.v, wfn.dm1, wfn.dm2)
    print("Start EOM calcularion")
    exce, coeffs = dense(eomstate.lhs, eomstate.rhs, params.tol, params.orthog)
    print("Done")

    if params.get_tdm:
        tdms = eomstate.compute_tdm(coeffs)
        dump(inputfile, params, exce, coeffs, tdms=tdms)
    else:
        dump(inputfile, params, exce, coeffs)


if __name__ == "__main__":
    main()
