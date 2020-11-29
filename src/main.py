"""
Equations-of-motion state base class.

"""


import os, sys

import numpy as np
import load
import integrals
import density
import eom
import solver
import output


inputfile = sys.argv[1]
params = load.parse_inputfile(inputfile)
load.check_inputs(params)

ham = integrals.ElectronIntegrals(params.oneint_file, params.twoint_file)
wfn = density.WfnRDMs(params.npart, params.dm1_file, params.dm2_file)

if params.eom == "ip":
    eommethod = eom.IonizationEOMState
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

eomee = eommethod(ham.h, ham.v, wfn.dm1, wfn.dm2)
exce, coeffs = solver.dense(eomee.lhs, eomee.rhs, params.tol, params.orthog)

output.dump(inputfile, exce, coeffs, tdms=None)
