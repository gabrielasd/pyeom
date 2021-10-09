"""Excitation energies calculation example."""

from eomee.scripts.run_eom import run_eom
from eomee.tools import find_datafiles


#
# The following script gives an example of how to run an excited state
# calculation using eomee
#
mol='be_sto3g'
state='ip'
one_int_files=find_datafiles(f'{mol}_oneint.npy')
two_int_files=find_datafiles(f'{mol}_twoint.npy')
int_files=[one_int_files, two_int_files]
dm_files=None
occs=(2,2)
solver='dense'
nvals=3
filename=None
save_lhs=False
save_rhs=False
save_eigvals=False
save_coeffs=False
save_nexcs=False

answers = run_eom(
        state,
        int_files,
        dm_files=dm_files,
        occs= occs,
        solver= solver,
        nvals= nvals,
        filename= filename,
        save_lhs= save_lhs,
        save_rhs= save_rhs,
        save_eigvals= save_eigvals,
        save_coeffs= save_coeffs,
        save_nexcs= save_nexcs,
    )
print("EOM excited states", answers)
