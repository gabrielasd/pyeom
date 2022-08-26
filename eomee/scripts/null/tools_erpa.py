import os
import numpy as np


def get_fchk(folder, method, basis, job_name, charge=0, mult=1, state=0):
    basisname = basis.lower().replace("-", "").replace("*", "p").replace("+", "d")
    prefix = os.path.basename(folder).split("_")[0] # ID of molecule
    tag = f"q{str(charge).zfill(3)}_m{mult:02d}_k{state:02}"
    sub_folder = f"{prefix}_{tag}_{job_name.lower()}_{method.lower()}_{basisname}"

    dir_path = f'{folder}/{sub_folder}'
    fchk_path = f'{folder}/{sub_folder}/{sub_folder}.fchk'
    if not os.path.isdir(dir_path):
        raise AssertionError(f"Directory {dir_path} does not exist!")
    if not os.path.isfile(fchk_path):
        raise AssertionError(f"File {fchk_path} does not exist!")

    return os.path.abspath(fchk_path)


def get_data(molecule, method, basis=None):
    import os
    name = f'{molecule}/{molecule}_{method}'
    if not basis is None:
        basisname = basis.lower().replace("-", "").replace("*", "p").replace("+", "d")
        name += '_{basisname}'
    
    return os.path.abspath(name)


def get_rcoord(atcoords,atm1, atm2):
    d_coords = atcoords[atm1]-atcoords[atm2]
    return np.sqrt(np.dot(d_coords.T, d_coords))


def get_HHbond(atcoords, atm1, atm2):
    d_coords = np.array(atcoords[atm1][1:]) - np.array(atcoords[atm2][1:])
    return np.sqrt(np.dot(d_coords.T, d_coords))


def get_xyzfile(folder):
    dir_path = f'{folder}'
    xyz_path = f'{folder}/{folder}.xyz'
    if not os.path.isdir(dir_path):
        raise AssertionError(f"Directory {dir_path} does not exist!")
    if not os.path.isfile(xyz_path):
        raise AssertionError(f"File {xyz_path} does not exist!")

    return os.path.abspath(xyz_path)


def read_xyz(filename):
    with open(filename, 'r') as mol:
        lines = mol.readlines()
    content = [line.strip('\n') for line in lines[2:]]
    content = [line.split() for line in content]
    if content[-1] == []:
        content = content[:-1]
    coords = [(atm, float(x), float(y), float(z)) for atm, x,y,z in content]

    return coords


def get_folder_path(folder, job_name, method, basis, charge=0, mult=1, state=0):
    molid = os.path.basename(folder).split("_")[0] # ID of molecule
    basisname = basis.lower().replace("-", "").replace("*", "p").replace("+", "d")
    tag = f"q{str(charge).zfill(3)}_m{mult:02d}_k{state:02}"
    sub_folder = f"{molid}_{tag}_{job_name.lower()}_{method.lower()}_{basisname}"
    return f'{folder}/{sub_folder}'


def make_job_dir(folder, job_name, method, basis, charge=0, mult=1, state=0):
    dir_path = get_folder_path(folder, job_name, method, basis, charge=charge, mult=mult, state=state)
    print(dir_path)

    if os.path.isdir(dir_path):
        raise AssertionError(f"Directory {dir_path} already exists!")
    
    os.system(f'mkdir {dir_path}')
    # return dir_path


# def get_folder(folder, job_name, method, basis, charge=0, mult=1, state=0):    
#     dir_path = get_folder_path(folder, job_name, method, basis, charge=charge, mult=mult, state=state)
#     if not os.path.isdir(dir_path):
#         raise AssertionError(f"Directory {dir_path} doesn't exist!")
#     return dir_path

def from_spins(blocks):
    r"""
    Return a two- or four- index array in the spin representation from blocks.

    A two-index array is recontrcuted from blocks (a, b).
    A four-index array is recontrcuted from blocks (aa, ab, ba, bb).

    """
    if len(blocks) == 2:
        for b in blocks:
            if b.ndim != 2:
                raise ValueError("Input must have ndim == 2")
        n = blocks[0].shape[0]
        k = 2 * n
        y = np.zeros((k, k))
        y[:n, :n] = blocks[0]
        y[n:, n:] = blocks[1]
    elif len(blocks) == 4:
        for b in blocks:
            if b.ndim != 4:
                raise ValueError("Input must have ndim == 4")
        n = blocks[0].shape[0]
        k = 2 * n
        y = np.zeros((k, k, k, k))
        y[:n, :n, :n, :n] = blocks[0]
        y[:n, n:, :n, n:] = blocks[1]
        y[n:, :n, n:, :n] = blocks[2]
        y[n:, n:, n:, n:] = blocks[3]
    else:
        raise ValueError("Invalid input")
    return y


def solve_dense(lhs, rhs, tol=1.0e-7, orthog="symmetric", err="ignore"):
    r"""
    Solve the EOM eigenvalue system.

    Parameters
    ----------
    tol : float, optional
        Tolerance for small singular values. Default: 1.0e-10
    orthog : str, optional
        Matrix orthogonalization method. Default is symmetric orthogonalization
        in which the inverse square root of the right hand side matrix is taken.
    err : ("warn" | "ignore" | "raise")
        What to do if a divide-by-zero floating point error is raised.
        Default behavior is to ignore divide by zero errors.

    Returns
    -------
    w : np.ndarray((m,))
        Eigenvalue array (m eigenvalues).
    v : np.ndarray((m, n))
        Eigenvector matrix (m eigenvectors).

    """
    from scipy.linalg import eig, svd
    # Invert RHS matrix
    U, s, V = svd(rhs)
    if orthog == "symmetric":
        # Apply inverse square root to eigvals of RHS
        with np.errstate(divide=err):
            s = s ** (-0.5)
        # Check singular value threshold
        s[s >= 1 / tol] = 0.0
        # Transform back to RHS^(-0.5)
        S_inv = np.diag(s)
        rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))

        # Apply RHS^-0.5 * LHS * RHS^-0.5
        A = np.dot(rhs_inv, np.dot(lhs, rhs_inv))
        # Run scipy `linalg.eig` eigenvalue solver
        w, v = eig(A)
        # Transform back to original eigenvector matrix
        v = np.dot(rhs_inv, v)
        # Return w (eigenvalues)
        #    and v (eigenvector column matrix -- so transpose it!)
        return np.real(w), np.real(v.T)
    elif orthog == "asymmetric":
        # Check singular value threshold
        with np.errstate(divide=err):
            s = s ** (-1)
        s[s >= 1 / tol] = 0.0
        # S^(-1)
        S_inv = np.diag(s)
        # rhs^(-1)
        rhs_inv = np.dot(V.T, np.dot(S_inv, U.T))

        # Apply RHS^-1 * LHS
        A = np.dot(rhs_inv, lhs)
        # Run scipy `linalg.eig` eigenvalue solver
        w, v = eig(A)
        # Return w (eigenvalues)
        #    and v (eigenvector column matrix -- so transpose it!)
        return np.real(w), np.real(v.T)


def solve_lowdin(lhs, rhs, tol=1.0e-7, err="ignore"):
    w, v = np.linalg.eigh(rhs)
    w[w < 0.] = 0.
    with np.errstate(divide=err):
        inv_sqrt_w = w ** (-0.5)
    inv_sqrt_w[inv_sqrt_w > 1/tol] = 0.
    ort_m = np.dot(v, np.dot(np.diag(inv_sqrt_w), v.T))
    F_ = np.dot(ort_m.T, np.dot(lhs, ort_m))
    es, cv = np.linalg.eigh(F_)
    coeffs = np.dot(ort_m, cv)
    return es, coeffs.T


def solve_lowdin_img(lhs, rhs, tol=1.0e-7):
    from numpy.lib.scimath import sqrt as csqrt
    from scipy.linalg import pinv

    assert np.allclose(rhs, rhs.T)
    assert np.allclose(lhs, lhs.T)
    w, v = np.linalg.eigh(rhs)
    w[np.abs(w) < tol] = 0.
    sqrt_w = csqrt(w)
    inv_sqrt_w = pinv(np.diag(sqrt_w))
    ort_m = np.dot(v, np.dot(inv_sqrt_w, v.T))
    F_ = np.dot(ort_m.T, np.dot(lhs, ort_m))
    assert np.allclose(F_, F_.T)
    es, cv= np.linalg.eigh(F_)
    coeffs = np.dot(ort_m, cv)
    return np.real(es), np.real(coeffs.T)