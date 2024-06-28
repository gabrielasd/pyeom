# This file is part of EOMEE.
#
# EOMEE is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# EOMEE is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with EOMEE. If not, see <http://www.gnu.org/licenses/>.

r"""Ionization Potential EOM state class."""


import numpy as np

from .base import EOMState


__all__ = [
    "IP",
    "IPc",
    "IPa",
    "IPcm",
    "IPam",
]


class IP(EOMState):
    r"""Ionization potential EOM or Extended Koopman's Theorem class ([EKT]_).

    The :math:`(N-1)`-electron wavefunction and ionization energies are obtained by solving the generalized
    eigenvalue problem:

    .. math::

        &\mathbf{A} \mathbf{C}_\lambda = \Delta_\lambda \mathbf{U} \mathbf{C}_\lambda

    where the matrices :math:`\mathbf{A}` and :math:`\mathbf{U}` are defined as:

    .. math::

        A_{m,n} &= \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m \left[\hat{H}, a_n \right] \middle| \Psi^{(N)}_0 \right>

        U_{m,n} &= \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m a_n \middle| \Psi^{(N)}_0 \right>

    These matrices can be built from the ground state's, :math:`\Psi^{(N)}_0`, one- and two-electron
    reduced density matrices. The negative of :math:`\mathbf{A}` gives the generalized Fock matrix and
    :math:`\mathbf{U}` is the one-electron density matrix. Their dimensions correspond to the number 
    of spin-orbitals in the basis set :math:`n`.

    The eigenvectors, :math:`\mathbf{C}_\lambda` determine the best linear combination of anihilation
    operators `a_n` that produce the :math:`\lambda`th ionized state from the ground state:

    :math:`| \Psi^{(N-1)}_\lambda > = \sum_n { c_{n;\lambda} a_n} | \Psi^{(N)}_0 >`

    .. [EKT] O. W. Day, D. W. Smith, and C. Garrod, Int. J. Quantum Chem., Symp. 8, 501 (1974).

    Example
    -------
    >>> ekt = eomee.IP(h, v, dm1, dm2)
    >>> ekt.neigs # number of solutions
    >>> ekt.lhs # left-hand-side matrix
    >>> # solve the generalized eigenvalue problem
    >>> ekt.solve_dense()

    """

    @property
    def neigs(self):
        r"""
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        # Number of q_n terms = n_{\text{basis}}
        return self._n

    def _compute_lhs(self):
        r"""
        Compute

        .. math::

            A_{mn} = \sum_q { -h_{nq} \gamma_{mq} } - 0.5 \sum_{qrs} { \left< nq||rs \right> \Gamma_{mqrs} }.

        """
        # A_mn = -h_nq \gamma_mq - 0.5 <v_nqrs> \Gamma_mqrs
        a = np.dot(self._dm1, self._h)
        b = np.tensordot(self._dm2, self._v, ((1, 2, 3), (1, 2, 3)))
        b *= 0.5
        b += a
        return -b

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{mn} = \gamma_{mn}`.

        """
        return np.copy(self._dm1)
    
    def normalize_eigvect(self, coeffs):
        r""" Normalize coefficients vector. """
        if not coeffs.shape[0] == self.neigs:
            raise ValueError("Coefficients vector has the wrong shape, expected {self.neigs}, got {coeffs.shape[0]}.")
        norm_factor = np.dot(coeffs, np.dot(self.rhs, coeffs.T))
        sqr_n = np.sqrt(np.abs(norm_factor))
        return (coeffs.T / sqr_n).T
    
    def compute_td(self, coeffs):
        r"""
        Compute the transition density matrix.

        .. math::
        < \Psi^{(N)}_0 | a^\dagger_p | \Psi^{(N - 1)}_\lambda > = \sum_{q} \gamma_{pq} c_{q;\lambda}

        Parameters
        ----------
        coeffs : np.ndarray(n)
            Coefficients vector for the lambda-th ionized state.
        
        Returns
        -------
        tdm : np.ndarray(n)
            transition DMs.

        """
        if not coeffs.shape[0] == self.neigs:
            raise ValueError("Coefficients vector has the wrong shape, expected {self.neigs}, got {coeffs.shape[0]}.")
        return np.einsum("pq,q->p", self.rhs, coeffs)


class IPc(EOMState):
    r"""Ionizatio npotential class with double commutator EOM equation.

    The :math:`(N-1)`-electron wavefunction is given by:

    :math:`| \Psi^{(N-1)}_\lambda > = \sum_n { c_{n;\lambda} a_n} | \Psi^{(N)}_0 >`

    The ionization energies and wavefunction coefficients are found solving the matrix equation:

    .. math::

        &\mathbf{A} \mathbf{C}_\lambda = \Delta_\lambda \mathbf{U} \mathbf{C}_\lambda

    where the matrices :math:`\mathbf{A}` and :math:`\mathbf{U}` are defined as:

        A_{m,n} &= \left< \Psi^{(N)}_0 \middle| \left[ a^{\dagger}_m, \left[\hat{H}, a_n \right]\right]\middle| \Psi^{(N)}_0 \right>

        U_{m,n} &= \left< \Psi^{(N)}_0 \middle| \left[a^{\dagger}_m, a_n \right] \middle| \Psi^{(N)}_0 \right>

    and can be built from the ground state's, :math:`\Psi^{(N)}_0`, one- and two-electron reduced density matrices.

    """

    @property
    def neigs(self):
        r"""
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        # Number of q_n terms = n_{\text{basis}}
        return self._n

    def _compute_lhs(self):
        r"""
        Compute

        .. math::

            A_{mn} = h_{nm} -2 \sum_q { h_{nq} \gamma_{mq} } + \sum_{qs} { \left< nq||ms \right> \gamma_{qs} }
            + \sum_{qrs} { \left< nq||rs \right> \Gamma_{mqsr} }.

        """
        # A_mn = h_mn - 2 * \gamma_mq * h_qn
        a = np.copy(self._h)
        a -= 2 * np.dot(self._dm1, self._h)
        # A_mn += <v_msnq> \gamma_sq - \Gamma_mqrs <v_nqrs>
        a += np.einsum("msnq,sq", self._v, self._dm1, optimize=True)
        a -= np.einsum("mqrs,nqrs", self._dm2, self._v, optimize=True)
        return a

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{mn} = 2 \gamma_{mn} - \delta_{nm}`.

        """
        m = 2 * np.copy(self._dm1)
        m -= np.eye(self._n, dtype=self._dm1.dtype)
        return m
    
    def normalize_eigvect(self, coeffs):
        r""" Normalize coefficients vector. """
        if not coeffs.shape[0] == self.neigs:
            raise ValueError("Coefficients vector has the wrong shape, expected {self.neigs}, got {coeffs.shape[0]}.")
        norm_factor = np.dot(coeffs, np.dot(self.rhs, coeffs.T))
        sqr_n = np.sqrt(np.abs(norm_factor))
        return (coeffs.T / sqr_n).T
    
    def compute_td(self, coeffs):
        r"""
        Compute the transition density matrix.

        .. math::
        < \Psi^{(N)}_0 | a^\dagger_p | \Psi^{(N - 1)}_\lambda >

        Parameters
        ----------
        coeffs : np.ndarray(n)
            Coefficients vector for the lambda-th ionized state.
        
        Returns
        -------
        tdm : np.ndarray(n)
            transition DMs.

        """
        if not coeffs.shape[0] == self.neigs:
            raise ValueError("Coefficients vector has the wrong shape, expected {self.neigs}, got {coeffs.shape[0]}.")
        return np.einsum("pq,q->p", self.rhs, coeffs)


class IPcm(IPc):
    r"""
    Ionizatio npotential class with double commutator on the left-hand side of the EOM equation
    and none on the right-hand side (mixed double commutator EOM form, IPcm).

    The elements of the left-hand and right-hand side matrices are given by:

    .. math::

        A_{m,n} &= \left< \Psi^{(N)}_0 \middle| \left[ a^{\dagger}_m, \left[\hat{H}, a_n \right]\right]\middle| \Psi^{(N)}_0 \right>

        U_{m,n} &= \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m a_n \middle| \Psi^{(N)}_0 \right>

    """

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{mn} = \gamma_{mn}`.

        """
        return self._dm1


class IPa(EOMState):
    r"""Ionization potential class with anticommutator EOM equation.

    The ionization energies and wavefunction coefficients are found solving the matrix equation:

    .. math::

        &\mathbf{A} \mathbf{C}_\lambda = \Delta_\lambda \mathbf{U} \mathbf{C}_\lambda

    where the matrices :math:`\mathbf{A}` and :math:`\mathbf{U}` are defined as:

        A_{m,n} &= \left< \Psi^{(N)}_0 \middle| \Big\{ a^{\dagger}_m, \left[\hat{H}, a_n \right]\Big\} \middle| \Psi^{(N)}_0 \right>

        U_{m,n} &= \left< \Psi^{(N)}_0 \middle| \Big\{a^{\dagger}_m, a_n \Big\} \middle| \Psi^{(N)}_0 \right>

    These matrices only require the one-electron reduced density matrix of the ground state, :math:`\Psi^{(N)}_0`.

    """

    @property
    def neigs(self):
        r"""
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        # Number of q_n terms = n_{\text{basis}}
        return self._n

    def _compute_lhs(self):
        r"""
        Compute :math:`A_{mn} = -h_{nm} + \sum_{qr} { \left< qn||mr \right> \gamma_{qr} }`.

        """
        # A_mn = -h_mn
        a = -np.copy(self._h)
        # A_mn += <v_qnmr> \gamma_qr
        a += np.einsum("qnmr,qr", self._v, self._dm1, optimize=True)
        return a

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{mn} = \delta_{mn}`.

        """
        m = np.eye(self._n, dtype=self._dm1.dtype)
        return m
    
    def normalize_eigvect(self, coeffs):
        r""" Normalize coefficients vector. """
        if not coeffs.shape[0] == self.neigs:
            raise ValueError("Coefficients vector has the wrong shape, expected {self.neigs}, got {coeffs.shape[0]}.")
        norm_factor = np.dot(coeffs, np.dot(self.rhs, coeffs.T))
        sqr_n = np.sqrt(np.abs(norm_factor))
        return (coeffs.T / sqr_n).T
    
    def compute_td(self, coeffs):
        r"""
        Compute the transition density matrix.

        .. math::
        < \Psi^{(N)}_0 | a^\dagger_p | \Psi^{(N - 1)}_\lambda >

        Parameters
        ----------
        coeffs : np.ndarray(n)
            Coefficients vector for the lambda-th ionized state.
        
        Returns
        -------
        tdm : np.ndarray(n)
            transition DMs.

        """
        if not coeffs.shape[0] == self.neigs:
            raise ValueError("Coefficients vector has the wrong shape, expected {self.neigs}, got {coeffs.shape[0]}.")
        return np.einsum("pq,q->p", self.rhs, coeffs)


class IPam(IPa):
    r"""
    Ionizatio npotential class with anticommutator on the left-hand side of the EOM equation
    and none on the right-hand side (mixed anticommutator EOM form, IPam).

    The elements of the left-hand and right-hand side matrices are given by:

    .. math::

        A_{m,n} &= \left< \Psi^{(N)}_0 \middle| \Big\{ a^{\dagger}_m, \left[\hat{H}, a_n \right]\Big\} \middle| \Psi^{(N)}_0 \right>

        U_{m,n} &= \left< \Psi^{(N)}_0 \middle| a^{\dagger}_m a_n \middle| \Psi^{(N)}_0 \right>

    """

    @property
    def neigs(self):
        r"""
        Return the size of the eigensystem.

        Returns
        -------
        neigs : int
            Size of eigensystem.

        """
        # Number of q_n terms = n_{\text{basis}}
        return self._n

    def _compute_rhs(self):
        r"""
        Compute :math:`M_{mn} = \gamma_{mn}`.

        """
        return self._dm1


EKT = IP
