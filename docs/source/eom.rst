..
    : This file is part of EOMEE.
    :
    : EOMEE is free software: you can redistribute it and/or modify it under
    : the terms of the GNU General Public License as published by the Free
    : Software Foundation, either version 3 of the License, or (at your
    : option) any later version.
    :
    : EOMEE is distributed in the hope that it will be useful, but WITHOUT
    : ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    : FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    : for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with EOMEE. If not, see <http://www.gnu.org/licenses/>.

Equations-of-Motion
###################

Method
======

Under the Equation-of-motion (EOM) method, the wavefunction of an :math:`(N \pm K)`-electron system
in the kth excited state :math:`\left| \Psi^{(N \pm K)}_k \right>` is given by:

.. math::

    \left| \Psi^{(N \pm K)}_k \right> &= \hat{Q}^{\pm K}_k \left| \Psi^{(N)}_0 \right>\\
        &= \left| \Psi^{(N \pm K)}_k \middle> \middle< \Psi^{(N)}_0 \middle| \Psi^{(N)}_0 \right>

where the transition operator :math:`\hat{Q}^{\pm K}_k` takes the system from the reference state
:math:`\left| \Psi^{(N)}_k \right>` (usually taken as the ground state, :math:`k=0`) to the excited
state, with a possible change in the number of particles from :math:`N`-electrons to
:math:`N \pm K`.

The Schrödinger equation for each state is:

.. math::

    \hat{H} \hat{Q}^{\pm K}_k \left| \Psi^{(N)}_0 \right>
        &= E^{N \pm K}_k \hat{Q}^{\pm K}_k \left| \Psi^{(N)}_0 \right>\\
    \hat{H} \left| \Psi^{(N)}_0 \right> &= E^{N}_0 \left| \Psi^{(N)}_0 \right>

By multiplying the second expression by :math:`\hat{Q}^{\pm K}_k` and subtracting, one is left with
an equation that allows to directly solve for the transition energy
(:math:`\Delta_k = E^{N \pm K}_k - E^{N}_0`):

.. math::

    \hat{H} \hat{Q}^{\pm K}_k \left| \Psi^{(N)}_0 \right>
        - \hat{Q}^{\pm K}_k \hat{H} \left| \Psi^{(N)}_0 \right>
        &= \Delta_k \hat{Q}^{\pm K}_k \left| \Psi^{(N)}_0 \right>\\
    \left[\hat{H}, \hat{Q} \right] \left| \Psi^{(N)}_0 \right>
        &= \Delta_{k} \hat{Q}^{\pm K}_k \left| \Psi^{(N)}_0 \right>

This expression is preferable as one can gain the advantages of systematic cancellation of errors
(since the energy differences are approximated directly) and the operator
:math:`\left[\hat{H}, \hat{Q} \right]` will usually be a simpler (fewer-body) operator than
:math:`\hat{H} \hat{Q}^{\pm K}_k`.

To find the transition energies and  :math:`\left| \Psi^{(N \pm K)}_k \right>` that satisfy the
equation above one can project on a set of arbitrary states, however because we are interested in
working in terms of the reduced density matrices, it is convenient to select those defined in terms
of the basis set that expands :math:`\hat{Q}^{\pm K}_k`:

.. math::
    \hat{Q}^{\pm K}_k = \sum_n c^{\pm K}_{n;k} \hat{q}^{\pm K}_n

so that one gets the EOM in its simplest form, stated as a generalized eigenvalue problem:

.. math::

    \left< \Psi^{(N)}_0 \middle|
            \left( {q^{\pm K}_m} \right)^{\dagger} \left[\hat{H}, \hat{Q}^{\pm K}_k \right]
        \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle|
            \left( {q^{\pm K}_m} \right)^{\dagger} \hat{Q}^{\pm K}_k
        \middle| \Psi^{(N)}_0 \right> \forall m

Notice that the transition operator should satisfy:

.. math::
    (\hat{Q}^{\pm K}_k)^\dagger \left| \Psi^{(N)}_0 \right> = 0

which is known as the "killer condition". This allows to add/subtract a zero term:

.. math::
    \left< \Psi^{(N)}_0 \middle| \hat{Q}^{\pm K}_k
        = \middle< \Psi^{(N)}_0 \right| \hat{H} \hat{Q}^{\pm K}_k = 0

to/from the EOM equation above and get the alternative formulations:

.. math::

    \left< \Psi^{(N)}_0 \middle| \left[
            \left( {q^{\pm K}_m} \right)^{\dagger}, \left[\hat{H}, \hat{Q}^{\pm K}_k \right]
        \right]_{\pm} \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle|
            \left( {q^{\pm K}_m} \right)^{\dagger} \hat{Q}^{\pm K}_k
        \middle| \Psi^{(N)}_0 \right> \forall m\\
    \left< \Psi^{(N)}_0 \middle| \left[
            \left( {q^{\pm K}_m} \right)^{\dagger}, \left[\hat{H}, \hat{Q}^{\pm K}_k \right]
        \right]_{\pm} \middle| \Psi^{(N)}_0 \right>
        = \Delta_{k} \left< \Psi^{(N)}_0 \middle| \left[
            \left( {q^{\pm K}_m} \right)^{\dagger}, \hat{Q}^{\pm K}_k
        \right]_{\pm} \middle| \Psi^{(N)}_0 \right> \forall m

Provided the exact :math:`\hat{Q}^{\pm K}_k` and :math:`\left| \Psi^{(N \pm K)}_k \right>` were
known, this formulations are exact. However, in practice, a finite basis set approximation of
:math:`\hat{Q}^{\pm K}_k` is used, being the most common choises:

.. math::
    \hat{Q}^{-1}_k &= \sum_n c_{n;k} a_n\\
    \hat{Q}^{+1}_k &= \sum_n c_{n;k} a^\dagger_n\\
    \hat{Q}^{0}_k &= \sum_n c_{pq;k} a^\dagger_p a_q\\
    \hat{Q}^{-2}_k &= \sum_n c_{pq;k} a_p a_q\\
    \hat{Q}^{+2}_k &= \sum_n c_{pq;k} a^\dagger_p a^\dagger_q\\

With these definitions the equations above only require the one- and two-reduced density matrices.

Finally, different wavefunction ansätze can be used to define :math:`\left| \Psi^{(N)}_0 \right>`,
though traditionally, a single Slater determinant has been used.
