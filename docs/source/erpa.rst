..
    : This file is part of PyEOM.
    :
    : PyEOM is free software: you can redistribute it and/or modify it under
    : the terms of the GNU General Public License as published by the Free
    : Software Foundation, either version 3 of the License, or (at your
    : option) any later version.
    :
    : PyEOM is distributed in the hope that it will be useful, but WITHOUT
    : ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    : FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    : for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with PyEOM. If not, see <http://www.gnu.org/licenses/>.

Extended Random Phase Approximation
###################################

The Extended Random Phase Approximation (ERPA) method allows us to compute the correlation energy
of a system from a zeroth-order Hamiltonian and reduced density matrices (RDMs).

Method
======

Lets assume we know the exact wavefunction and energy of some model Hamiltonian :math:`\hat{H}^0`:

.. math::

    \hat{H}^{0} \left| \Phi_\nu \middle> = E^{0}_\nu \middle| \Phi_\nu \right>

which is a good approximation to the corresponding eigenstate :math:`\left| \Psi_\nu \right>` of
the exact Hamiltonian for the system, :math:`\hat{H}`:

.. math::

    \hat{H} \left| \Psi_\nu \middle> = E_\nu \middle| \Psi_\nu \right>

The goal is to render :math:`\left| \Phi_\nu \right>` and :math:`\hat{H}^0` fully correlated and
to evaluate the additional correlation energy recovered (here we will mainly be interested in the
ground-satate, :math:`\nu = 0`).

The basic idea is that we have a Hamiltonian:

.. math::

    \hat{H}^{\alpha} = \sum_{pq} h^{\alpha}_{pq} a^{\dagger}_p a_q + \frac{1}{2} \sum_{pqrs}
        v^{\alpha}_{pqrs} a^{\dagger}_p a^{\dagger}_q a_s a_r

with eigenstate:

.. math::

    \hat{H}^{\alpha} \left| \Psi^\alpha_0 \middle> = E^{\alpha}_0 \middle| \Psi^\alpha_0
        \right>

and :math:`\hat{H}^\alpha` and :math:`\left| \Psi^\alpha_0 \right>` can be perturbed
along some arbitrary parameter :math:`\alpha` from a zeroth order state (:math:`\alpha = 0`) to a
fully correlated state (:math:`\alpha = 1`):

.. math::

    E^{\alpha=1}_0 - E^{\alpha=0}_0 = \int_{0}^{1} { \frac{ \partial E^{\alpha}_0 }{ \partial \alpha}
    d\alpha }

Therefore, at :math:`\alpha = 0` we have :math:`\hat{H}^{\alpha=0}=\hat{H}^0`, :math:`\left|
\Psi^{\alpha=0}_0 \right>=\left| \Phi_0 \right>` and :math:`\hat{H}^{\alpha=1}=\hat{H}`,
:math:`\left| \Psi^{\alpha=1}_0 \right>=\left| \Psi_0 \right>` at :math:`\alpha = 1`.

From the Hellmann-Feynman theorem we have:

.. math::

    \frac{ \partial E^{\alpha}_0 }{ \partial \alpha} &= \left< \Psi^\alpha_0 \middle| \frac{ \partial
        \hat{H}^{\alpha} }{ \partial \alpha} \middle| \Psi^\alpha_0 \right> \\ &= \left<
        \Psi^\alpha_0 \middle| \sum_{pq} \frac{ \partial h^{\alpha}_{pq} }{ \partial \alpha} 
    a^{\dagger}_p a_q + \frac{1}{2} \sum_{pqrs} \frac{ \partial v^{\alpha}_{pqrs} }{ \partial
    \alpha} a^{\dagger}_p a^{\dagger}_q a_s a_r \middle| \Psi^\alpha_0 \right> \\ 
        &= \sum_{pq} \frac{ d h^{\alpha}_{pq} }{ d \alpha} \gamma^{\alpha}_{pq} + \frac{1}{2}
    \sum_{pqrs} \frac{ d v^{\alpha}_{pqrs} }{ d \alpha} \Gamma^{\alpha}_{pqrs}

where the standard definitions:

.. math::

    \gamma^{\alpha}_{pq} = \left< \Psi^\alpha_0 \middle| a^{\dagger}_p a_q \middle| \Psi^\alpha_0
        \right> \\ 
    \Gamma^{\alpha}_{pqrs} = \left< \Psi^\alpha_0 \middle| a^{\dagger}_p a^{\dagger}_q
        a_s a_r \middle| \Psi^\alpha_0 \right>

were used for the 1- and 2-electron reduced density matrices (RDMs) for the eigenstate.

We only know :math:`\gamma^\alpha` and :math:`\Gamma^\alpha` at :math:`\alpha = 0`. If the
eigenstate is a good model of static correlation, we can assume an adiabatic connection along
:math:`\alpha` for the 1-electron RDM (see `K. Pernal 2017 <https://doi.org/10.1002/qua.25462>`_),

.. math::

    \gamma^{\alpha}_{pq} = \gamma^{\alpha=0}_{pq} , \forall 0 \le \alpha \le 1

and use McLachlan and Ball's ERPA relation for :math:`\Gamma^\alpha` for the 1-RDM and transition
1-RDMs, constructed via resolution of the identity with arbitrary states. We'll consider three
possible resolutions:

* hole-particle resolution:

.. math::

    \Gamma^{\alpha}_{pqrs} &= \left< \Psi^\alpha_0 \middle| a^{\dagger}_p a^{\dagger}_q a_s a_r
        \middle| \Psi^\alpha_0 \right> \\ &= \delta_{sq} \gamma^\alpha_{pr} - \left< \Psi^\alpha_0
    \middle| a^{\dagger}_p a_s a^{\dagger}_q a_r \middle| \Psi^\alpha_0 \right> \\ 
        &= \delta_{sq} \gamma^\alpha_{pr} - \sum^\infty_{\nu=0} \left< \Psi^\alpha_0 \middle|
    a^{\dagger}_p a_s \middle| \Psi^\beta_\nu \middle> \middle< \Psi^\beta_\nu \middle|
    a^{\dagger}_q a_r \middle| \Psi^\alpha_0 \right>

* hole-hole resolution:

.. math::

    \Gamma^{\alpha}_{pqrs} &= \left< \Psi^\alpha_0 \middle| a^{\dagger}_p a^{\dagger}_q a_s a_r
        \middle| \Psi^\alpha_0 \right> \\ 
        &= \sum^\infty_{\nu=0} \left< \Psi^\alpha_0 \middle|
    a^{\dagger}_p a^{\dagger}_q \middle| \Psi^\beta_{N-2;\nu} \middle> \middle< \Psi^\beta_{N-2;\nu}
    \middle| a_s a_r \middle| \Psi^\alpha_0 \right>

* particle-particle resolution:

.. math::

    \Gamma^{\alpha}_{pqrs} &= \left< \Psi^\alpha_0 \middle| a^{\dagger}_p a^{\dagger}_q a_s a_r
        \middle| \Psi^\alpha_0 \right> \\ 
        &= \delta_{pr} \gamma^\alpha_{qs} + \delta_{qs} \gamma^\alpha_{pr} 
    - \delta_{ps} \gamma^\alpha_{qr} - \delta_{qr} \gamma^\alpha_{ps} - \delta_{pr} \delta_{qs} +
    \delta_{ps} \delta_{qr} + \left< \Psi^\alpha_0 \middle| a_s a_r
    a^{\dagger}_p a^{\dagger}_q \middle| \Psi^\alpha_0 \right> \\ 
        &= \delta_{pr} \gamma^\alpha_{qs} + \delta_{qs} \gamma^\alpha_{pr} 
    - \delta_{ps} \gamma^\alpha_{qr} - \delta_{qr} \gamma^\alpha_{ps} - \delta_{pr} \delta_{qs} 
    + \delta_{ps} \delta_{qr} \\ 
        &+ \sum^\infty_{\nu=0} \left< \Psi^\alpha_0 \middle| a_s a_r \middle| \Psi^\beta_{N+2;\nu}
    \middle> \middle< \Psi^\beta_{N+2;\nu} \middle| a^{\dagger}_p a^{\dagger}_q 
    \middle| \Psi^\alpha_0 \right>

For an interacting Hamiltonian, it is fine to set :math:`\beta = \alpha`. :math:`\Gamma^\alpha` can
then be evaluated from transition 1-RDMs, which can be computed via Rowe's Equation of Motion
method. Finally, with :math:`\Gamma^{\alpha=1}` approximated, we can transform the Hamiltonian
:math:`\hat{H}^{\alpha = 0}` to :math:`\hat{H}^{\alpha = 1}`, in order to evaluate the additional
correlation energy:

* hole-particle resolution:

.. math::

    \frac{ \partial E^{\alpha} }{ \partial \alpha} = \sum_{pq} \frac{ d h^{\alpha}_{pq} }{ d \alpha}
        \gamma^{\alpha=0}_{pq} + \frac{1}{2} \sum_{pqrs} \frac{ d v^{\alpha}_{pqrs} }{ d \alpha} 
    \left( \delta_{sq} \gamma^{\alpha=0}_{pr} - \gamma^{\alpha=0}_{ps} \gamma^{\alpha=0}_{qr} -
    \sum^\infty_{\nu=1} \gamma^{\alpha}_{ps;0\nu} \gamma^{\alpha}_{qr;0\nu} \right)

.. math::

    E^{\alpha=1} - E^{\alpha=0} &= \sum_{pq} (h^{\alpha=1}_{pq} - h^{\alpha=0}_{pq})
        \gamma^{\alpha=0}_{pq} \\ 
        &+ \frac{1}{2} \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{pqrs})
    ( \delta_{pr} \gamma^{\alpha=0}_{qs} + \delta_{qs} \gamma^{\alpha=0}_{pr} - \delta_{ps}
    \gamma^{\alpha=0}_{qr} - \delta_{qr} \gamma^{\alpha=0}_{ps} - \delta_{pr} \delta_{qs} +
    \delta_{ps} \delta_{qr})\\ 
        &+ \frac{1}{2} \int_{0}^{1} \sum_{pqrs} \frac{ d v^{\alpha}_{pqrs} }{ d \alpha} 
    \left( \sum^\infty_{\nu=1} \gamma^{\alpha}_{ps;0\nu} \gamma^{\alpha}_{qr;0\nu} \right) d \alpha

* hole-hole resolution:

.. math::

    E^{\alpha=1} - E^{\alpha=0} = \sum_{pq} (h^{\alpha=1}_{pq} - h^{\alpha=0}_{pq})
        \gamma^{\alpha=0}_{pq} + \frac{1}{2} \int_{0}^{1} \sum_{pqrs} \frac{ d v^{\alpha}_{pqrs} }
    { d \alpha} \left( \sum^\infty_{\nu=0} \gamma^{\alpha}_{pq;0\nu} \gamma^{\alpha}_{sr;0\nu}
    \right) d \alpha

* particle-particle resolution:

.. math::

    E^{\alpha=1} - E^{\alpha=0} &= \sum_{pq} (h^{\alpha=1}_{pq} - h^{\alpha=0}_{pq})
        \gamma^{\alpha=0}_{pq} + \frac{1}{2} \sum_{pqrs} (v^{\alpha=1}_{pqrs} - v^{\alpha=0}_{pqrs})
    (\delta_{sq} \gamma^{\alpha=0}_{pr} - \gamma^{\alpha=0}_{ps} \gamma^{\alpha=0}_{qr})\\ 
        &-\frac{1}{2} \int_{0}^{1} \sum_{pqrs} \frac{ d v^{\alpha}_{pqrs} }{ d \alpha} \left(
    \sum^\infty_{\nu=0} \gamma^{\alpha}_{sr;0\nu} \gamma^{\alpha}_{pq;0\nu} \right) d \alpha