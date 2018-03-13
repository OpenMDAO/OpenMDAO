.. _`hohmann_tutorial`:

Hohmann Transfer Example - Optimizing a Spacecraft Manuever
===========================================================

This example will demonstrate the use of OpenMDAO for optimizing
a simple orbital mechanics problem.  We seek the minimum possible
delta-V to transfer a spacecraft from Low Earth Orbit (LEO) to
geostationary orbit (GEO) using a two-impulse *Hohmann Transfer*.

The Hohmann Transfer is a maneuver which minimizes the delta-V for
transferring a spacecraft from one circular orbit to another.  Hohmann
transfers have a practical application in that they can be used
to transfer satellites from LEO parking orbits to geostationary orbit.

To do so, the vehicle first imparts a delta-V along the velocity vector
while in LEO.  This boosts apogee radius to the radius of the geostationary
orbit (42164 km).  In this model we will model this delta-V as an *impulsive*
maneuver which changes the spacecraft's velocity instantaneously.

We will assume that the first impulse is performed at the
ascending node in LEO.  Thus perigee of the transfer orbit is coincident
with the ascending node of the transfer orbit.  Apogee of the transfer orbit
is thus coincident with the descending node, where we will perform the
second impulse.

After the first impulse, the spacecraft coasts to apogee.  Once there
it performs a second impulsive burn along the velocity vector to raise perigee radius
to the radius of GEO, thus circularizing the orbit.

Simple, right?  The issue is that, unless they launch from the equator,
launch vehicles do not put satellites in a low Earth parking orbit
with the same inclination as geostationary orbit.  For instance, a due-east launch
from Kennedy Space Center will result in a parking orbit with an inclination of
28.5 degrees.  We therefore need to change the inclination of our satellite during
its two impulsive burn maneuvers.  The question is, *what change in inclination
at each burn will result in the minimum possible delta-V?*

.. figure:: images/hohmann_transfer.png
   :align: center
   :width: 500 px
   :alt: An inclined Hohmann Transfer diagram

   An inclined Hohmann Transfer diagram

The trajectory optimization problem can thus be stated as:

.. math::
    Minimize  J=\Delta V

    s.t.

    \Delta i_1 + \Delta i_2 = 28.5^o

The total :math:`\Delta V` is the sum of the two impulsive :math:`\Delta Vs`.

.. figure:: images/hohmann_dv1.png
   :align: center
   :width: 500 px
   :alt: Vector diagram of the first impulsive :math:`\Delta V`

   Vector diagram of the first impulsive :math:`\Delta V`

The component of the :math:`\Delta V` in the orbital plane is along the
local horizontal plane.  The orbit-normal component is in the
direction of the desired inclination change.  Knowing the
velocity magnitude before (:math:`v_c`) and after (:math:`v_p`) the impulse, and the
change in inclination due to the impulse (:math:`\Delta i`), the :math:`\Delta V`
is then computed from the law of cosines:

.. math::
    \Delta V_1 = v_c^2 + v_p^2 - 2 v_c v_p \cos{\Delta i}

In the first impulse, :math:`v_1` is the circular velocity in LEO.  In
this case :math:`v_c` refers to the circular velocity in geostationary
orbit, and :math:`v_a` is the velocity at apogee of the transfer
orbit.

We can compute the circular velocity in either orbit from
the following equation:

.. math::
    v_c = \sqrt{\mu/r}

where :math:`\mu` is the gravitational parameter of the Earth
and :math:`r` is the distance from the center of the Earth.

The velocity after the first impulse is the periapsis velocity
of the transfer orbit.  This can be solved for based on what we
know about the orbit.

The specific angular momentum of the transfer orbit is constant.
At periapsis, it is simply the product of the velocity and radius.
Therefore, rearranging we have:

.. math::
    v_p = \frac{h}{r_p}

The specific angular momentum can also be computed as:

.. math::
    h = \sqrt{p \mu}

Where :math:`p` is the semilatus rectum of the orbit and :math:`\mu` is
the gravitational parameter of the central body.

The semilatus rectum is computed as:

.. math::

    p = a*(1.0-e^2)


Where :math:`a` and :math:`e` are the semi-major axis and eccentricity of the transfer orbit, respectively.
Since we know :math:`r_a` and :math:`r_p` of the transfer orbit, it's semimajor axis is simply:

.. math::

    e = (a-r_p)/a

The eccentricity is known by the relationship of :math:`a` and :math:`e` to :math:`r_p` (or :math:`r_a`):

.. math::

    a = (r_a+r_p)/2.0

Thus we can compute periapsis velocity based on the periapsis and apoapsis
radii of the transfer orbit, and the gravitational parameter of the central body.

For the second impulse, the final velocity is the circular velocity of the
final orbit, which can be computed in the same way as the circular velocity
of the initial orbit.  The initial velocity at the second impulse is the
apoapsis velocity of the transfer orbit, which is:

.. figure:: images/hohmann_dv2.png
   :align: center
   :width: 500 px
   :alt: Vector diagram of the second impulsive :math:`\Delta V`

   Vector diagram of the second impulsive :math:`\Delta V`

.. math::

    \Delta V = \sqrt{ v_a^2 + v_c^2 - 2 v_a v_c \cos{\Delta i} }

.. math::

    v_a = \frac{h}{r_a}

Having already computed the specific angular momentum of the transfer orbit, this is
easily computed.

Finally we have the necessary calculations to compute the :math:`\Delta V` of the Hohmann
transfer with a plane change.


~~~~~~~~~~
Components
~~~~~~~~~~

The first component we define computes the circular velocity given the
radius from the center of the central body and the gravitational parameter
of the central body.

.. literalinclude:: ../../../test_suite/test_examples/test_hohmann_transfer.py
   :pyobject: VCircComp

The transfer orbit component computes the velocity magnitude at periapsis
and apoapsis of an orbit, given the radii of periapsis and apoapsis, and
the gravitational parameter of the central body.

.. literalinclude:: ../../../test_suite/test_examples/test_hohmann_transfer.py
   :pyobject: TransferOrbitComp

The delta-V component is used to compute the delta-V performed in changing
the velocity vector, giving the magnitudes of the initial and final velocities
and the angle between them.

.. literalinclude:: ../../../test_suite/test_examples/test_hohmann_transfer.py
   :pyobject: DeltaVComp

~~~~~~~~~~~~~~~~~~~~~~~
Putting it all together
~~~~~~~~~~~~~~~~~~~~~~~

Now we assemble the model for our problem.

An IndepVarComp provides values for the gravitational parameter, the radii
of the two circular orbits, and the delta-V to be performed at each of the
two impulses.

Next, two instances of VCircComp are used to compute the velocity of the
spacecraft in the initial and final circular orbits.

The TransferOrbitComp is used to compute the periapsis and apoapsis velocity
of the spacecraft in the transfer orbit.

Now we can use the DeltaVComp to provide the magnitude of the delta-V
at each of the two impulses.

Lastly, we use two ExecComps to provide some simple calculations.  One
sums the delta-Vs of the two impulses to provide the total delta-V of the
transfer.  We will use this as the objective for the optimization.

The other ExecComp sums up the inclination change at each impulse.  We
will provide this to the driver as a constraint to ensure that our total
inclination change meets our requirements.

We will use the initial and final radii of the orbits, and the inclination
change at each of the two impulses as our design variables.

To run the model, we provide values for the design variables and invoke `run_model`.

To find the optimal solution for the model, we invoke `run_driver`, where we have
defined the driver of the problem to be the :ref:`ScipyOptimizeDriver <scipy_optimize_driver>`.

.. embed-code::
    openmdao.test_suite.test_examples.test_hohmann_transfer.TestHohmannTransfer.test_dv_at_apogee
    :layout: interleave

~~~~~~~
Summary
~~~~~~~

We built a model representing a Hohmann transfer with a plane change.  This model
utilized components with both analytic partial derivatives and approximated partials
using finite differencing.  We utilized ExecComps for some simple calculations
to reduce the amount of code we needed to write.  Finally, we used this model
to demonstrate that performing the necessary plane change entirely at apoapsis is
somewhat less optimal, from a delta-V standpoint, than performing some of the plane
change at the first impulse.
