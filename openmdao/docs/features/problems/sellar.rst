Sellar - Simple two discipline problem
=======================================

The first discipline is define by the following equation:

.. math::

    y_1(x, y_2, z_1, z_2) = z_1^2 + x_1 + z_2 - 0.2y_2

This is built as an openmdao :ref:`Component <usr_openmdao.core.component.py>` like this

.. embed-code::
    openmdao.test_suite.components.sellar.SellarDis1

----

The second discipline is given by another equation:

.. math::

  y_2(x, y_1, z_1, z_2) = \sqrt{y_1} + z_1 + z_2

which is translated into a :ref:`Component <usr_openmdao.core.component.py>` like this

.. embed-code::
    openmdao.test_suite.components.sellar.SellarDis2


----

The first discipline outputs :math:`y_1`, which is an input to the second discipline. Similarly, the second discipline outputs :math:`y_2` which an input to the first discipline. This interdependence causes a cycle that must be converged with a nonlinear solver to get a valid answer.

:math:`x` and :math:`z` are design variables so we define an optimization problem as follows:

.. math::

    \begin{align}
    \text{min}: & \ \ \ & x_1^2 + z_2 + y_1 + e^-{y_2} \\
    \text{w.r.t.}: & \ \ \ &  x_1, z_1, z_2 \\
    \text{subject to}: & \ \ \ & \\
    & \ \ \ & 3.16 - y_1 >=0 \\
    & \ \ \ & y_2 - 24.0 >=0
    \end{align}

.. embed-code::
    openmdao.test_suite.components.sellar.SellarNoDerivatives
