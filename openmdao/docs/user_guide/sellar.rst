.. _sellar:

****************************************
Sellar - A Simple Two-Discipline Problem
****************************************

In previous tutorials, you built and optimized models comprised of only a single component.
Now, we'll work through a slightly more complex problem that involves two disciplines and hence two components.
In this tutorial, you'll learn how to group components together into a larger model and how to use
different kinds of nonlinear solvers to converge multidisciplinary models with coupling between components.

The Sellar problem is a really simple two discipline toy problem with each discipline described by a single
equation. The output of each component feeds into the input of the other, which creates a coupled model that needs to
be converged in order for the outputs to be valid.

----

Building the Components
****************************************



The first discipline is defined by the following equation:

.. math::

    y_1(x, y_2, z_1, z_2) = z_1^2 + x_1 + z_2 - 0.2y_2

This is built as an openmdao :ref:`Component <openmdao.core.component.py>` like this:

.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarDis1

----

The second discipline is given by another equation:

.. math::

  y_2(x, y_1, z_1, z_2) = \sqrt{y_1} + z_1 + z_2

Which is translated into a :ref:`Component <openmdao.core.component.py>` as seen here:

.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarDis2


----

Grouping components and connecting them together
**************************************************

The first discipline outputs :math:`y_1`, which is an input to the second discipline.
Similarly, the second discipline outputs :math:`y_2` which an input to the first discipline.
This interdependence causes a cycle that must be converged with a nonlinear solver in order to get a valid answer.

Models with cycles in them are often referred to as Multidisciplinary Analyses or **MDA** for short.
You can pick which kind of solver you would like to use to converge the MDA. The most common choices are:


    #. :ref:`NonlinearBlockGaussSeidel <nlbgs>`
    #. :ref:`NewtonSolver <nlnewton>`

The :code:`NonlinearBlockGaussSeidel` solver, also sometimes called a "fixed point iteration solver", is a gradient free method
that works well in many situations.
More tightly coupled problems, or problems with instances of :ref:`ImplicitComponent <comp-type-3-implicitcomp>` that don't implement their own `solve_nonlinear` method, will require the :code:`Newton` solver.

.. note::
    OpenMDAO comes with other nonlinear solvers you can use if they suit your problem.
    See the full list :ref:`here <feature_nonlinear_solvers>`


.. embed-code::
    openmdao.test_suite.components.sellar_feature.SellarMDA


There are a couple of important things to pay attention to here.
First, notice that we're working with a new type of class, called a :ref:`Group <feature_grouping_components>`.
:code:`Group` is the container that lets you build up complex model hierarchies.
Groups can contain other groups, components, or combinations of groups and components.

You can directly create instances of :code:`Group` to work with, or you can sub-class from it to define your own custom
groups. We're doing both things here. First, we define our own custom :code:`Group` sub-class called :code:`SellarMDA`.
In our run-script well create an instance of :code:`SellarMDA` to actually run it.
Then inside the :code:`setup` method of :code:`SellarMDA` we're also working directly with a group instance by doing this:

.. code::

    cycle = self.add_subsystem('cycle', Group(), promotes=['x', 'z', 'y1', 'y2'])
    d1 = cycle.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
    d2 = cycle.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

Our :code:`SellarMDA` group, when instantiated, will represent a three level hierarchy with the following structure

[INSERT TREE DIAGRAM HERE]

The sub-group, named :code:`cycle`, is useful here, because it contains the multi-disciplinary coupling of the Sellar problem.
This allows us to assign the non-linear solver to :code:`cycle` to just converge those two components, before moving on to the final
calculations for the :code:`obj_cmp`,:code:`con_cmp1`, and :code:`con_cmp2` to compute the actual outputs of the problem.



