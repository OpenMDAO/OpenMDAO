Setting up a simple analysis
=================================================

This tutorial illustrates how to build, run, and optimize a very simple model in
OpenMDAO. It will introduce the basic types of OpenMDAO classes, and the
sequence in which they must be created and used and show you how to set up
design variables, objective, and constraints for optimization.

Consider a paraboloid, defined by the explicit function

.. math::

  f(x,y) = (x-3.0)^2 + x \times y + (y+4.0)^2 - 3.0 ,

where :math:`x` and :math:`y` are the inputs to the function.
The minimum of this function is located at

.. math::

  x = \frac{20}{3} \quad , \quad y = -\frac{22}{3} .



.. embed-code::
    openmdao.test_suite.components.paraboloid


Lets break this script down and understand each section

Preamble
---------
::

    from __future__ import division, print_function
    from openmdao.api import ExplicitComponent

At the top of any script you'll see these lines (or lines very similar to these) which import needed classes and functions. On the first import line the `print_function` is used so the code in the script will work in Python 2 or 3. If you want to know whats going on with the division operator, check out this `detailed explanation <https://www.python.org/dev/peps/pep-0238/>`_. The second import line brings in OpenMDAO classes that are needed to build and run a model.
As you progress to more complex models you can expect to import more classes from `openmdao.api`, but for now we only need these 4.

Defining a component
---------------------
The component is the basic building block of a model. You will always define components as a sub-class of either `ExplicitComponent` or `ImplicitComponent`. Since our simple paraboloid function is explicit, we'll use the `ExplicitComponent`. You see three methods defined:

    - `setup`: define all your inputs and outputs here
    - `compute`: calculation of all output values for the given inputs
    - `compute_partials`: derivatives of all the outputs values with respect to all the inputs

.. note::

    What about implicit functions? Check out this tutorial [TODO: LINK ME!] about using an `ImplicitComponent`


.. embed-code::
    openmdao.test_suite.components.paraboloid.Paraboloid


The run-script
---------------------

In this example we've set up the run script at the bottom of the file.
The start of the run script is denoted by the following statement:

:code:`if __name__ == '__main__':`

All OpenMDAO models are built up from a hierarchy of `Group` instances that organize the components.
Here the hierarchy is very simple, consisting of a single root group that holds two components.
The first component is an `IndepVarComp` instance.
This is a special component that OpenMDAO provides for you to specify the independent variables in your problem.
The second component is an instance of the `Paraboloid` class that we just defined.

As part of the the model hierarchy, you will also define any connections to move data between components in the relevant group.
Here, we connect the design variables to the inputs on the paraboloid component.

Once the model hierarchy is defined,
we pass it to the constructor of the `Problem` class then call the `setup()` method on that problem which tells the framework to do some initial work to get the data structures in place for execution.
In this case, we call `run_model()` to actually perform the computation. Later, we'll see how to explicitly set drivers and will be calling `run_driver()` instead.

Here we called run_model twice.
The first times with the initial values of 3.0 and -4.0 for `x` and `y`.
The second time we changed those values and re-ran.
There are a few details to note here.
First, notice the way we printed the outputs via :code:`prob['parab_comp.f_xy']` and similarly how we set the new values for `x` and `y`.
You can both get and set values using the problem, which works with dimensional values in the units of the source variable.
In this case, there are no units on the source (i.e. `des_vars.x`).
You can read more about how OpenMDAO handles units and scaling here[LINK TO FEATURE DOC].

.. code::

    if __name__ == "__main__":
        from openmdao.api import Problem
        from openmdao.api import Group
        from openmdao.api import IndepVarComp

        model = Group()
        ivc = IndepVarComp()
        ivc.add_output('x', 3.0)
        ivc.add_output('y', -4.0)
        model.add_subsystem('des_vars', ivc)
        model.add_subsystem('parab_comp', Paraboloid())

        model.connect('des_vars.x', 'parab_comp.x')
        model.connect('des_vars.y', 'parab_comp.y')

        prob = Problem(model)
        prob.setup()
        prob.run_model()
        print(prob['parab_comp.f_xy'])

        prob['des_vars.x'] = 5.0
        prob['des_vars.y'] = -2.0
        prob.run_model()
        print(prob['parab_comp.f_xy'])
