.. _tutorial_paraboloid_analysis:

In the previous tutorial, we discussed the three basic kinds of Components in the OpenMDAO framework.
This tutorial focuses on using one of those, :ref:`ExplicitComponent <comp-type-2-explicitcomp>`, to build a simple analysis of a paraboloid function.
We'll explain the basic structure of a run file, show you how to set inputs, run the model, and check the output files.

**************************************
Paraboloid - A Single-Discipline Model
**************************************

Consider a paraboloid, defined by the explicit function

.. math::

  f(x,y) = (x-3.0)^2 + x \times y + (y+4.0)^2 - 3.0 ,

where :math:`x` and :math:`y` are the inputs to the function.
The minimum of this function is located at

.. math::

  x = \frac{20}{3} \quad , \quad y = -\frac{22}{3} .


Here is a complete script that defines this equation as a component and then executes it with different input values,
printing the results to the console when it's done.
Take a look at the full run script first, then we'll break it down part by part to explain what each one does.


.. embed-code::
    openmdao.test_suite.components.paraboloid_feature


Next, let's break this script down and understand each section:

Preamble
---------
::

    import openmdao.api as om

At the top of any script you'll see these lines (or lines very similar to these) which import needed classes and functions.
On the first import line, the `print_function` is used so that the code in the script will work in either Python 2 or 3.
If you want to know whats going on with the division operator, check out this `detailed explanation <https://www.python.org/dev/peps/pep-0238/>`_.

The second import line make availabe the most common OpenMDAO classes that are needed to build and run a model.
The `openmdao.api` module contains the class `ExplicitComponent` that we need to define the Paraboloid model. This
can be accessed via "om.ExplicitComponent" when we need it. As you progress to more complex models, you will learn about
more classes available in `openmdao.api`,but for now we only need this one to define our paraboloid component.

Defining a Component
---------------------
The component is the basic building block of a model.
You will always define components as a subclass of either :ref:`ExplicitComponent <openmdao.core.explicitcomponent.py>`
or :ref:`ImplicitComponent <openmdao.core.implicitcomponent.py>`.
Since our simple paraboloid function is explicit, we'll use the :ref:`ExplicitComponent <openmdao.core.explicitcomponent.py>`.
You see two methods defined:

    - `setup`: define all your inputs and outputs here, and declare derivatives
    - `compute`: calculation of all output values for the given inputs

In the `setup` method you define the inputs and outputs of the component,
and in this case you also ask OpenMDAO to approximate all the partial derivatives (derivatives of outputs with respect to inputs)
using the `finite difference method <https://en.wikipedia.org/wiki/Finite_difference_method>`_.

.. note::

    One of OpenMDAO's most unique features is its support for analytic derivatives.
    Providing analytic partial derivatives from your components can result in much more efficient optimizations.
    We'll get to using analytic derivatives in later tutorials.

.. embed-code::
    openmdao.test_suite.components.paraboloid_feature.Paraboloid


The Run Script
---------------------

In this example we've set up the run script at the bottom of the file.
The start of the run script is denoted by the following statement:

:code:`if __name__ == '__main__':`

All OpenMDAO models are built up from a hierarchy of :ref:`Group <openmdao.core.group.py>` instances that organize the components.
In this example, the hierarchy is very simple, consisting of a single root group that holds a single component that is an
instance of the `Paraboloid` class that we just defined.

Once the model hierarchy is defined,
we pass it to the constructor of the :ref:`Problem <openmdao.core.problem.py>` class.
Then we call the `setup()` method on that problem, which tells the framework to do some initial work to get the data structures in place for execution.
In this case, we call `run_model()` to actually perform the computation. Later, we'll see how to explicitly set drivers and
will be calling `run_driver()` instead.

Here we called run_model twice.
The first time `x` and `y` have the initial values of 3.0 and -4.0 respectively.
The second time we changed those values to 5.0 and -2.0 and then re-ran.
There are a few details to note here.
First, notice the way we printed the outputs via :code:`prob.get_val('parab_comp.f_xy')` and similarly how we set the new values for `x` and `y`.
You can get and set values using the "get_val" and "set_val" methods on problem. By default, these methods will set and get values defined in
the dimensional units of the specified input or output.
In this case, there are no units on the source (i.e. `des_vars.x`).

.. note::
    Detailed information on :ref:`units <units>` and :ref:`scaling <scale_outputs_and_resids>` can be found in the feature documentation.

.. code::

    if __name__ == "__main__":

        model = om.Group()
        model.add_subsystem('parab_comp', Paraboloid())

        prob = om.Problem(model)
        prob.setup()

        prob.set_val('parab_comp.x', 3.0)
        prob.set_val('parab_comp.y', -4.0)

        prob.run_model()
        print(prob['parab_comp.f_xy'])

        prob.set_val('parab_comp.x', 5.0)
        prob.set_val('parab_comp.y', -2.0)

        prob.run_model()
        print(prob['parab_comp.f_xy'])