Setting up your first model
=================================================

This tutorial illustrates how to build, run, and optimize your first model in
OpenMDAO. It will introduce the basic types of OpenMDAO classes, and the
sequence in which they must be created and used and show you how to set up
design variables, objective, and constraints for optimziation.

Consider a paraboloid function, defined by the explicit function

.. math::

  f(x,y) = (x-3.0)^2 + x \times y + (y+4.0)^2 - 3.0 ,

where :math:`x` and :math`y` are the inputs to the function.
The minimum of this function is located at

.. math::

  x = \frac{20}{3} \quad , \quad y = -\frac{22}{3} .

Any calculations where the outputs are explicit functions of the inputs will
represented by defining a sub-class of `ExplicitComponent`[LINKED TO FEATURE
DOCS].

.. note:: 

    What about implicit functions? Check out this tutorial about using an `ImplicitComponent`

::

    from __future__ import division, print_function

    from openmdao.api import ExplicitComponent


    class Paraboloid(ExplicitComponent):

        def initialize_variables(self):
            self.add_input('x', val=0.0)
            self.add_input('y', val=0.0)
            self.add_output('f', val=0.0)

        def compute(self, inputs, outputs):
            x = inputs['x']
            y = inputs['y']
            outputs['f'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

        def compute_jacobian(self, inputs, outputs, jacobian):
            x = inputs['x']
            y = inputs['y']
            jacobian['f', 'x'] = 2.0*x - 6.0 + y
            jacobian['f', 'y'] = 2.0*y + 8.0 + x


    if __name__ == '__main__':
        from openmdao.api import Problem, Group, IndepVarComp

        model = Group()
        model.add_subsystem('inputs_comp', IndepVarComp((
            ('x', 3.0),
            ('y', -4.0),
        )))
        model.add_subsystem('parab_comp', Paraboloid())

        model.connect('inputs_comp.x', 'parab_comp.x')
        model.connect('inputs_comp.y', 'parab_comp.y')

        prob = Problem(model)
        prob.setup()
        prob.run()
        print(prob['parab_comp.f'])

Lets break this script down an understand each section

Preamble
---------
::

    from __future__ import division, print_function

    from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp

At the top of any script you'll see these lines (or lines very similar to these) which import basic functionality from python. the `print_function` is used so the code in the script will work in python 2.0 or 3.0. If you want to know whats going on with the division operator, check out this `detailed explanation <https://www.python.org/dev/peps/pep-0238/>`.

Defining a component
---------------------
The component is the basic building block of a model. There are two basic types of components: 

Here, our component represents a single explicit function 