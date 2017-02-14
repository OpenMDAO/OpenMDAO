""" Definition of the Paraboloid component, which evaluates the equation
(x-3)^2 + xy + (y+4)^2 = 3
"""

from openmdao.core.explicitcomponent import ExplicitComponent


class Paraboloid(ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def initialize_variables(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def compute_partials(self, inputs, outputs, partials):
        """
        Jacobian for our paraboloid.
        """
        x = inputs['x']
        y = inputs['y']

        partials['f_xy', 'x'] = 2.0*x - 6.0 + y
        partials['f_xy', 'y'] = 2.0*y + 8.0 + x
