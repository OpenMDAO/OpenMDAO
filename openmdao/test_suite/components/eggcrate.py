""" Definition of the Egg Crate component, which evaluates the equation
    x^2 + y^2 + 25 * (sin(x)^2 + sin(y)^2)
    http://benchmarkfcns.xyz/benchmarkfcns/eggcratefcn.html
"""
from math import sin,cos

import openmdao.api as om


class EggCrate(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = x^2 + y^2 + 25 * (sin(x)^2 + sin(y)^2).
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_output('f_xy', val=0.0)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        f(x,y) = x^2 + y^2 + 25 * (sin(x)^2 + sin(y)^2)

        Global optimal solution (minimum): x = 0.0; y = 0.0
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = x**2 + y**2 + 25.0 * (sin(x)**2 + sin(y)**2)

    def compute_partials(self, inputs, partials):
        """
        Jacobian for our paraboloid.
        """
        x = inputs['x']
        y = inputs['y']

        partials['f_xy', 'x'] = 2 * (x + 25 * cos(x) * sin(x))
        partials['f_xy', 'y'] = 2 * (y + 25 * cos(y) * sin(y))


if __name__ == "__main__":
    import numpy as np

    model = om.Group()
    ivc = om.IndepVarComp()
    ivc.add_output('x', 3.0)
    ivc.add_output('y', -4.0)
    model.add_subsystem('des_vars', ivc)
    model.add_subsystem('eggcrate_comp', EggCrate())

    model.connect('des_vars.x', 'eggcrate_comp.x')
    model.connect('des_vars.y', 'eggcrate_comp.y')

    prob = om.Problem(model)
    prob.driver = om.ScipyOptimizeDriver()  # so 'openmdao cite' will report it for cite docs
    prob.setup()
    prob.run_model()
    print(prob['eggcrate_comp.f_xy'])

    prob['des_vars.x'] = 0.1
    prob['des_vars.y'] = -0.1
    prob.run_model()
    np.testing.assert_almost_equal(prob['eggcrate_comp.f_xy'], [0.51833555] )