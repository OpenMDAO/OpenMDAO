""" Definition of the Paraboloid component, which evaluates the equation
(x-3)^2 + xy + (y+4)^2 = 3
"""
from __future__ import division, print_function
from openmdao.core.explicitcomponent import ExplicitComponent


class Paraboloid(ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        """
        Jacobian for our paraboloid.
        """
        x = inputs['x']
        y = inputs['y']

        partials['f_xy', 'x'] = 2.0*x - 6.0 + y
        partials['f_xy', 'y'] = 2.0*y + 8.0 + x


if __name__ == "__main__":
    from openmdao.api import Problem, Group, ScipyOptimizeDriver, ExecComp, IndepVarComp

    model = Group()
    ivc = IndepVarComp()
    ivc.add_output('x', 3.0)
    ivc.add_output('y', -4.0)
    model.add_subsystem('des_vars', ivc)
    model.add_subsystem('parab_comp', Paraboloid())

    model.connect('des_vars.x', 'parab_comp.x')
    model.connect('des_vars.y', 'parab_comp.y')

    prob = Problem(model)
    prob.driver = ScipyOptimizeDriver()  # so 'openmdao cite' will report it for cite docs
    prob.setup()
    prob.run_model()
    print(prob['parab_comp.f_xy'])

    prob['des_vars.x'] = 5.0
    prob['des_vars.y'] = -2.0
    prob.run_model()
    print(prob['parab_comp.f_xy'])
