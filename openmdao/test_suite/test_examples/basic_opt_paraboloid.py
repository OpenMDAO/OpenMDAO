from __future__ import print_function, division, absolute_import

import unittest

from openmdao.utils.assert_utils import assert_rel_error

from openmdao.api import Problem, ScipyOptimizer, ExecComp, IndepVarComp

from openmdao.test_suite.components.paraboloid import Paraboloid

class BasicOptParaboloid(unittest.TestCase):

    def test_unconstrainted(self):
        from openmdao.api import Problem, ScipyOptimizer, IndepVarComp

        # We'll use the component that was defined in the last tutorial
        from openmdao.test_suite.components.paraboloid import Paraboloid

        # build the model
        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp())
        indeps.add_output('x', 3.0)
        indeps.add_output('y', -4.0)

        prob.model.add_subsystem('paraboloid', Paraboloid())

        prob.model.connect('indeps.x', 'paraboloid.x')
        prob.model.connect('indeps.y', 'paraboloid.y')

        # setup the optimization
        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'COBYLA'

        prob.model.add_design_var('indeps.x', lower=-50, upper=50)
        prob.model.add_design_var('indeps.y', lower=-50, upper=50)
        prob.model.add_objective('paraboloid.f_xy')

        prob.setup()
        prob.run_driver()

        # minimum value
        assert_rel_error(self, prob['paraboloid.f_xy'], -27.33333, 1e-6)

        # location of the minimum
        assert_rel_error(self, prob['indeps.x'], 6.6667, 1e-4)
        assert_rel_error(self, prob['indeps.y'], -7.33333, 1e-4)


    def test_constrained(self):
        from openmdao.api import Problem, ScipyOptimizer, ExecComp, IndepVarComp

        # We'll use the component that was defined in the last tutorial
        from openmdao.test_suite.components.paraboloid import Paraboloid

        # build the model
        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp())
        indeps.add_output('x', 3.0)
        indeps.add_output('y', -4.0)

        prob.model.add_subsystem('parab', Paraboloid())

        # define the component whos output will be constrained
        prob.model.add_subsystem('const', ExecComp('g = x + y'))

        prob.model.connect('indeps.x', ['parab.x', 'const.x'])
        prob.model.connect('indeps.y', ['parab.y', 'const.y'])

        # setup the optimization
        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'COBYLA'

        prob.model.add_design_var('indeps.x', lower=-50, upper=50)
        prob.model.add_design_var('indeps.y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')

        # to add the constraint to the model
        prob.model.add_constraint('const.g', lower=0, upper=10.)
        # prob.model.add_constraint('const.g', equals=0.)

        prob.setup()
        prob.run_driver()

        # minimum value
        assert_rel_error(self, prob['parab.f_xy'], -27., 1e-6)

        # location of the minimum
        assert_rel_error(self, prob['indeps.x'], 7, 1e-4)
        assert_rel_error(self, prob['indeps.y'], -7, 1e-4)


if __name__ == "__main__":

    unittest.main()