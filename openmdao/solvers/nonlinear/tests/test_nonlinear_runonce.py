"""Test the NonlinearRunOnce linear solver class."""

import unittest

from openmdao.api import Problem, ScipyKrylov, IndepVarComp, Group
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.groups.parallel_groups import ConvergeDivergeGroups


class TestNonlinearRunOnceSolver(unittest.TestCase):

    def test_converge_diverge_groups(self):
        # Test derivatives for converge-diverge-groups topology.
        prob = Problem()
        prob.model = ConvergeDivergeGroups()
        prob.model.linear_solver = ScipyKrylov()
        prob.set_solver_print(level=0)

        prob.model.nonlinear_solver = NonlinearRunOnce()
        g1 = prob.model.g1
        g2 = g1.g2
        g3 = prob.model.g3
        g1.nonlinear_solver = NonlinearRunOnce()
        g2.nonlinear_solver = NonlinearRunOnce()
        g3.nonlinear_solver = NonlinearRunOnce()

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        # Make sure value is fine.
        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

    def test_feature_solver(self):
        from openmdao.api import Problem, Group, NonlinearRunOnce, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.nonlinear_solver = NonlinearRunOnce()

        prob.setup(check=False, mode='fwd')

        prob['x'] = 4.0
        prob['y'] = 6.0

        prob.run_model()

        assert_rel_error(self, prob['f_xy'], 122.0)

if __name__ == "__main__":
    unittest.main()
