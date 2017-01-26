"""Test the LinearSystemComp."""
import unittest
import numpy as np

from openmdao.api import Group, Problem, IndepVarComp
from openmdao.api import LinearSystemComp, ScipyIterativeSolver
from openmdao.devtools.testutil import assert_rel_error


class TestLinearSystem(unittest.TestCase):
    """Test the LinearSystemComp class with a 3x3 linear system."""

    def setUp(self):
        """Set up a problem with a 3x3 linear system."""
        model = Group()

        x = np.array([1, 2, -3])
        A = np.array([[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]])
        b = A.dot(x)
        b_T = A.T.dot(x)

        model.add_subsystem('p1', IndepVarComp('A', A))
        model.add_subsystem('p2', IndepVarComp('b', b))

        lingrp = model.add_subsystem('lingrp', Group(), promotes=['*'])
        lingrp.add_subsystem('lin', LinearSystemComp(size=3))

        model.connect('p1.A', 'lin.A')
        model.connect('p2.b', 'lin.b')

        prob = Problem(model)
        prob.setup()

        self.prob = prob
        self.x = x
        self.b = b
        self.b_T = b_T

    def test_linear_system(self):
        """Check against the scipy solver."""
        prob = self.prob

        lingrp = prob.model.get_subsystem('lingrp')
        lingrp.ln_solver = ScipyIterativeSolver()

        prob.run_model()

        assert_rel_error(self, prob['lin.x'], self.x, .0001)
        assert_rel_error(self, prob.model._residuals.get_norm(), 0.0, 1e-10)

    def test_linear_system_solve_linear(self):
        """Check against solve_linear."""
        prob = self.prob

        lingrp = prob.model.get_subsystem('lingrp')
        lingrp.ln_solver = ScipyIterativeSolver()

        prob.run_model()

        # Forward mode with RHS of self.b
        lingrp._vectors['residual']['linear']['lin.x'] = self.b
        lingrp._solve_linear(['linear'], 'fwd')
        sol = lingrp._vectors['output']['linear']['lin.x']
        assert_rel_error(self, sol, self.x, .0001)

        # Reverse mode with RHS of self.b_T
        lingrp._vectors['output']['linear']['lin.x'] = self.b_T
        lingrp._solve_linear(['linear'], 'rev')
        sol = lingrp._vectors['residual']['linear']['lin.x']
        assert_rel_error(self, sol, self.x, .0001)

        # Compare against calculated derivs
        # Ainv = np.linalg.inv(A)
        # dx_dA = np.outer(Ainv, -x).reshape(3, 9)
        # dx_db = Ainv

        # J = prob.calc_gradient(['p1.A', 'p2.b'], ['lin.x'], mode='fwd', return_format='dict')
        # assert_rel_error(self, J['lin.x']['p1.A'], dx_dA, .0001)
        # assert_rel_error(self, J['lin.x']['p2.b'], dx_db, .0001)

        # J = prob.calc_gradient(['p1.A', 'p2.b'], ['lin.x'], mode='rev', return_format='dict')
        # assert_rel_error(self, J['lin.x']['p1.A'], dx_dA, .0001)
        # assert_rel_error(self, J['lin.x']['p2.b'], dx_db, .0001)

        # J = prob.calc_gradient(['p1.A', 'p2.b'], ['lin.x'], mode='fd', return_format='dict')
        # assert_rel_error(self, J['lin.x']['p1.A'], dx_dA, .0001)
        # assert_rel_error(self, J['lin.x']['p2.b'], dx_db, .0001)


if __name__ == "__main__":
    unittest.main()
