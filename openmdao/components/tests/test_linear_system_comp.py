"""Test the LinearSystemComp."""

import unittest

import numpy as np

from openmdao.api import Group, Problem, IndepVarComp
from openmdao.api import LinearSystemComp, ScipyKrylov, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error


class TestLinearSystemComp(unittest.TestCase):
    """Test the LinearSystemComp class with a 3x3 linear system."""

    def test_linear_system_comp(self):
        """Check against the scipy solver."""

        model = Group()

        x = np.array([1, 2, -3])
        A = np.array([[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]])
        b = A.dot(x)

        model.add_subsystem('p1', IndepVarComp('A', A))
        model.add_subsystem('p2', IndepVarComp('b', b))

        lingrp = model.add_subsystem('lingrp', Group(), promotes=['*'])
        lingrp.add_subsystem('lin', LinearSystemComp(size=3))

        model.connect('p1.A', 'lin.A')
        model.connect('p2.b', 'lin.b')

        prob = Problem(model)
        prob.setup()

        lingrp.linear_solver = ScipyKrylov()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['lin.x'], x, .0001)
        assert_rel_error(self, prob.model._residuals.get_norm(), 0.0, 1e-10)

    def test_linear_system_comp_solve_linear(self):
        """Check against solve_linear."""

        x = np.array([1, 2, -3])
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = A.dot(x)
        b_T = A.T.dot(x)

        def check_derivs(lin_sys_comp):

            prob = Problem()

            prob.model.add_subsystem('p1', IndepVarComp('A', A))
            prob.model.add_subsystem('p2', IndepVarComp('b', b))

            lingrp = prob.model.add_subsystem('lingrp', Group(), promotes=['*'])
            lingrp.add_subsystem('lin', lin_sys_comp)

            prob.model.connect('p1.A', 'lin.A')
            prob.model.connect('p2.b', 'lin.b')

            prob.setup(check=False)
            prob.set_solver_print(level=0)

            prob.run_model()
            prob.model.run_linearize()

            # prob.check_partials()

            # Compare against calculated derivs
            # Ainv = np.linalg.inv(A) # Don't use linalg.inv or a mathematician will die
            Ainv = np.array([[3., -2., 1.],
                             [-3., 3., -2.],
                             [1., -1., 1.]])

            dx_dA = np.outer(Ainv, -x).reshape(3, 9)
            dx_db = Ainv

            d_inputs, d_outputs, d_residuals = lingrp.get_linear_vectors()

            # Forward mode with RHS of self.b
            d_residuals['lin.x'] = b
            lingrp.run_solve_linear(['linear'], 'fwd')
            sol = d_outputs['lin.x']
            assert_rel_error(self, sol, x, .0001)

            # Reverse mode with RHS of self.b_T
            d_outputs['lin.x'] = b_T
            lingrp.run_solve_linear(['linear'], 'rev')
            sol = d_residuals['lin.x']
            assert_rel_error(self, sol, x, .0001)

            J = prob.compute_totals(['lin.x'], ['p1.A', 'p2.b'], return_format='flat_dict')
            assert_rel_error(self, J['lin.x', 'p1.A'], dx_dA, .0001)
            assert_rel_error(self, J['lin.x', 'p2.b'], dx_db, .0001)

        lin_sys_comp = LinearSystemComp(size=3)
        check_derivs(lin_sys_comp)

    def test_feature_basic(self):
        import numpy as np

        from openmdao.api import Group, Problem, IndepVarComp
        from openmdao.api import LinearSystemComp, ScipyKrylov


        model = Group()

        x = np.array([1, 2, -3])
        A = np.array([[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]])
        b = A.dot(x)

        model.add_subsystem('p1', IndepVarComp('A', A))
        model.add_subsystem('p2', IndepVarComp('b', b))

        lingrp = model.add_subsystem('lingrp', Group(), promotes=['*'])
        lingrp.add_subsystem('lin', LinearSystemComp(size=3))

        model.connect('p1.A', 'lin.A')
        model.connect('p2.b', 'lin.b')

        prob = Problem(model)
        prob.setup()

        lingrp.linear_solver = ScipyKrylov()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['lin.x'], x, .0001)
        assert_rel_error(self, prob.model._residuals.get_norm(), 0.0, 1e-10)

if __name__ == "__main__":
    unittest.main()
