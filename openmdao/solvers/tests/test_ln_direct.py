"""Test the DirectSolver linear solver class."""

from __future__ import division, print_function

import unittest


from openmdao.api import Problem, Group, IndepVarComp, DirectSolver
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.solvers.tests.linear_test_base import LinearSolverTests

class TestDirectSolver(LinearSolverTests.LinearSolverTestCase):

    ln_solver_class = DirectSolver

    # DirectSolver doesn't iterate.
    def test_solve_linear_maxiter(self):
        pass

    def test_solve_on_subsystem(self):
        """solve an implicit system with DirectSolver attached to a subsystem"""

        p = Problem()
        model = p.model = Group()
        dv = model.add_subsystem('des_vars', IndepVarComp())
        # just need a dummy variable so the sizes don't match between root and g1
        dv.add_output('dummy', val=1.0, shape=10)

        g1 = model.add_subsystem('g1', TestImplicitGroup(lnSolverClass=DirectSolver))

        p.model.ln_solver.options['maxiter'] = 1
        p.setup(check=False)

        p.set_solver_print(level=0)

        # forward
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        g1._linearize()
        g1._solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        # The empty first entry in _data is due to the dummy
        #     variable being in a different variable set not owned by g1
        assert_rel_error(self, output[1], g1.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], g1.expected_solution[1], 1e-15)

        # reverse
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        g1.ln_solver._linearize()
        g1._solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output[1], g1.expected_solution[0], 3e-15)
        assert_rel_error(self, output[5], g1.expected_solution[1], 3e-15)


class TestDirectSolverFeature(unittest.TestCase):

    def test_specify_solver(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = DirectSolver()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

if __name__ == "__main__":
    unittest.main()
