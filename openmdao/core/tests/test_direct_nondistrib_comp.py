import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class QuadraticComp(om.ImplicitComponent):
    """
    A Simple Implicit Component representing a Quadratic Equation.

    R(a, b, c, x) = ax^2 + bx + c

    Solution via Quadratic Formula:
    x = (-b + sqrt(b^2 - 4ac)) / 2a
    """

    def setup(self):
        self.add_input('a', val=1.0)
        self.add_input('b', val=1.0)
        self.add_input('c', val=1.0)
        self.add_output('x', val=1.0)

        self.declare_partials(of='x', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        residuals['x'] = a * x ** 2 + b * x + c

    def linearize(self, inputs, outputs, partials):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']

        partials['x', 'a'] = x ** 2
        partials['x', 'b'] = x
        partials['x', 'c'] = 1.0
        partials['x', 'x'] = 2 * a * x + b

        self.inv_jac = 1.0 / (2 * a * x + b)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class NondistribDirectCompTestCase(unittest.TestCase):
    N_PROCS = 2

    def test_direct(self):
        p = om.Problem()

        comp = p.model.add_subsystem('comp', QuadraticComp())
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        comp.linear_solver = om.DirectSolver()

        p.setup(force_alloc_complex=True)
        p.run_model()

        partials = p.check_partials(includes=['comp'], method='cs', out_stream=None)
        assert_check_partials(partials)

    def test_broyden(self):
        p = om.Problem()

        comp = p.model.add_subsystem('comp', QuadraticComp())
        comp.nonlinear_solver = om.BroydenSolver()
        comp.linear_solver = om.DirectSolver()

        p.setup(force_alloc_complex=True)
        p.run_model()

        partials = p.check_partials(includes=['comp'], method='cs', out_stream=None)
        assert_check_partials(partials)




if __name__ == '__main__':
    unittest.main()
