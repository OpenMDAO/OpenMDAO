"""test the KSP iterative solver class."""

import unittest

import numpy as np

from openmdao.solvers.ln_petsc_ksp import PetscKSP

from openmdao.api import Problem, Group
from openmdao.api import ImplicitComponent, ScipyIterativeSolver
from openmdao.api import NonlinearBlockGS

from openmdao.vectors.petsc_vector import PETScVector


class Comp(ImplicitComponent):

    def __init__(self, use_varsets=True):
        super(Comp, self).__init__()
        self._use_var_sets = use_varsets

    def initialize_variables(self):
        if self._use_var_sets:
            self.add_input('a', var_set=1)
            self.add_input('b', var_set=0)
            self.add_input('c', var_set=1)
            self.add_input('d', var_set=2)

            self.add_output('w', var_set=5)
            self.add_output('x', var_set=1)
            self.add_output('y', var_set=1)
            self.add_output('z', var_set=5)
        else:
            self.add_input('a')
            self.add_input('b')
            self.add_input('c')
            self.add_input('d')

            self.add_output('w')
            self.add_output('x')
            self.add_output('y')
            self.add_output('z')

    def apply_nonlinear(self, inputs, outputs, residuals):
        print(self.path_name, 'apply_nonlinear')
        residuals['w'] = outputs['w'] + 2 * inputs['a']
        residuals['x'] = outputs['x'] + 3 * inputs['b']
        residuals['y'] = outputs['y'] + 4 * inputs['c']
        residuals['z'] = outputs['z'] + 5 * inputs['d']

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        print(' -- ',self.path_name, 'apply_linear')
        if mode == 'fwd':
            if 'w' in d_outputs:
                d_residuals['w'] += d_outputs['w']
                if 'a' in d_inputs:
                    d_residuals['w'] += 2 * d_inputs['a']
            if 'x' in d_outputs:
                d_residuals['x'] += d_outputs['x']
                if 'b' in d_inputs:
                    d_residuals['x'] += 3 * d_inputs['b']
            if 'y' in d_outputs:
                d_residuals['y'] += d_outputs['y']
                if 'c' in d_inputs:
                    d_residuals['y'] += 4 * d_inputs['c']
            if 'z' in d_outputs:
                d_residuals['z'] += d_outputs['z']
                if 'd' in d_inputs:
                    d_residuals['z'] += 5 * d_inputs['d']
        else:
            if 'w' in d_outputs:
                d_outputs['w'] += d_residuals['w']
                if 'a' in d_inputs:
                    d_inputs['a'] += 2 * d_residuals['w']
            if 'x' in d_outputs:
                d_outputs['x'] += d_residuals['x']
                if 'b' in d_inputs:
                    d_inputs['b'] += 3 * d_residuals['x']
            if 'y' in d_outputs:
                d_outputs['y'] += d_residuals['y']
                if 'c' in d_inputs:
                    d_inputs['c'] += 4 * d_residuals['y']
            if 'z' in d_outputs:
                d_outputs['z'] += d_residuals['z']
                if 'd' in d_inputs:
                    d_inputs['d'] += 5 * d_residuals['z']


class Prob(Problem):

    def __init__(self, lnSolverClass=ScipyIterativeSolver,
                       nlSolverClass=NonlinearBlockGS, use_varsets=True):
        root = Group()

        root.add_subsystem("C1", Comp(use_varsets))
        root.add_subsystem("C2", Comp(use_varsets))
        root.connect("C1.w", "C2.a")
        root.connect("C1.x", "C2.b")
        root.connect("C1.y", "C2.c")
        root.connect("C1.z", "C2.d")

        root.connect("C2.w", "C1.a")
        root.connect("C2.x", "C1.b")
        root.connect("C2.y", "C1.c")
        root.connect("C2.z", "C1.d")

        root.ln_solver = lnSolverClass()
        root.nl_solver = nlSolverClass()

        self.expected_output = [
            [ 1./4.,  1./5.,  1./4.,  1./5. ],
            [ 1./3.,  1./6.,  1./3.,  1./6. ]
        ]

        super(Prob, self).__init__(root)


class TestPetscKSP(unittest.TestCase):

    def assertEqualArrays(self, a, b):
        self.assertTrue(np.linalg.norm(a-b) < 1e-15)

    def test_solve_linear_scipy(self):
        p = Prob(lnSolverClass=ScipyIterativeSolver)
        p.setup()

        # forward
        root = p.root
        root._vectors['residual'][''].set_const(1.0)
        root._vectors['output'][''].set_const(0.0)
        root._solve_linear([''], 'fwd')
        output = root._vectors['output']['']._data
        self.assertEqualArrays(output[0], p.expected_output[0])
        self.assertEqualArrays(output[1], p.expected_output[1])

        # reverse
        root = p.root
        root._vectors['output'][''].set_const(1.0)
        root._vectors['residual'][''].set_const(0.0)
        root._solve_linear([''], 'rev')
        output = root._vectors['residual']['']._data
        self.assertEqualArrays(output[0], p.expected_output[0])
        self.assertEqualArrays(output[1], p.expected_output[1])

    def test_solve_linear_ksp(self):
        p = Prob(lnSolverClass=PetscKSP)
        p.setup(VectorClass=PETScVector)

        # forward
        root = p.root
        root._vectors['residual'][''].set_const(1.0)
        root._vectors['output'][''].set_const(0.0)
        root._solve_linear([''], 'fwd')
        output = root._vectors['output']['']._data
        self.assertEqualArrays(output[0], p.expected_output[0])
        self.assertEqualArrays(output[1], p.expected_output[1])

        # reverse
        root = p.root
        root._vectors['output'][''].set_const(1.0)
        root._vectors['residual'][''].set_const(0.0)
        root._solve_linear([''], 'rev')
        output = root._vectors['residual']['']._data
        self.assertEqualArrays(output[0], p.expected_output[0])
        self.assertEqualArrays(output[1], p.expected_output[1])


if __name__ == "__main__":
    unittest.main()
