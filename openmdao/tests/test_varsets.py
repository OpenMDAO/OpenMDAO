
import unittest
import numpy as np
import scipy

from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from openmdao.api import ImplicitComponent, ScipyIterativeSolver
from openmdao.api import NonlinearBlockGS

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
        residuals['w'] = outputs['w'] + 2 * inputs['a']
        residuals['x'] = outputs['x'] + 3 * inputs['b']
        residuals['y'] = outputs['y'] + 4 * inputs['c']
        residuals['z'] = outputs['z'] + 5 * inputs['d']

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
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


class TestVarSets(unittest.TestCase):
    def assertEqualArrays(self, a, b):
        self.assertTrue(np.linalg.norm(a-b) < 1e-15)

    def setUp(self):
        p = Problem(Group())
        root = p.root
        root.add_subsystem("C1", Comp())
        root.add_subsystem("C2", Comp())
        root.connect("C1.w", "C2.a")
        root.connect("C1.x", "C2.b")
        root.connect("C1.y", "C2.c")
        root.connect("C1.z", "C2.d")

        root.connect("C2.w", "C1.a")
        root.connect("C2.x", "C1.b")
        root.connect("C2.y", "C1.c")
        root.connect("C2.z", "C1.d")

        gmres = scipy.sparse.linalg.gmres
        root.ln_solver = ScipyIterativeSolver()
        root.nl_solver = NonlinearBlockGS()
        p.setup()

        root.suppress_solver_output = True

        self.p = p

        # now create the same problem with no varsets
        p = Problem(Group())
        root = p.root
        root.add_subsystem("C1", Comp(use_varsets=False))
        root.add_subsystem("C2", Comp(use_varsets=False))
        root.connect("C1.w", "C2.a")
        root.connect("C1.x", "C2.b")
        root.connect("C1.y", "C2.c")
        root.connect("C1.z", "C2.d")

        root.connect("C2.w", "C1.a")
        root.connect("C2.x", "C1.b")
        root.connect("C2.y", "C1.c")
        root.connect("C2.z", "C1.d")

        gmres = scipy.sparse.linalg.gmres
        root.ln_solver = ScipyIterativeSolver()
        root.nl_solver = NonlinearBlockGS()
        p.setup()

        root.suppress_solver_output = True

        self.p_no_varsets = p

    def test_apply_linear(self):
        root = self.p.root
        root._vectors['output'][''].set_const(1.0)
        root._apply_linear([''], 'fwd')
        output = root._vectors['residual']['']._data
        self.assertEqualArrays(output[0], [4, 5, 4, 5])
        self.assertEqualArrays(output[1], [3, 6, 3, 6])

        # for no varsets, number should be the same, but reordered
        root = self.p_no_varsets.root

        root._vectors['output'][''].set_const(1.0)
        root._apply_linear([''], 'fwd')
        output_novs = root._vectors['residual']['']._data

        expected = np.array([output[i][j]
                                 for i,j in self.p._assembler._variable_set_indices['output']])
        self.assertEqualArrays(output_novs, expected)

        root = self.p.root
        root._vectors['residual'][''].set_const(1.0)
        root._apply_linear([''], 'rev')
        output = root._vectors['output']['']._data
        self.assertEqualArrays(output[0], [4, 5, 4, 5])
        self.assertEqualArrays(output[1], [3, 6, 3, 6])

        # for no varsets, number should be the same, but reordered
        root = self.p_no_varsets.root
        root._vectors['residual'][''].set_const(1.0)
        root._apply_linear([''], 'rev')
        output_novs = root._vectors['output']['']._data
        expected = np.array([output[i][j]
                                 for i,j in self.p._assembler._variable_set_indices['output']])
        self.assertEqualArrays(output_novs, expected)

    def test_solve_linear(self):
        root = self.p.root
        root._vectors['residual'][''].set_const(1.0)
        root._vectors['output'][''].set_const(0.0)
        root._solve_linear([''], 'fwd')
        output = root._vectors['output']['']._data
        self.assertEqualArrays(output[0], [ 1./4.,  1./5. ,  1./4.,  1./5. ])
        self.assertEqualArrays(output[1], [ 1./3.,  1./6. ,  1./3.,  1./6. ])

        # now try without varsets for comparison
        root = self.p_no_varsets.root
        root._vectors['residual'][''].set_const(1.0)
        root._vectors['output'][''].set_const(0.0)
        root._solve_linear([''], 'fwd')
        output_novs = root._vectors['output']['']._data
        expected = np.array([output[i][j]
                                 for i,j in self.p._assembler._variable_set_indices['output']])
        self.assertEqualArrays(output_novs, expected)

        root = self.p.root
        root._vectors['output'][''].set_const(1.0)
        root._vectors['residual'][''].set_const(0.0)
        root._solve_linear([''], 'rev')
        output = root._vectors['residual']['']._data
        self.assertEqualArrays(output[0], [ 1./4.,  1./5. , 1./4.,  1./5. ])
        self.assertEqualArrays(output[1], [ 1./3.,  1./6. ,  1./3.,  1./6. ])

        # now try without varsets for comparison
        root = self.p_no_varsets.root
        root._vectors['output'][''].set_const(1.0)
        root._vectors['residual'][''].set_const(0.0)
        root._solve_linear([''], 'rev')
        output_novs = root._vectors['residual']['']._data
        expected = np.array([output[i][j]
                                 for i,j in self.p._assembler._variable_set_indices['output']])
        self.assertEqualArrays(output_novs, expected)
