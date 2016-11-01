
import unittest
import numpy as np
import scipy

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ImplicitComponent
from openmdao.solvers.solver import ScipyIterativeSolver

class Comp(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('a', var_set=1)
        self.add_input('b', var_set=0)
        self.add_input('c', var_set=1)
        self.add_input('d', var_set=2)
        
        self.add_output('w', var_set=5)
        self.add_output('x', var_set=1)
        self.add_output('y', var_set=1)
        self.add_output('z', var_set=5)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['w'] = outputs['w'] + 2 * inputs['a']
        residuals['x'] = outputs['x'] + 3 * inputs['b']
        residuals['y'] = outputs['y'] + 4 * inputs['c']
        residuals['z'] = outputs['z'] + 5 * inputs['d']

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_residuals['w'] = 0
            d_residuals['x'] = 0
            d_residuals['y'] = 0
            d_residuals['z'] = 0
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
            d_outputs['w'] = 0
            d_outputs['x'] = 0
            d_outputs['y'] = 0
            d_outputs['z'] = 0
            d_inputs['a'] = 0
            d_inputs['b'] = 0
            d_inputs['c'] = 0
            d_inputs['d'] = 0
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
    def test_solve_linear(self):
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
        root._solvers_linear = ScipyIterativeSolver(options={'solver':gmres})
        p.setup()
        
        root.set_solver_print(False)
        root._solve_nonlinear()
    
        root._vectors['residual'][''].set_const(1.0)
        root._solve_linear([''], 'fwd')
        output = root._vectors['output']['']._data[0]
        self.assertEqualArrays(output, [0.4, 0.2])
    
        root._vectors['output'][''].set_const(1.0)
        root._solve_linear([''], 'rev')
        output = root._vectors['residual']['']._data[0]
        self.assertEqualArrays(output, [0.2, 0.4])
        