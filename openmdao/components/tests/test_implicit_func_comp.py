import unittest
import math

import numpy as np
from scipy import linalg

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import openmdao.func_api as omf


class TestImplicitFuncComp(unittest.TestCase):
    def test_apply_linear(self):

        x_=np.array([1, 2, -3])
        A = np.array([[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]])
        b = A.dot(x_)

        def resid_func(A, b, x):
            rx = A.dot(x) - b
            return rx

        f = (omf.wrap(resid_func)
                .add_input('A', shape=A.shape)
                .add_input('b', shape=b.shape)
                .add_output('x', resid='rx', val=x_))

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('A', A))
        model.add_subsystem('p2', om.IndepVarComp('b', b))

        lingrp = model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', om.ImplicitFuncComp(f))

        model.connect('p1.A', 'lin.A')
        model.connect('p2.b', 'lin.b')

        prob.setup()

        lingrp.linear_solver = om.ScipyKrylov()

        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['lin.x'], x_, .0001)
        assert_near_equal(prob.model._residuals.get_norm(), 0.0, 1e-10)

        model.run_apply_nonlinear()

        with model._scaled_context_all():
            val = model.lingrp.lin._residuals['x']
            assert_near_equal(val, np.zeros((3, )), tolerance=1e-8)

    def test_solve_linear(self):
        """Check against solve_linear."""

        x = np.array([1, 2, -3])
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = A.dot(x)
        b_T = A.T.dot(x)

        def solve_linear_func(d_x, mode, lup):
            if mode == 'fwd':
                return linalg.lu_solve(lup, d_x, trans=0)
            else:  # rev
                return linalg.lu_solve(lup, d_x, trans=1)

        def linearize_func(A, b, x, J):
            J['x', 'A'] = np.tile(x, 3).flat
            J['x', 'x'] = A.flat
            J['x', 'b'] = np.full(3, -1.0)

            # return LU decomp for use later in solve_linear
            return linalg.lu_factor(A)

        def resid_func(A, b, x):
            rx = A.dot(x) - b
            return rx

        f = (omf.wrap(resid_func)
                .add_input('A', val=A)
                .add_input('b', val=b)
                .add_output('x', resid='rx', val=x)
                .declare_partials(of='x', wrt='b', rows=np.arange(3), cols=np.arange(3))
                .declare_partials(of='x', wrt='A', rows=np.repeat(np.arange(3), 3), cols=np.arange(9))
                .declare_partials(of='x', wrt='x', rows=np.repeat(np.arange(3), 3), cols=np.tile(np.arange(3), 3)))

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('A', A))
        model.add_subsystem('p2', om.IndepVarComp('b', b))

        lingrp = model.add_subsystem('lingrp', om.Group(), promotes=['*'])
        lingrp.add_subsystem('lin', om.ImplicitFuncComp(f, linearize=linearize_func, solve_linear=solve_linear_func))

        prob.model.connect('p1.A', 'lin.A')
        prob.model.connect('p2.b', 'lin.b')

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()
        prob.model.run_linearize()

        # Compare against calculated derivs
        Ainv = np.array([[3., -2., 1.],
                         [-3., 3., -2.],
                         [1., -1., 1.]])

        dx_dA = np.outer(Ainv, -x).reshape(3, 9)
        dx_db = Ainv

        d_inputs, d_outputs, d_residuals = lingrp.get_linear_vectors()

        # Forward mode with RHS of self.b
        d_residuals['lin.x'] = b
        lingrp.run_solve_linear('fwd')
        sol = d_outputs['lin.x']
        assert_near_equal(sol, x, .0001)

        # Reverse mode with RHS of self.b_T
        d_outputs['lin.x'] = b_T
        lingrp.run_solve_linear('rev')
        sol = d_residuals['lin.x']
        assert_near_equal(sol, x, .0001)

        prob.model.lingrp.lin._no_check_partials = False  # override skipping of check_partials

        J = prob.compute_totals(['lin.x'], ['p1.A', 'p2.b', 'lin.x'], return_format='flat_dict')
        assert_near_equal(J['lin.x', 'p1.A'], dx_dA, .0001)
        assert_near_equal(J['lin.x', 'p2.b'], dx_db, .0001)

        data = prob.check_partials(out_stream=None)

        abs_errors = data['lingrp.lin'][('x', 'x')]['abs error']
        self.assertTrue(len(abs_errors) > 0)
        self.assertTrue(abs_errors[0] < 1.e-6)
