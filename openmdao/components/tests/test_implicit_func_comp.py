import unittest
import math

import numpy as np
from scipy import linalg

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import openmdao.func_api as omf


class TestImplicitFuncComp(unittest.TestCase):
    def test_apply_nonlinear(self):

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

    def test_component_model_w_linearize(self):

        def apply_nonlinear(z, x, y1, y2, R_y1, R_y2):

            z0 = z[0]
            z1 = z[1]

            return 0, (y1**.5 + z0 + z1) - y2, (z0**2 + z1 + x - 0.2*y2) - y1 - R_y1, (y1**.5 + z0 + z1) - y2 - R_y2

        f = (omf.wrap(apply_nonlinear)
                .add_input('z', val=np.array([-1., -1.]))
                .add_input('x', val=2.)
                .add_output('y1', resid='r_y1')
                .add_output('y2', resid='r_y2')
                .add_output('R_y1', resid='r_R_y1')
                .add_output('R_y2', resid='r_R_y1')
                .declare_partials('y1', 'y1')
                .declare_partials('y2', ['z', 'y1', 'y2'])
                .declare_partials('R_y1', ['R_y1', 'x', 'z', 'y1', 'y2'])
                .declare_partials('R_y2', ['R_y2','z', 'y1', 'y2']))

        def linearize(z, x, y1, y2, R_y1, R_y2, J):

            J['y1', 'y1'] = -1.
            J['R_y1','R_y1'] = -1

            J['R_y1', 'x'] = [1]
            J['R_y1', 'z'] = [2*z[0], 1]
            J['R_y1', 'y1'] = -1.
            J['R_y1', 'y2'] = -0.2

            J['y2','y2'] = -1

            J['R_y2','R_y2'] = -1
            J['y2','z'] = [1, 1]
            J['y2','y1'] = 0.5*y1**-0.5

            J['R_y2','y2'] = -1
            J['R_y2','z'] = [1, 1]
            J['R_y2','y1'] = 0.5*y1**-0.5

        p_opt = om.Problem()

        p_opt.model = om.ImplicitFuncComp(f, linearize=linearize) #, solve_nonlinear=solve_nonlinear)

        newton = p_opt.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        newton.options['iprint'] = 0
        p_opt.model.linear_solver = om.DirectSolver(assemble_jac=True)

        p_opt.driver = om.ScipyOptimizeDriver()
        p_opt.driver.options['disp'] = False

        p_opt.model.add_design_var('y1', lower=-10, upper=10)
        p_opt.model.add_constraint('R_y1', equals=0)

        # this objective doesn't really matter... just need something there
        p_opt.model.add_objective('y2')

        p_opt.setup()

        # set
        p_opt['y2'] = 5
        p_opt['y1'] = 5

        p_opt.run_driver()

        np.testing.assert_almost_equal(p_opt['y1'], 2.109516506074582, decimal=5)
        np.testing.assert_almost_equal(p_opt['y2'], -0.5475825303740725, decimal=5)
        np.testing.assert_almost_equal(p_opt['x'], 2.0, decimal=5)
        np.testing.assert_almost_equal(p_opt['z'], np.array([-1., -1.]), decimal=5)

    def test_solve_nonlinear(self):

        size = 3

        def apply_nl(y, x):
            A = np.array([[1.0, 8.0, 0.0], [-1.0, 10.0, 2.0], [3.0, 100.5, 1.0]])
            R_x = y - A.dot(x)
            return R_x

        def solve_nl(y, x):
            A = np.array([[1.0, 8.0, 0.0], [-1.0, 10.0, 2.0], [3.0, 100.5, 1.0]])
            x = np.linalg.inv(A).dot(y)
            return x

        f = (omf.wrap(apply_nl)
                .defaults(shape=size)
                .add_output('x', resid='R_x')
                .declare_partials(of='*', wrt='*', method='cs'))

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p',  om.IndepVarComp('x', val=np.ones(size)))
        model.add_subsystem('comp', om.ImplicitFuncComp(f, solve_nonlinear=solve_nl))

        model.connect('p.x', 'comp.y')

        model.linear_solver = om.DirectSolver() # needed since comp has no solve_linear

        model.add_design_var('p.x', lower=-11, upper=11)
        model.add_constraint('p.x', upper=3.3)
        model.add_objective('comp.x')

        prob.setup()
        prob.run_model()

        assert_check_totals(prob.check_totals(out_stream=None))

    def test_solve_nonlinear2(self):

        def apply_nl(a, b, c, x):
            R_x = a * x ** 2 + b * x + c
            return R_x

        def solve_nl(a, b, c, x):
            x = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
            return x

        f = (omf.wrap(apply_nl)
                .add_output('x', resid='R_x', val=0.0)
                .declare_partials(of='*', wrt='*', method='cs')
                )

        p = om.Problem()
        p.model.add_subsystem('comp', om.ImplicitFuncComp(f, solve_nonlinear=solve_nl))

        # need this since comp is implicit and doesn't have a solve_linear
        p.model.linear_solver = om.DirectSolver()

        p.setup()

        p.set_val('comp.a', 2.)
        p.set_val('comp.b', -8.)
        p.set_val('comp.c', 6.)
        p.run_model()

        assert_check_partials(p.check_partials(includes=['comp'], out_stream=None), atol=1e-5)
        assert_check_totals(p.check_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'], out_stream=None))

    def test_solve_linear_linearize(self):

        def apply_nl(a, b, c, x):
            R_x = a * x ** 2 + b * x + c
            return R_x

        def linearize(a, b, c, x, partials):
            partials['x', 'a'] = x ** 2
            partials['x', 'b'] = x
            partials['x', 'c'] = 1.0
            partials['x', 'x'] = 2 * a * x + b

            inv_jac = 1.0 / (2 * a * x + b)
            return inv_jac

        def solve_linear(d_x, mode, inv_jac):
            if mode == 'fwd':
                d_x = inv_jac * d_x
                return d_x
            elif mode == 'rev':
                dR_x = inv_jac * d_x
                return dR_x

        f = (omf.wrap(apply_nl)
                .add_output('x', resid='R_x', val=0.0)
                .declare_partials(of='*', wrt='*')
                )

        p = om.Problem()
        p.model.add_subsystem('comp', om.ImplicitFuncComp(f, solve_linear=solve_linear, linearize=linearize))

        # need iterative nonlinear solver since implicit comp doesn't have one
        newton = p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        newton.options['iprint'] = 0

        p.setup()

        p.set_val('comp.a', 2.)
        p.set_val('comp.b', -8.)
        p.set_val('comp.c', 6.)
        p.run_model()

        assert_check_partials(p.check_partials(includes=['comp'], out_stream=None), atol=1e-5)
        assert_check_totals(p.check_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'], out_stream=None))
