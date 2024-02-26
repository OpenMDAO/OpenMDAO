import unittest
import math

import numpy as np
from scipy.optimize import newton

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import openmdao.func_api as omf
from openmdao.core.tests.test_partial_color import _check_partial_matrix

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None


class TestImplicitFuncComp(unittest.TestCase):
    def test_apply_nonlinear_linsys(self):

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

    def setup_apply_nonlinear_linsys_coloring(self, method, mode):

        x_=np.array([[1], [2], [-3]])
        A = np.array([[5.0, 0., 2.0], [1.0, 7.0, 0.], [1.0, 0.0, 0.]])
        if mode == 'rev':
            A = A.T
        b = A.dot(x_)

        def resid_func(A, b, x):
            rx = A.dot(x) - b
            return rx

        f = (omf.wrap(resid_func)
                .add_input('A', shape=A.shape)
                .add_input('b', shape=b.shape)
                .add_output('x', resid='rx', val=x_)
                .declare_coloring(wrt='*', method=method)
                )

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('A', A))
        model.add_subsystem('p2', om.IndepVarComp('b', b))

        comp = model.add_subsystem('comp', om.ImplicitFuncComp(f))

        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=0)
        comp.linear_solver = om.DirectSolver(assemble_jac=True)

        model.connect('p1.A', 'comp.A')
        model.connect('p2.b', 'comp.b')

        prob.setup(mode=mode)

        prob.set_solver_print(level=0)
        prob.run_model()
        return prob

    def test_apply_nonlinear_linsys_coloring_cs(self):
        prob = self.setup_apply_nonlinear_linsys_coloring('cs', 'fwd')
        partials = prob.check_partials(includes=['comp'], out_stream=None)
        assert_check_partials(partials, atol=1e-5)
        assert_check_totals(prob.check_totals(of=['comp.x'], wrt=['comp.A', 'comp.b'], out_stream=None), atol=3e-5, rtol=3e-5)

    @unittest.skipIf(jax is None, "jax is not installed")
    def test_apply_nonlinear_linsys_coloring_jax_fwd(self):
        prob = self.setup_apply_nonlinear_linsys_coloring('jax', 'fwd')
        partials = prob.check_partials(includes=['comp'], out_stream=None)
        assert_check_partials(partials, atol=1e-5)
        assert_check_totals(prob.check_totals(of=['comp.x'], wrt=['comp.A', 'comp.b'], out_stream=None), atol=3e-5, rtol=3e-5)

    @unittest.skipIf(jax is None, "jax is not installed")
    def test_apply_nonlinear_linsys_coloring_jax_rev(self):
        prob = self.setup_apply_nonlinear_linsys_coloring('jax', 'rev')
        partials = prob.check_partials(includes=['comp'], out_stream=None)
        assert_check_partials(partials, atol=1e-5)
        assert_check_totals(prob.check_totals(of=['comp.x'], wrt=['comp.A', 'comp.b'], out_stream=None), atol=3e-5, rtol=3e-5)

    def test_component_model_w_linearize(self):

        def apply_nonlinear(z, x, y1, y2, R_y1, R_y2):

            z0 = z[0]
            z1 = z[1]

            return (
                0.,
                (y1**.5 + z0 + z1) - y2,
                (z0**2 + z1 + x - 0.2*y2) - y1 - R_y1,
                (y1**.5 + z0 + z1) - y2 - R_y2
            )

        f = (omf.wrap(apply_nonlinear)
                .add_input('z', val=np.array([-1., -1.]))
                .add_input('x', val=2.)
                .add_output('y1', resid='r_y1')
                .add_output('y2', resid='r_y2')
                .add_output('R_y1', resid='r_R_y1')
                .add_output('R_y2', resid='r_R_y1')
                .declare_partials('y1', 'y1')
                .declare_partials('y2', ['z', 'y1', 'y2', 'x'])
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

        p_opt.model.add_subsystem('comp', om.ImplicitFuncComp(f, linearize=linearize,), promotes=['*'])

        p_opt.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=0)
        p_opt.model.linear_solver = om.DirectSolver(assemble_jac=True)

        p_opt.driver = om.ScipyOptimizeDriver()
        p_opt.driver.options['disp'] = False

        p_opt.model.add_design_var('x', lower=-10, upper=10)
        p_opt.model.add_constraint('R_y1', equals=0)

        p_opt.model.add_objective('y2')

        p_opt.setup()

        # set
        p_opt['y2'] = 5
        p_opt['y1'] = 5

        p_opt.run_driver()

        np.testing.assert_almost_equal(p_opt['y1'], 5., decimal=5)
        np.testing.assert_almost_equal(p_opt['y2'], 0.2360679774997898, decimal=5)
        np.testing.assert_almost_equal(p_opt['x'], 5.047213595499958, decimal=5)
        np.testing.assert_almost_equal(p_opt['z'], np.array([-1., -1.]), decimal=5)

    def test_apply_nonlinear(self):

        def apply_nl(a, b, c, x):
            R_x = a * x ** 2 + b * x + c
            return R_x

        f = (omf.wrap(apply_nl)
                .add_output('x', resid='R_x', val=0.0)
                .declare_partials(of='*', wrt='*', method='cs')
                )

        p = om.Problem()
        p.model.add_subsystem('comp', om.ImplicitFuncComp(f))

        # need this since comp is implicit and doesn't have a solve_linear
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=0)

        p.setup()

        p.set_val('comp.a', 2.)
        p.set_val('comp.b', -8.)
        p.set_val('comp.c', 6.)
        p.run_model()

        assert_check_partials(p.check_partials(includes=['comp'], out_stream=None), atol=1e-5)
        assert_check_totals(p.check_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'], out_stream=None))

    def test_apply_nonlinear_option(self):

        def apply_nl(a, b, c, x, opt):
            R_x = a * x ** 2 + b * x + c
            if opt == 'foo':
                R_x = -R_x
            return R_x

        f = (omf.wrap(apply_nl)
                .add_output('x', resid='R_x', val=0.0)
                .declare_option('opt', default='foo')
                .declare_partials(of='*', wrt='*', method='cs')
                )

        p = om.Problem()
        p.model.add_subsystem('comp', om.ImplicitFuncComp(f))

        # need this since comp is implicit and doesn't have a solve_linear
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=0)

        p.setup()

        p.set_val('comp.a', 2.)
        p.set_val('comp.b', -8.)
        p.set_val('comp.c', 6.)
        p.run_model()

        assert_check_partials(p.check_partials(includes=['comp'], out_stream=None), atol=1e-5)
        assert_check_totals(p.check_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'], out_stream=None))

    def test_apply_nonlinear_no_method(self):

        def apply_nl(a, b, c, x):
            R_x = a * x ** 2 + b * x + c
            return R_x

        f = (omf.wrap(apply_nl)
                .add_output('x', resid='R_x', val=0.0)
                .declare_partials(of='*', wrt='*')
                )

        p = om.Problem()
        p.model.add_subsystem('comp', om.ImplicitFuncComp(f))

        p.setup()

        with self.assertRaises(RuntimeError) as cm:
            p.run_model()

        self.assertEqual(cm.exception.args[0],
                         "'comp' <class ImplicitFuncComp>: declare_partials must be called with method equal to 'cs', 'fd', or 'jax'.")

    def test_solve_nonlinear(self):

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

    def _solve_lin_nl_linearize(self, mode):

        def apply_nl(a, b, c, x):
            R_x = a * x ** 2 + b * x + c
            return R_x

        def solve_nl(a, b, c, x):
            x = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
            return x

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
        p.model.add_subsystem('comp', om.ImplicitFuncComp(f,
                                                          solve_linear=solve_linear,
                                                          linearize=linearize,
                                                          solve_nonlinear=solve_nl))

        p.setup(mode=mode)

        p.set_val('comp.a', 2.)
        p.set_val('comp.b', -8.)
        p.set_val('comp.c', 6.)
        p.run_model()

        assert_check_partials(p.check_partials(includes=['comp'], out_stream=None), atol=1e-5)
        assert_check_totals(p.check_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'], out_stream=None))

    def test_solve_lin_nl_linearize_fwd(self):
        self._solve_lin_nl_linearize('fwd')

    def test_solve_lin_nl_linearize_rev(self):
        self._solve_lin_nl_linearize('rev')

    def test_solve_lin_nl_linearize_reordered_args(self):

        def apply_nl(x, a, b, c):
            R_x = a * x ** 2 + b * x + c
            return R_x

        def solve_nl(x, a, b, c):
            x = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
            return x

        def linearize(x, a, b, c, partials):
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
        p.model.add_subsystem('comp', om.ImplicitFuncComp(f,
                                                          solve_linear=solve_linear,
                                                          linearize=linearize,
                                                          solve_nonlinear=solve_nl))

        p.setup()

        p.set_val('comp.a', 2.)
        p.set_val('comp.b', -8.)
        p.set_val('comp.c', 6.)
        p.run_model()

        assert_check_partials(p.check_partials(includes=['comp'], out_stream=None), atol=1e-5)
        assert_check_totals(p.check_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'], out_stream=None))


@unittest.skipIf(jax is None, "jax is not installed")
class TestJax(unittest.TestCase):
    def check_derivs(self, mode, use_jit, nondiff):
        if nondiff:
            def apply_nl(a, b, c, x, ex1, ex2):
                R_x = a * x ** 2 + b * x + c
                return R_x

            def solve_nl(a, b, c, x, ex1, ex2):
                x = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
                return x

            f = (omf.wrap(apply_nl)
                    .add_output('x', resid='R_x', val=0.0)
                    .declare_option('ex1', default='foo')
                    .declare_option('ex2', default='bar')
                    .declare_partials(of='*', wrt='*', method='jax')
                    )
        else:
            def apply_nl(a, b, c, x):
                R_x = a * x ** 2 + b * x + c
                return R_x

            def solve_nl(a, b, c, x):
                x = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
                return x

            f = (omf.wrap(apply_nl)
                    .add_output('x', resid='R_x', val=0.0)
                    .declare_partials(of='*', wrt='*', method='jax')
                    )

        p = om.Problem()
        p.model.add_subsystem('comp', om.ImplicitFuncComp(f, solve_nonlinear=solve_nl, use_jit=use_jit))

        # need this since comp is implicit and doesn't have a solve_linear
        p.model.comp.linear_solver = om.DirectSolver()

        p.setup(check=True, mode=mode)

        p.set_val('comp.a', 1.)
        p.set_val('comp.b', -4.)
        p.set_val('comp.c', 3.)
        p.run_model()

        J = p.compute_totals(wrt=['comp.a', 'comp.b', 'comp.c'], of=['comp.x', 'comp.x'])
        assert_near_equal(J['comp.x', 'comp.a'], [[-4.5]])
        assert_near_equal(J['comp.x', 'comp.b'], [[-1.5]])
        assert_near_equal(J['comp.x', 'comp.c'], [[-0.5]])

    def test_fwd(self):
        self.check_derivs('fwd', use_jit=False, nondiff=False)

    def test_fwd_jit(self):
        self.check_derivs('fwd', use_jit=True, nondiff=False)

    def test_rev(self):
        self.check_derivs('rev', use_jit=False, nondiff=False)

    def test_rev_jit(self):
        self.check_derivs('rev', use_jit=True, nondiff=False)

    def test_fwd_nondiff(self):
        self.check_derivs('fwd', use_jit=False, nondiff=True)

    def test_fwd_jit_nondiff(self):
        self.check_derivs('fwd', use_jit=True, nondiff=True)

    def test_rev_nondiff(self):
        self.check_derivs('rev', use_jit=False, nondiff=True)

    def test_rev_jit_nondiff(self):
        self.check_derivs('rev', use_jit=True, nondiff=True)

    def _partials_implicit_2in2out(self, mode, method, use_jit):

        def apply_nl(x1, x2, y1, y2):
            mat = np.array([[0.11534105, 0.9799784 , 0.        , 0.86227559, 0.        ],
                            [0.        , 0.20731316, 0.89688114, 0.96884353, 0.        ],
                            [0.        , 0.        , 0.97299632, 0.68203646, 0.75099805],
                            [0.42413042, 0.7716147 , 0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.51819104, 0.        , 0.43046408]])
            vec = np.hstack((x1 - y1, x2 - y2))
            result = mat.dot(vec)
            return result[:3], result[3:]

        f = (omf.wrap(apply_nl)
             .add_input('x1', shape=3)
             .add_input('x2', shape=2)
             .add_output('y1', resid='R_y1', shape=3)
             .add_output('y2', resid='R_y2', shape=2)
             .declare_partials(of='*', wrt='*', method=method)
            )

        prob = om.Problem()
        model = prob.model

        comp = model.add_subsystem('comp', om.ImplicitFuncComp(f, use_jit=use_jit))
        comp.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=0)
        comp.linear_solver = om.DirectSolver()

        prob.setup(check=False, mode=mode)

        prob['comp.y1'] = -3.
        prob['comp.y2'] = 3
        prob.set_solver_print(level=0)
        prob.run_model()

        jac = comp._jacobian._subjacs_info
        # redefining the same mat as used above here because mat above needs to be
        # internal to the func in order to be contained in a jax_context (which converts
        # np to jnp and numpy to jnp to make jax happy but allowing user to define the function
        # using normal numpy)
        check = np.array([[0.11534105, 0.9799784 , 0.        , 0.86227559, 0.        ],
                            [0.        , 0.20731316, 0.89688114, 0.96884353, 0.        ],
                            [0.        , 0.        , 0.97299632, 0.68203646, 0.75099805],
                            [0.42413042, 0.7716147 , 0.        , 0.        , 0.        ],
                            [0.        , 0.        , 0.51819104, 0.        , 0.43046408]])
        _check_partial_matrix(comp, jac, check, method)

    def test_partials_implicit_2in2out_fwd_jax_nojit(self):
        self._partials_implicit_2in2out(mode='fwd', method='jax', use_jit=False)

    def test_partials_implicit_2in2out_fwd_jax_jit(self):
        self._partials_implicit_2in2out(mode='fwd', method='jax', use_jit=True)

    def test_partials_implicit_2in2out_rev_jax_nojit(self):
        self._partials_implicit_2in2out(mode='rev', method='jax', use_jit=False)

    def test_partials_implicit_2in2out_rev_jax_jit(self):
        self._partials_implicit_2in2out(mode='rev', method='jax', use_jit=True)

    def test_partials_implicit_2in2out_fwd_cs_nojit(self):
        self._partials_implicit_2in2out(mode='fwd', method='cs', use_jit=False)

if __name__ == "__main__":
    unittest.main()
