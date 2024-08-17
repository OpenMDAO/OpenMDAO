import unittest
import sys
from scipy import linalg

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
import openmdao.api as om
from openmdao.utils.jax_utils import jax, ImplicitCompJaxify


class QuadraticComp(om.ImplicitComponent):
    def __init__(self, shape=(), **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def setup(self):
        self.add_input('a', shape=self.shape)
        self.add_input('b', shape=self.shape)
        self.add_input('c', shape=self.shape)
        self.add_output('x', val=5., shape=self.shape)

        self.declare_partials(of=['*'], wrt=['*'])

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()

    def apply_nonlinear(self, inputs, outputs, residuals):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        residuals['x'] = a * x ** 2 + b * x + c



class JaxQuadraticCompPrimal(om.JaxImplicitComponent):
    def __init__(self, shape=(), **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def setup(self):
        self.add_input('a', shape=self.shape)
        self.add_input('b', shape=self.shape)
        self.add_input('c', shape=self.shape)
        self.add_output('x', val=5., shape=self.shape)

        self.declare_partials(of=['*'], wrt=['*'])

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()

    def compute_primal(self, a, b, c, x):
        return a * x ** 2 + b * x + c



class SimpleLinearSystemComp(om.ImplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lup = None

    def initialize(self):
        self.options.declare('size', default=1, types=int)

    def setup(self):
        self.options['derivs_method'] = 'jax'
        size = self.options['size']

        self._lup = None
        shape = (size, )

        self.add_input("A", val=np.eye(size))
        self.add_input("b", val=np.ones(shape))
        self.add_output("x", shape=shape, val=.1)

        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

    def setup_partials(self):
        size = self.options['size']
        mat_size = size * size
        full_size = size

        row_col = np.arange(full_size, dtype="int")

        self.declare_partials('x', 'b', val=np.full(full_size, -1.0), rows=row_col, cols=row_col)

        rows = np.repeat(np.arange(full_size), size)

        cols = np.arange(mat_size)

        self.declare_partials('x', 'A', rows=rows, cols=cols)

        cols = np.tile(np.arange(size), size)
        cols += np.repeat(np.arange(1), mat_size) * size

        self.declare_partials(of='x', wrt='x', rows=rows, cols=cols)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x'] = inputs['A'].dot(outputs['x']) - inputs['b']



class JaxLinearSystemCompPrimal(om.JaxImplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lup = None

    def initialize(self):
        self.options.declare('size', default=1, types=int)

    def setup(self):
        size = self.options['size']

        self._lup = None
        shape = (size, )

        self.add_input("A", val=np.eye(size))
        self.add_input("b", val=np.ones(shape))
        self.add_output("x", shape=shape, val=.1)

        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

    def setup_partials(self):
        size = self.options['size']
        mat_size = size * size
        full_size = size

        row_col = np.arange(full_size, dtype="int")

        self.declare_partials('x', 'b', val=np.full(full_size, -1.0), rows=row_col, cols=row_col)

        rows = np.repeat(np.arange(full_size), size)

        cols = np.arange(mat_size)

        self.declare_partials('x', 'A', rows=rows, cols=cols)

        cols = np.tile(np.arange(size), size)
        cols += np.repeat(np.arange(1), mat_size) * size

        self.declare_partials(of='x', wrt='x', rows=rows, cols=cols)

    def compute_primal(self, A, b, x):
        return A.dot(x) - b


@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
class TestJaxAST(unittest.TestCase):
    def test_ast_continuous(self):
        class ASTContinuousCompTester(om.ImplicitComponent):
            def setup(self):
                self.add_input('in_scalar', val=7.0)
                self.add_input('in_array', val=np.ones((2, 3)))
                self.add_input('in_array2', val=np.ones((3,4)))
                self.add_output('out_scalar', val=5.0)
                self.add_output('out_array', val=np.ones((2, 3)))
                self.add_output('out_array2', val=np.ones((3, 4)))

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['out_scalar'] = inputs['in_scalar'] * 2.0
                residuals['out_array'] = inputs['in_array'] * 2.0
                residuals['out_array2'] = np.dot(inputs['in_array'], inputs['in_array2'])

        def check_prob(multi_setup=False):
            p = om.Problem()
            comp = p.model.add_subsystem('comp', ASTContinuousCompTester())
            p.setup()
            p.final_setup()

            if multi_setup:
                p.setup()  # make sure a second setup call doesn't break the conversion

            converter = ImplicitCompJaxify(comp, verbose=True)

            expected = """
def compute_primal(self, in_scalar, in_array, in_array2, out_scalar, out_array, out_array2):
    out_scalar = in_scalar * 2.0
    out_array = in_array * 2.0
    out_array2 = jnp.dot(in_array, in_array2)
    return (out_scalar, out_array, out_array2)
""".strip()

            self.assertEqual(converter.get_compute_primal_src().strip(), expected)

        check_prob(multi_setup=False)
        check_prob(multi_setup=True)

    def test_ast_discrete(self):
        class ASTDiscreteCompTester(om.ImplicitComponent):
            def setup(self):
                self.add_input('in_scalar', val=7.0)
                self.add_input('in_array', val=np.ones((2, 3)))
                self.add_input('in_array2', val=np.ones((3,4)))
                self.add_output('out_scalar', val=5.0)
                self.add_output('out_array', val=np.ones((2, 3)))
                self.add_output('out_array2', val=np.ones((3, 4)))
                self.add_discrete_input('disc_in', val=2)
                self.add_discrete_output('disc_out', val=3)

            def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs, discrete_outputs):
                outputs['out_scalar'] = inputs['in_scalar'] * 2.0
                outputs['out_array'] = inputs['in_array'] * 2.0
                outputs['out_array2'] = np.dot(inputs['in_array'], inputs['in_array2'])
                if discrete_inputs['disc_in'] > 0:
                    outputs['out_scalar'] *= 2.0
                    outputs['out_array'] *= 2.0
                    outputs['out_array2'] *= 2.0
                else:
                    outputs['out_scalar'] *= 3.0
                    outputs['out_array'] *= 3.0
                    outputs['out_array2'] *= 3.0

        p = om.Problem()
        comp = p.model.add_subsystem('comp', ASTDiscreteCompTester())
        p.setup()
        p.final_setup()

        converter = ImplicitCompJaxify(comp)

        expected = """
def compute_primal(self, disc_in, in_scalar, in_array, in_array2, out_scalar, out_array, out_array2):
    disc_out, = self._discrete_outputs.values()
    out_scalar = in_scalar * 2.0
    out_array = in_array * 2.0
    out_array2 = jnp.dot(in_array, in_array2)
    if disc_in > 0:
        out_scalar *= 2.0
        out_array *= 2.0
        out_array2 *= 2.0
    else:
        out_scalar *= 3.0
        out_array *= 3.0
        out_array2 *= 3.0
    self._discrete_outputs.set_vals((disc_out,))
    return (disc_out, out_scalar, out_array, out_array2)
""".strip()

        self.assertEqual(converter.get_compute_primal_src().strip(), expected)


@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
class TestJaxImplicitComp(unittest.TestCase):

    def test_quad_comp_converted(self):
        p = om.Problem()
        # create an IVC manually so we can set the shapes.  Otherwise must set shape in the component
        # itself.
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('a'))
        ivc.add_output('b')
        ivc.add_output('c')
        comp = p.model.add_subsystem('comp', QuadraticComp(derivs_method='jax'))
        p.model.connect('ivc.a', 'comp.a')
        p.model.connect('ivc.b', 'comp.b')
        p.model.connect('ivc.c', 'comp.c')
        p.setup(mode='rev')

        p.set_val('ivc.a', 1.0)
        p.set_val('ivc.b', -4.0)
        p.set_val('ivc.c', 3.0)

        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.x'), 3.0)
        assert_check_totals(p.check_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'], method='fd',
                                           show_only_incorrect=True, abs_err_tol=3e-5, rel_err_tol=5e-6), atol=3e-5, rtol=5e-6)
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    def test_jax_quad_comp(self):
        p = om.Problem()
        # create an IVC manually so we can set the shapes.  Otherwise must set shape in the component
        # itself.
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('a'))
        ivc.add_output('b')
        ivc.add_output('c')
        comp = p.model.add_subsystem('comp', JaxQuadraticCompPrimal())
        p.model.connect('ivc.a', 'comp.a')
        p.model.connect('ivc.b', 'comp.b')
        p.model.connect('ivc.c', 'comp.c')
        p.setup(mode='rev')

        p.set_val('ivc.a', 1.0)
        p.set_val('ivc.b', -4.0)
        p.set_val('ivc.c', 3.0)

        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.x'), 3.0)
        assert_check_totals(p.check_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'], method='fd',
                                           show_only_incorrect=True, abs_err_tol=3e-5, rel_err_tol=5e-6), atol=3e-5, rtol=5e-6)
        assert_check_partials(p.check_partials(show_only_incorrect=True))

    def test_lin_system_converted(self):
        x = np.array([1, 1, 1])
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = np.array([1, 2, -3])

        prob = om.Problem()

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('A', A)
        ivc.add_output('b', b)

        lingrp = prob.model.add_subsystem('lingrp', om.Group())
        lingrp.add_subsystem('lin', SimpleLinearSystemComp(size=3, derivs_method='jax'))

        prob.model.connect('ivc.A', 'lingrp.lin.A')
        prob.model.connect('ivc.b', 'lingrp.lin.b')

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()

        assert_near_equal(prob['lingrp.lin.x'], np.array([-4, 9, -4]))

        assert_check_totals(prob.check_totals(of=['lingrp.lin.x'], wrt=['ivc.b', 'ivc.A'],
                                              abs_err_tol=2e-4, rel_err_tol=3e-6, show_only_incorrect=True),
                            atol=2e-4, rtol=3e-6)
        assert_check_partials(prob.check_partials(show_only_incorrect=True), rtol=1e-5)

    def test_jax_lin_system(self):
        x = np.array([1, 1, 1])
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = np.array([1, 2, -3])

        prob = om.Problem()

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('A', A)
        ivc.add_output('b', b)

        lingrp = prob.model.add_subsystem('lingrp', om.Group())
        lingrp.add_subsystem('lin', JaxLinearSystemCompPrimal(size=3))

        prob.model.connect('ivc.A', 'lingrp.lin.A')
        prob.model.connect('ivc.b', 'lingrp.lin.b')

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()

        assert_near_equal(prob['lingrp.lin.x'], np.array([-4, 9, -4]))

        assert_check_totals(prob.check_totals(of=['lingrp.lin.x'], wrt=['ivc.b', 'ivc.A'],
                                              abs_err_tol=2e-4, rel_err_tol=3e-6, show_only_incorrect=True),
                            atol=2e-4, rtol=3e-6)
        assert_check_partials(prob.check_partials(show_only_incorrect=True), rtol=1e-5)

    # def test_jax_implicit_comp_primal(self):
    #     N_cp = 50
    #     N_predict = 1000

    #     p = om.Problem()

    #     fit = p.model.add_subsystem('fit', PolynomialFitJaxPrimal(N_cp=N_cp, N_predict=N_predict))
    #     fit.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=10, iprint=2)
    #     fit.linear_solver = om.DirectSolver()

    #     p.setup(force_alloc_complex=True)

    #     np.random.seed(0)

    #     X_cp = np.linspace(0, np.pi, N_cp)
    #     Y_cp = np.sin(X_cp) + (np.random.random(N_cp) - 0.5)*0.1

    #     X_true = np.linspace(0, np.pi, N_predict)
    #     Y_true = np.sin(X_true)

    #     p['fit.x_cp'] = X_cp
    #     p['fit.y_cp'] = Y_cp
    #     p['fit.x'] = X_true

    #     p.run_model()

    #     assert_near_equal(p['fit.y'], Y_true, 1e-1)

    #     assert_check_totals(p.check_totals(of=['fit.A', 'fit.y'], wrt=['fit.x_cp', 'fit.y_cp', 'fit.x'],
    #                                        method='fd', show_only_incorrect=True))
    #     assert_check_partials(p.check_partials(method='cs', show_only_incorrect=True), rtol=1e-5)

    #def test_jax_implicit_comp2primal_w_option(self):
        ## this component defines its own compute_primal method
        #p = om.Problem()
        #ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        #ivc.add_output('y', val=np.ones(y_shape))
        #comp = p.model.add_subsystem('comp', MyCompJax2PrimalOption(derivs_method='jax', mult=1.5))
        ## deriv shape: [[(2, 4, 2, 3), (2, 4, 3, 4)], [(3, 4, 2, 3), (3, 4, 3, 4)]]
        #p.model.connect('ivc.x', 'comp.x')
        #p.model.connect('ivc.y', 'comp.y')

        #p.setup(mode='rev')

        #x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        #y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape) * 3.0
        #p.set_val('ivc.x', x)
        #p.set_val('ivc.y', y)
        #p.final_setup()
        #p.run_model()

        #assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        #assert_near_equal(p.get_val('comp.zz'), y * 3.0)
        #p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        #p.check_partials(show_only_incorrect=True)

        #comp.options['mult'] = 3.5
        #p.set_val('ivc.x', x)
        #p.set_val('ivc.y', y)
        #p.run_model()

        #assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        #assert_near_equal(p.get_val('comp.zz'), y * 7.0)
        #p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        #p.check_partials(show_only_incorrect=True)

        #comp.stat = 1./3.5
        #p.set_val('ivc.x', x)
        #p.set_val('ivc.y', y)
        #p.run_model()

        #assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        #assert_near_equal(p.get_val('comp.zz'), y)
        #p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        #p.check_partials(show_only_incorrect=True)

    #def test_jax_implicit_comp_with_option(self):
        #p = om.Problem()
        #ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        #ivc.add_output('y', val=np.ones(y_shape))
        #comp = p.model.add_subsystem('comp', MyCompJaxWithOption(mult=1.7, derivs_method='jax'))
        #p.model.connect('ivc.x', 'comp.x')
        #p.model.connect('ivc.y', 'comp.y')

        #p.setup(mode='rev')

        #x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        #y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        #p.set_val('ivc.x', x)
        #p.set_val('ivc.y', y)
        #p.final_setup()
        #p.run_model()

        #assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        #assert_near_equal(p.get_val('comp.zz'), y * 1.7)
        #p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        #p.check_partials(show_only_incorrect=True)

        #p.set_val('ivc.x', x)
        #p.set_val('ivc.y', y)
        #comp.options['mult'] = 1.9
        #p.run_model()
        #assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        #assert_near_equal(p.get_val('comp.zz'), y * 1.9)

    #def test_jax_implicit_comp_with_discrete(self):
        #p = om.Problem()
        #ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        #ivc.add_output('y', val=np.ones(y_shape))
        #ivc.add_discrete_output('disc_out', val=3)
        #p.model.add_subsystem('comp', MyCompJaxWithDiscrete(derivs_method='jax'))
        #p.model.connect('ivc.x', 'comp.x')
        #p.model.connect('ivc.y', 'comp.y')
        #p.model.connect('ivc.disc_out', 'comp.disc_in')

        #p.setup(mode='rev')

        #x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        #y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        #p.set_val('ivc.x', x)
        #p.set_val('ivc.y', y)
        #p.final_setup()
        #p.run_model()

        #assert_near_equal(p.get_val('comp.z'), np.dot(x, y))
        #assert_near_equal(p.get_val('comp.zz'), y * 3.0)
        #p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        #p.check_partials(show_only_incorrect=True)

        #p.set_val('ivc.disc_out', -2)
        #p.run_model()
        #assert_near_equal(p.get_val('comp.z'), -np.dot(x, y))
        #assert_near_equal(p.get_val('comp.zz'), y * 2.5)
        #p.check_totals(of=['comp.z','comp.zz'], wrt=['comp.x', 'comp.y'], method='fd', show_only_incorrect=True)
        #p.check_partials(show_only_incorrect=True)


#@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
#class TestJaxGroup(unittest.TestCase):
    #def test_jax_group_outer_ivc(self):
        #p = om.Problem()
        #ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        #ivc.add_output('y', val=np.ones(y_shape))
        #G = p.model.add_subsystem('G', om.Group())
        #G.options['derivs_method'] = 'jax'
        #G.add_subsystem('comp2', MyCompJax2())
        #G.add_subsystem('comp', MyCompJax1())

        #p.model.connect('ivc.x', ['G.comp.x', 'G.comp2.x'])
        #p.model.connect('ivc.y', 'G.comp2.y')
        #G.connect('comp2.zz', 'comp.y')

        #p.setup(mode='rev')

        #x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        #y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        #p.set_val('ivc.x', x)
        #p.set_val('ivc.y', y)
        #p.final_setup()
        #p.run_model()

        #assert_near_equal(p.get_val('G.comp2.z'), np.dot(x, y))
        #assert_near_equal(p.get_val('G.comp2.zz'), y * 2.5)
        #assert_check_partials(p.check_partials(show_only_incorrect=True))
        #assert_check_totals(p.check_totals(of=['G.comp.z','G.comp2.z', 'G.comp2.zz'],
                                             #wrt=['G.comp2.x', 'G.comp2.y'], method='fd', show_only_incorrect=True))

    #def test_jax_group_auto_ivc(self):
        #p = om.Problem()
        #G = p.model.add_subsystem('G', om.Group())
        #G.options['derivs_method'] = 'jax'
        #G.add_subsystem('comp2', MyCompJax2Shaped(x_shape, y_shape))
        #G.add_subsystem('comp', MyCompJax1Shaped(x_shape, y_shape))

        #G.connect('comp2.zz', 'comp.y')

        #p.setup(mode='rev')

        #x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        #y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        #p.set_val('G.comp.x', x)
        #p.set_val('G.comp2.x', x)
        #p.set_val('G.comp2.y', y)
        #p.final_setup()
        #p.run_model()

        #assert_near_equal(p.get_val('G.comp2.z'), np.dot(x, y))
        #assert_near_equal(p.get_val('G.comp2.zz'), y * 2.5)
        #assert_check_partials(p.check_partials(show_only_incorrect=True))
        #assert_check_totals(p.check_totals(of=['G.comp.z','G.comp2.z', 'G.comp2.zz'],
                                             #wrt=['G.comp2.x', 'G.comp2.y'], method='fd', show_only_incorrect=True))

    #@unittest.skip("unskip this after inner ivc issue is fixed for fd")
    #def test_jax_group_inner_ivc(self):
        #p = om.Problem()
        #G = p.model.add_subsystem('G', om.Group())
        #ivc = G.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        #ivc.add_output('y', val=np.ones(y_shape))
        #G.options['derivs_method'] = 'jax'
        #G.add_subsystem('comp2', MyCompJax2())
        #G.add_subsystem('comp', MyCompJax1())

        #p.model.connect('G.ivc.x', ['G.comp.x', 'G.comp2.x'])
        #p.model.connect('G.ivc.y', 'G.comp2.y')
        #G.connect('comp2.zz', 'comp.y')

        #p.setup(mode='fwd')

        #x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        #y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape)* 3.0
        #p.set_val('G.ivc.x', x)
        #p.set_val('G.ivc.y', y)
        #p.final_setup()
        #p.run_model()

        #assert_near_equal(p.get_val('G.comp2.z'), np.dot(x, y))
        #assert_near_equal(p.get_val('G.comp2.zz'), y * 2.5)

        #print('-' * 80)
        #assert_check_partials(p.check_partials(show_only_incorrect=True))
        #assert_check_totals(p.check_totals(of=['G.comp.z','G.comp2.z', 'G.comp2.zz'],
                                             #wrt=['G.ivc.x', 'G.ivc.y'], method='fd', show_only_incorrect=True))

    #def test_jax_group_top_level(self):
        #p = om.Problem()
        #ivc = p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.ones(x_shape)))
        #ivc.add_output('y', val=np.ones(y_shape))
        #G = p.model
        #G.options['derivs_method'] = 'jax'
        #G.add_subsystem('comp2', MyCompJax2())
        #G.add_subsystem('comp', MyCompJax1())

        #p.model.connect('ivc.x', ['comp.x', 'comp2.x'])
        #p.model.connect('ivc.y', 'comp2.y')
        #G.connect('comp2.zz', 'comp.y')

        #p.setup(mode='rev')

        #x = np.arange(1,np.prod(x_shape)+1).reshape(x_shape) * 2.0
        #y = np.arange(1,np.prod(y_shape)+1).reshape(y_shape) * 3.0
        #p.set_val('ivc.x', x)
        #p.set_val('ivc.y', y)
        #p.final_setup()
        #p.run_model()

        #assert_near_equal(p.get_val('comp2.z'), np.dot(x, y))
        #assert_near_equal(p.get_val('comp2.zz'), y * 2.5)
        #assert_check_partials(p.check_partials(show_only_incorrect=True))
        #assert_check_totals(p.check_totals(of=['comp.z','comp2.z', 'comp2.zz'],
                                             #wrt=['ivc.x', 'comp2.y'], method='fd', show_only_incorrect=True))


#if sys.version_info >= (3, 9):

    #class CompRetValue(om.JaxImplicitComponent):
        #def __init__(self, shape, nins=1, nouts=1, **kwargs):
            #super().__init__(**kwargs)
            #self.shape = shape
            #self.nins = nins
            #self.nouts = nouts

            #self.compute_primal = getattr(self, f'compute_primal_{self.nins}_{self.nouts}')

        #def setup(self):
            #if self.shape == ():
                #for i in range(self.nins):
                    #self.add_input(f'x{i}', val=0.)
                #for i in range(self.nouts):
                    #self.add_output(f'y{i}', val=0.)
            #else:
                #for i in range(self.nins):
                    #self.add_input(f'x{i}', val=jnp.zeros(self.shape))
                #for i in range(self.nouts):
                    #self.add_output(f'y{i}', val=jnp.zeros(self.shape))

        #def setup_partials(self):
            #self.declare_partials('*', '*')

        #def compute_primal_1_1(self, x0):
            #return x0**2

        #def compute_primal_2_1(self, x0, x1):
            #return x0**2 + x1**2

        #def compute_primal_1_2(self, x0):
            #return x0**2, x0*2

        #def compute_primal_2_2(self, x0, x1):
            #return x0**2, x1**2

    #class CompRetTuple(om.JaxImplicitComponent):
        #def __init__(self, shape, nins=1, nouts=1, **kwargs):
            #super().__init__(**kwargs)
            #self.shape = shape
            #self.nins = nins
            #self.nouts = nouts

            #self.compute_primal = getattr(self, f'compute_primal_{self.nins}_{self.nouts}')

        #def setup(self):
            #if self.shape == ():
                #for i in range(self.nins):
                    #self.add_input(f'x{i}', val=0.)
                #for i in range(self.nouts):
                    #self.add_output(f'y{i}', val=0.)
            #else:
                #for i in range(self.nins):
                    #self.add_input(f'x{i}', val=jnp.zeros(self.shape))
                #for i in range(self.nouts):
                    #self.add_output(f'y{i}', val=jnp.zeros(self.shape))

        #def setup_partials(self):
            #self.declare_partials('*', '*')

        #def compute_primal_1_1(self, x0):
            #return (x0**2,)

        #def compute_primal_2_1(self, x0, x1):
            #return (x0**2 + x1**2,)

        #def compute_primal_1_2(self, x0):
            #return x0**2, x0*2

        #def compute_primal_2_2(self, x0, x1):
            #return x0**2, x1**2


    #class TopGrp(om.Group):
        #def __init__(self, shape, ret_tuple=False, nins=1, nouts=1, **kwargs):
            #super().__init__(**kwargs)
            #self.shape = shape
            #self.ret_tuple = ret_tuple
            #self.nins = nins
            #self.nouts = nouts

        #def setup(self):
            #self.add_subsystem('ivc', om.IndepVarComp())
            #if self.shape == ():
                #for i in range(self.nins):
                    #self.ivc.add_output(f'x{i}', 0.)

                #if self.ret_tuple:
                    #self.add_subsystem('comp', CompRetTuple(shape=self.shape, nins=self.nins, nouts=self.nouts))
                #else:
                    #self.add_subsystem('comp', CompRetValue(shape=self.shape, nins=self.nins, nouts=self.nouts))
            #else:
                #for i in range(self.nins):
                    #self.ivc.add_output(f'x{i}', np.zeros(self.shape))
                #if self.ret_tuple:
                    #self.add_subsystem('comp', CompRetTuple(shape=self.shape, nins=self.nins, nouts=self.nouts))
                #else:
                    #self.add_subsystem('comp', CompRetValue(shape=self.shape, nins=self.nins, nouts=self.nouts))

            #for io in range(self.nouts):
                #for ii in range(self.nins):
                    #if ii == io:
                        #self.connect(f'ivc.x{io}', f'comp.x{ii}')


#def _test_func_name(func, num, param):
    #args = []
    #for p in param.args:
        #if isinstance(p, str) or not isinstance(p, Iterable):
            #p = [p]
        #for item in p:
            #try:
                #arg = item.__name__
            #except:
                #arg = str(item)
            #args.append(arg)
    #return func.__name__ + '_' + '_'.join(args)


#@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
#class TestJaxShapesAndReturns(unittest.TestCase):
    #@parameterized.expand(itertools.product([(), (2,), (2,3)], [(1, 1), (2, 2), (1, 2), (2, 1)],[True, False]),
                          #name_func=_test_func_name)
    #def test_compute_primal_return_shapes(self, shape, sizetup, ret_tuple):
        #nins, nouts = sizetup
        #prob = om.Problem()
        #prob.model = model = TopGrp(shape=shape, ret_tuple=ret_tuple, nins=nins, nouts=nouts)

        #prob.set_solver_print(level=0)

        #ofs = [f'comp.y{i}' for i in range(nouts)]
        #wrts = [f'ivc.x{i}' for i in range(nins)]

        #prob.setup(force_alloc_complex=True, check=False, mode='fwd')
        #prob.final_setup()
        #prob.compute_totals(of=ofs, wrt=wrts)

        #assert_check_partials(prob.check_partials(method='cs', out_stream=None), atol=1e-5)
        #assert_check_totals(prob.check_totals(of=ofs, wrt=wrts, method='cs', out_stream=None), atol=1e-5)

    # TODO: test with mixed np and jnp in compute

if __name__ == '__main__':
    unittest.main()

    # from openmdao.utils.jax_utils import benchmark_component
    # result = benchmark_component(MyCompJax2Shaped, methods=('jax', 'cs'),
    #                              repeats=10, table_format='tabulator', xshape=(44, 330), yshape=(330, 55))
