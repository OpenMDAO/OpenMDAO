import unittest
import sys
import itertools

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
from openmdao.utils.jax_utils import jax, ImplicitCompJaxify
from openmdao.utils.testing_utils import parameterized_name


try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized


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
    def __init__(self, shape=(), assemble_jac=True, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.assemble_jac = assemble_jac

    def setup(self):
        self.add_input('a', shape=self.shape)
        self.add_input('b', shape=self.shape)
        self.add_input('c', shape=self.shape)
        self.add_output('x', val=5., shape=self.shape)

        self.declare_partials(of=['*'], wrt=['*'])

        self.linear_solver = om.DirectSolver(assemble_jac=self.assemble_jac)
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

    def compute_primal(self, a, b, c, x):
        return a * x ** 2 + b * x + c



class JaxLinearSystemCompConverted(om.ImplicitComponent):
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

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x'] = inputs['A'].dot(outputs['x']) - inputs['b']



class JaxLinearSystemCompPrimal(om.JaxImplicitComponent):
    def __init__(self, assemble_jac=True, **kwargs):
        super().__init__(**kwargs)
        self._lup = None
        self.assemble_jac = assemble_jac

    def initialize(self):
        self.options.declare('size', default=1, types=int)

    def setup(self):
        size = self.options['size']

        self._lup = None
        shape = (size, )

        self.add_input("A", val=np.eye(size))
        self.add_input("b", val=np.ones(shape))
        self.add_output("x", shape=shape, val=.1)

        self.linear_solver = om.DirectSolver(assemble_jac=self.assemble_jac)
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


class JaxLinearSystemCompPrimalwOption(om.JaxImplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lup = None

    def initialize(self):
        self.options.declare('size', default=1, types=int)
        self.options.declare('adder', default=1., types=float)

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

    def get_self_statics(self):
        return (self.options['adder'], )

    def compute_primal(self, A, b, x):
        return A.dot(x + self.options['adder']) - b


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
        p.model.add_subsystem('comp', QuadraticComp(derivs_method='jax'))
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

    @parameterized.expand(itertools.product([True, False], [(), (3,), (2,3)]), name_func=parameterized_name)
    def test_jax_quad_comp(self, matrix_free, shape):
        p = om.Problem()
        # create an IVC manually so we can set the shapes.  Otherwise must set shape in the component
        # itself.
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('a', shape=shape))
        ivc.add_output('b', shape=shape)
        ivc.add_output('c', shape=shape)
        comp = p.model.add_subsystem('comp', JaxQuadraticCompPrimal(shape=shape, assemble_jac=not matrix_free))
        comp.matrix_free = matrix_free
        p.model.connect('ivc.a', 'comp.a')
        p.model.connect('ivc.b', 'comp.b')
        p.model.connect('ivc.c', 'comp.c')
        p.setup(mode='rev')

        p.set_val('ivc.a', 1.0)
        p.set_val('ivc.b', -4.0)
        p.set_val('ivc.c', 3.0)

        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.x'), np.full(shape, 3.0, np.float64))
        assert_check_totals(p.check_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'], method='fd',
                                           show_only_incorrect=True, abs_err_tol=3.5e-5, rel_err_tol=5e-6), atol=3.5e-5, rtol=5e-6)
        assert_check_partials(p.check_partials(show_only_incorrect=True), atol=2.5e-6)

    def test_lin_system_converted(self):
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = np.array([1, 2, -3])

        prob = om.Problem()

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('A', A)
        ivc.add_output('b', b)

        lingrp = prob.model.add_subsystem('lingrp', om.Group())
        lingrp.add_subsystem('lin', JaxLinearSystemCompConverted(size=3, derivs_method='jax'))

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

    @parameterized.expand([True, False], name_func=parameterized_name)
    def test_jax_lin_system_primal(self, matrix_free):
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = np.array([1, 2, -3])

        prob = om.Problem()

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('A', A)
        ivc.add_output('b', b)

        lingrp = prob.model.add_subsystem('lingrp', om.Group())
        comp = lingrp.add_subsystem('lin', JaxLinearSystemCompPrimal(size=3, assemble_jac=not matrix_free))
        comp.matrix_free = matrix_free

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

    def test_jax_lin_system_primal_w_option(self):
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = np.array([1, 2, -3])

        prob = om.Problem()

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('A', A)
        ivc.add_output('b', b)

        lingrp = prob.model.add_subsystem('lingrp', om.Group())
        lin = lingrp.add_subsystem('lin', JaxLinearSystemCompPrimalwOption(size=3))

        prob.model.connect('ivc.A', 'lingrp.lin.A')
        prob.model.connect('ivc.b', 'lingrp.lin.b')

        prob.setup()
        prob.set_solver_print(level=0)

        prob.set_val('ivc.A', A)
        prob.set_val('ivc.b', b)

        prob.run_model()

        assert_near_equal(prob['lingrp.lin.x'], np.array([-5, 8, -5]))

        assert_check_totals(prob.check_totals(of=['lingrp.lin.x'], wrt=['ivc.b', 'ivc.A'],
                                              abs_err_tol=2e-4, rel_err_tol=3e-6, show_only_incorrect=True),
                            atol=2e-4, rtol=3e-6)
        assert_check_partials(prob.check_partials(show_only_incorrect=True), rtol=1e-5)

        lin.options['adder'] = 0.0

        prob.run_model()

        assert_near_equal(prob['lingrp.lin.x'], np.array([-4, 9, -4]))

        assert_check_totals(prob.check_totals(of=['lingrp.lin.x'], wrt=['ivc.b', 'ivc.A'],
                                              abs_err_tol=2e-4, rel_err_tol=3e-6, show_only_incorrect=True),
                            atol=2e-4, rtol=3e-6)
        assert_check_partials(prob.check_partials(show_only_incorrect=True), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
