import unittest
import sys
import itertools

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, assert_check_totals
from openmdao.utils.jax_utils import jax
from openmdao.utils.testing_utils import parameterized_name


try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized


class QuadraticComp(om.ImplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        self.add_input('a',)
        self.add_input('b',)
        self.add_input('c',)
        self.add_output('x', val=5.,)

        self.declare_partials(of=['*'], wrt=['*'])

    def setup_partials(self):
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        if self.matrix_free:
            self.linear_solver = om.ScipyKrylov()
        else:
            self.linear_solver = om.DirectSolver()

    def apply_nonlinear(self, inputs, outputs, residuals):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        residuals['x'] = a * x ** 2 + b * x + c



class JaxQuadraticCompPrimal(om.JaxImplicitComponent):
    def setup(self):
        self.add_input('a')
        self.add_input('b')
        self.add_input('c')
        self.add_output('x', val=5.)


    def setup_partials(self):
        if self.matrix_free:
            self.linear_solver = om.ScipyKrylov()
        else:
            self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

    def compute_primal(self, a, b, c, x):
        return a * x ** 2 + b * x + c


class JaxLinearSystemCompPrimal(om.JaxImplicitComponent):

    def initialize(self):
        self.options.declare('size', default=1, types=int)

    def setup(self):
        size = self.options['size']

        shape = (size, )

        self.add_input("A", val=np.eye(size))
        self.add_input("b", val=np.ones(shape))
        self.add_output("x", shape=shape, val=.1)

    def setup_partials(self):

        if self.matrix_free:
            self.linear_solver = om.ScipyKrylov()
        else:
            self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

    def compute_primal(self, A, b, x):
        return A.dot(x) - b


class JaxLinearSystemCompPrimalwOption(om.JaxImplicitComponent):

    def initialize(self):
        self.options.declare('size', default=1, types=int)
        self.options.declare('adder', default=1., types=float)

    def setup(self):
        size = self.options['size']

        shape = (size, )

        self.add_input("A", val=np.eye(size))
        self.add_input("b", val=np.ones(shape))
        self.add_output("x", shape=shape, val=.1)

    def setup_partials(self):

        if self.matrix_free:
            self.linear_solver = om.ScipyKrylov()
        else:
            self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

    def get_self_statics(self):
        return (self.options['adder'], )

    def compute_primal(self, A, b, x):
        return A.dot(x + self.options['adder']) - b


class BadShapeJaxLinearSystemCompPrimalwOption(JaxLinearSystemCompPrimalwOption):

    def compute_primal(self, A, b, x):
        return (A.dot(x + self.options['adder']) - b)[0]


class JaxLinearSystemCompPrimalwDiscrete(om.JaxImplicitComponent):

    def initialize(self):
        self.options.declare('size', default=1, types=int)

    def setup(self):
        size = self.options['size']

        shape = (size, )

        self.add_input("A", val=np.eye(size))
        self.add_input("b", val=np.ones(shape))
        self.add_discrete_input('c_discrete', val='1')
        self.add_output("x", shape=shape, val=.1)

    def setup_partials(self):

        if self.matrix_free:
            self.linear_solver = om.ScipyKrylov()
        else:
            self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

    def compute_primal(self, A, b, x, c_discrete):
        return A.dot(x + int(c_discrete)) - b


@unittest.skipIf(jax is None or sys.version_info < (3, 9), 'jax is not available or python < 3.9.')
class TestJaxImplicitComp(unittest.TestCase):

    @parameterized.expand(itertools.product(['fwd', 'rev'], ['matfree', ''], [(), (3,), (2,3)]), name_func=parameterized_name)
    def test_jax_quad_comp(self, mode, matrix_free, shape):
        p = om.Problem()
        # create an IVC manually so we can set the shapes.  Otherwise must set shape in the component
        # itself.
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp('a', shape=shape))
        ivc.add_output('b', shape=shape)
        ivc.add_output('c', shape=shape)
        comp = p.model.add_subsystem('comp', JaxQuadraticCompPrimal(default_shape=shape))
        comp.matrix_free = bool(matrix_free)
        p.model.connect('ivc.a', 'comp.a')
        p.model.connect('ivc.b', 'comp.b')
        p.model.connect('ivc.c', 'comp.c')
        p.setup(mode=mode)

        p.set_val('ivc.a', 1.0)
        p.set_val('ivc.b', -4.0)
        p.set_val('ivc.c', 3.0)

        p.final_setup()
        p.run_model()

        assert_near_equal(p.get_val('comp.x'), np.full(shape, 3.0, np.float64))
        assert_check_totals(p.check_totals(of=['comp.x'], wrt=['comp.a', 'comp.b', 'comp.c'], method='fd',
                                           show_only_incorrect=True, abs_err_tol=3.5e-5, rel_err_tol=5e-6), atol=3.5e-5, rtol=5e-6)
        assert_check_partials(p.check_partials(show_only_incorrect=True, abs_err_tol=3e-6, rel_err_tol=3e-6), atol=2.5e-6)

    @parameterized.expand(itertools.product(['fwd', 'rev'], ['matfree', '']), name_func=parameterized_name)
    def test_jax_lin_system_primal(self, mode, matrix_free):
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = np.array([1, 2, -3])

        prob = om.Problem()

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('A', A)
        ivc.add_output('b', b)

        lingrp = prob.model.add_subsystem('lingrp', om.Group())
        comp = lingrp.add_subsystem('lin', JaxLinearSystemCompPrimal(size=3))
        comp.matrix_free = bool(matrix_free)

        prob.model.connect('ivc.A', 'lingrp.lin.A')
        prob.model.connect('ivc.b', 'lingrp.lin.b')

        prob.setup(mode=mode)
        prob.set_solver_print(level=0)

        prob.run_model()

        assert_near_equal(prob['lingrp.lin.x'], np.array([-4, 9, -4]), tolerance=1e-14)

        assert_check_totals(prob.check_totals(of=['lingrp.lin.x'], wrt=['ivc.b', 'ivc.A'],
                                              abs_err_tol=2e-4, rel_err_tol=3e-6, show_only_incorrect=True),
                            atol=2e-4, rtol=3e-6)
        assert_check_partials(prob.check_partials(show_only_incorrect=True), rtol=1e-5)

    @parameterized.expand(itertools.product(['fwd', 'rev'], ['matfree', '']), name_func=parameterized_name)
    def test_jax_lin_system_primal_w_option(self, mode, matrix_free):
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = np.array([1, 2, -3])

        prob = om.Problem()

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('A', A)
        ivc.add_output('b', b)

        lingrp = prob.model.add_subsystem('lingrp', om.Group())
        lin = lingrp.add_subsystem('lin', JaxLinearSystemCompPrimalwOption(size=3))
        lin.matrix_free = bool(matrix_free)

        prob.model.connect('ivc.A', 'lingrp.lin.A')
        prob.model.connect('ivc.b', 'lingrp.lin.b')

        prob.setup(mode=mode)
        prob.set_solver_print(level=0)

        prob.set_val('ivc.A', A)
        prob.set_val('ivc.b', b)

        prob.run_model()

        assert_near_equal(prob['lingrp.lin.x'], np.array([-5, 8, -5]), tolerance=1e-14)

        assert_check_totals(prob.check_totals(of=['lingrp.lin.x'], wrt=['ivc.b', 'ivc.A'],
                                              abs_err_tol=2e-4, rel_err_tol=3e-6, show_only_incorrect=True),
                            atol=2e-4, rtol=3e-6)
        assert_check_partials(prob.check_partials(show_only_incorrect=True), rtol=1e-5)

        lin.options['adder'] = 0.0

        prob.run_model()

        assert_near_equal(prob['lingrp.lin.x'], np.array([-4, 9, -4]), tolerance=1e-14)

        assert_check_totals(prob.check_totals(of=['lingrp.lin.x'], wrt=['ivc.b', 'ivc.A'],
                                              abs_err_tol=2e-4, rel_err_tol=3e-6, show_only_incorrect=True),
                            atol=2e-4, rtol=3e-6)
        assert_check_partials(prob.check_partials(show_only_incorrect=True), rtol=1e-5)

    @parameterized.expand(itertools.product(['fwd', 'rev']), name_func=parameterized_name)
    def test_jax_bad_shape_lin_system_primal_w_option(self, mode):
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = np.array([1, 2, -3])

        prob = om.Problem()

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('A', A)
        ivc.add_output('b', b)

        lingrp = prob.model.add_subsystem('lingrp', om.Group())
        lin = lingrp.add_subsystem('lin', BadShapeJaxLinearSystemCompPrimalwOption(size=3))

        prob.model.connect('ivc.A', 'lingrp.lin.A')
        prob.model.connect('ivc.b', 'lingrp.lin.b')

        prob.setup(mode=mode)
        prob.set_val('ivc.A', A)
        prob.set_val('ivc.b', b)

        with self.assertRaises(Exception) as cm:
            prob.run_model()

        msg = ("'lingrp.lin' <class BadShapeJaxLinearSystemCompPrimalwOption>:"
               "\n   Shape mismatch for output 'x': expected (3,) but got ().")
        self.assertEqual(cm.exception.args[0], msg)

    @parameterized.expand(itertools.product(['fwd', 'rev'], ['matfree', '']), name_func=parameterized_name)
    def test_jax_lin_system_primal_w_discrete(self, mode, matrix_free):
        A = np.array([[1., 1., 1.], [1., 2., 3.], [0., 1., 3.]])
        b = np.array([1, 2, -3])

        prob = om.Problem()

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('A', A)
        ivc.add_output('b', b)

        lingrp = prob.model.add_subsystem('lingrp', om.Group())
        lin = lingrp.add_subsystem('lin', JaxLinearSystemCompPrimalwDiscrete(size=3))
        lin.matrix_free = bool(matrix_free)

        prob.model.connect('ivc.A', 'lingrp.lin.A')
        prob.model.connect('ivc.b', 'lingrp.lin.b')

        prob.setup(mode=mode)
        prob.set_solver_print(level=0)

        prob.set_val('ivc.A', A)
        prob.set_val('ivc.b', b)

        prob.run_model()

        assert_near_equal(prob['lingrp.lin.x'], np.array([-5, 8, -5]), tolerance=1e-14)

        assert_check_totals(prob.check_totals(of=['lingrp.lin.x'], wrt=['ivc.b', 'ivc.A'],
                                              abs_err_tol=2e-4, rel_err_tol=3e-6, show_only_incorrect=True),
                            atol=2e-4, rtol=3e-6)
        assert_check_partials(prob.check_partials(show_only_incorrect=True), rtol=1e-5)

        prob['lingrp.lin.c_discrete'] = '0'

        prob.run_model()

        assert_near_equal(prob['lingrp.lin.x'], np.array([-4, 9, -4]), tolerance=1e-14)

        assert_check_totals(prob.check_totals(of=['lingrp.lin.x'], wrt=['ivc.b', 'ivc.A'],
                                              abs_err_tol=2e-4, rel_err_tol=3e-6, show_only_incorrect=True),
                            atol=2e-4, rtol=3e-6)
        assert_check_partials(prob.check_partials(show_only_incorrect=True), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
