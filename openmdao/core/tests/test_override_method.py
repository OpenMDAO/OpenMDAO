from types import MethodType
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials


def compute_partials_func(self, inputs, partials):
    partials['area', 'length'] = inputs['width']
    partials['area', 'width'] = inputs['length']


def compute_jacvec_product_func(self, inputs, d_inputs, d_outputs, mode):
    if mode == 'fwd':
        if 'area' in d_outputs:
            d_outputs['area'] = 0.0
            if 'length' in d_inputs:
                d_outputs['area'] += inputs['width'] * d_inputs['length']
            if 'width' in d_inputs:
                d_outputs['area'] += inputs['length'] * d_inputs['width']
    elif mode == 'rev':
        if 'area' in d_outputs:
            if 'length' in d_inputs:
                d_inputs['length'] = 0.0
                d_inputs['length'] += inputs['width'] * d_outputs['area']
            if 'width' in d_inputs:
                d_inputs['width'] = 0.0
                d_inputs['width'] += inputs['length'] * d_outputs['area']


class TestExplicitComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('add_compute_jacvec_product', types=bool, desc="add compute_jacvec_product method if `True`, add `compute_partials` method otherwise")
        self.options.declare('pass_MethodType', types=bool, desc="pass a `MethodType` to `self.override_method` instead of a plain function")
        self.options.declare('attempt_buggy_method_name', types=bool, default=False, desc="pass an `name` arg to `override_method` for testing purposes")

    def setup(self):
        self.add_input('length', val=1.0)
        self.add_input('width', val=1.0)
        self.add_output('area', val=1.0)

        self.declare_partials("area", ["length", "width"])

        if self.options['add_compute_jacvec_product']:
            if self.options['pass_MethodType']:
                self.override_method("compute_jacvec_product", MethodType(compute_jacvec_product_func, self))
            else:
                self.override_method("compute_jacvec_product", compute_jacvec_product_func)
        else:
            if self.options['pass_MethodType']:
                self.override_method("compute_partials", MethodType(compute_partials_func, self))
            else:
                self.override_method("compute_partials", compute_partials_func)

        if self.options['attempt_buggy_method_name']:
            if self.options['pass_MethodType']:
                self.override_method("compute_foo", MethodType(compute_partials_func, self))
            else:
                self.override_method("compute_foo", compute_partials_func)

    def compute(self, inputs, outputs):
        outputs['area'] = inputs['length'] * inputs['width']


class TestAddMethodExplicitComp(unittest.TestCase):

    def test_override_method(self):
        for add_compute_jacvec_product in [True, False]:
            for pass_MethodType in [True, False]:
                prob = om.Problem()
                comp = TestExplicitComp(add_compute_jacvec_product=add_compute_jacvec_product, pass_MethodType=pass_MethodType)

                prob.model.add_subsystem('comp', comp, promotes=["*"])
                prob.setup()
                prob.set_val('length', 3.0)
                prob.set_val('width', 4.0)
                prob.run_model()

                assert_near_equal(prob.get_val('area'), 12.0)
                assert_check_partials(prob.check_partials(out_stream=None))

    def test_invalid_arguments(self):
        for add_compute_jacvec_product in [True, False]:
            for pass_MethodType in [True, False]:
                prob = om.Problem()
                comp = TestExplicitComp(add_compute_jacvec_product=add_compute_jacvec_product, pass_MethodType=pass_MethodType, attempt_buggy_method_name=True)
                prob.model.add_subsystem('comp', comp, promotes=["*"])
                with self.assertRaises(ValueError):
                    prob.setup()


def solve_nonlinear_func(self, inputs, outputs):
    a = inputs['a']
    b = inputs['b']
    c = inputs['c']
    outputs['x'] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)


def linearize_func(self, inputs, outputs, partials):
    a = inputs['a']
    b = inputs['b']
    # c = inputs['c']  # not needed
    x = outputs['x']

    partials['x', 'a'] = x ** 2
    partials['x', 'b'] = x
    partials['x', 'c'] = 1.0
    partials['x', 'x'] = 2 * a * x + b

    self.inv_jac = 1.0 / (2 * a * x + b)


def apply_linear_func(self, inputs, outputs,
                 d_inputs, d_outputs, d_residuals, mode):
    a = inputs['a']
    b = inputs['b']
    # c = inputs['c']  # not needed
    x = outputs['x']
    if mode == 'fwd':
        if 'x' in d_residuals:
            if 'x' in d_outputs:
                d_residuals['x'] += (2 * a * x + b) * d_outputs['x']
            if 'a' in d_inputs:
                d_residuals['x'] += x ** 2 * d_inputs['a']
            if 'b' in d_inputs:
                d_residuals['x'] += x * d_inputs['b']
            if 'c' in d_inputs:
                d_residuals['x'] += d_inputs['c']
    elif mode == 'rev':
       if 'x' in d_residuals:
            if 'x' in d_outputs:
                d_outputs['x'] += (2 * a * x + b) * d_residuals['x']
            if 'a' in d_inputs:
                d_inputs['a'] += x ** 2 * d_residuals['x']
            if 'b' in d_inputs:
                d_inputs['b'] += x * d_residuals['x']
            if 'c' in d_inputs:
                d_inputs['c'] += d_residuals['x']


def solve_linear_func(self, d_outputs, d_residuals, mode):
    if mode == 'fwd':
        d_outputs['x'] = self.inv_jac * d_residuals['x']
    elif mode == 'rev':
        d_residuals['x'] = self.inv_jac * d_outputs['x']


def guess_nonlinear_func(self, inputs, outputs, resids):
    a = inputs['a']
    b = inputs['b']
    c = inputs['c']
    x_expected = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    outputs['x'] = x_expected + 0.5


class QuadraticComp(om.ImplicitComponent):
    """
    A Simple Implicit Component representing a Quadratic Equation.

    R(a, b, c, x) = ax^2 + bx + c

    Solution via Quadratic Formula:
    x = (-b + sqrt(b^2 - 4ac)) / 2a
    """

    def initialize(self):
        self.options.declare('add_solve_nonlinear', types=bool, default=False)
        self.options.declare('add_linearize', types=bool, default=False)
        self.options.declare('add_apply_linear', types=bool, default=False)
        self.options.declare('add_solve_linear', types=bool, default=False)
        self.options.declare('add_guess_nonlinear', types=bool, default=False)
        self.options.declare('pass_MethodType', types=bool, desc="pass a `MethodType` to `self.override_method` instead of a plain function")
        self.options.declare('attempt_buggy_method_name', types=bool, default=False, desc="pass an `name` arg to `override_method` for testing purposes")

    def setup(self):
        self.add_input('a', val=1., tags=['tag_a'])
        self.add_input('b', val=1.)
        self.add_input('c', val=1.)
        self.add_output('x', val=0., tags=['tag_x'])

        self.declare_partials(of='*', wrt='*')

        pass_MethodType = self.options["pass_MethodType"]
        if self.options["add_linearize"]:
            if pass_MethodType:
                self.override_method("linearize", MethodType(linearize_func, self))
            else:
                self.override_method("linearize", linearize_func)

        if self.options["add_apply_linear"]:
            if pass_MethodType:
                self.override_method("apply_linear", MethodType(apply_linear_func, self))
            else:
                self.override_method("apply_linear", apply_linear_func)

        if self.options["add_solve_nonlinear"]:
            if pass_MethodType:
                self.override_method("solve_nonlinear", MethodType(solve_nonlinear_func, self))
            else:
                self.override_method("solve_nonlinear", solve_nonlinear_func)
        else:
            self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
            self.nonlinear_solver.options['iprint'] = 0
            self.nonlinear_solver.options['maxiter'] = 20
            self.nonlinear_solver.linear_solver = om.DirectSolver(assemble_jac=False)

        if self.options["add_solve_linear"]:
            if pass_MethodType:
                self.override_method("solve_linear", MethodType(solve_linear_func, self))
            else:
                self.override_method("solve_linear", solve_linear_func)
        else:
            self.linear_solver = om.DirectSolver(assemble_jac=False)

        if self.options["add_guess_nonlinear"]:
            if pass_MethodType:
                self.override_method("guess_nonlinear", MethodType(guess_nonlinear_func, self))
            else:
                self.override_method("guess_nonlinear", guess_nonlinear_func)

        if self.options['attempt_buggy_method_name']:
            if self.options['pass_MethodType']:
                self.override_method("compute_foo", MethodType(linearize_func, self))
            else:
                self.override_method("compute_foo", linearize_func)

    def apply_nonlinear(self, inputs, outputs, residuals):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        residuals['x'] = a * x ** 2 + b * x + c


class TestAddMethodImplicitComp(unittest.TestCase):

    def test_override_method(self):
        for add_linearize in [True, False]:
            add_apply_linear = not add_linearize
            for add_solve_nonlinear in [True, False]:
                for add_solve_linear in [True, False]:
                    for add_guess_nonlinear in [True, False]:
                        for pass_MethodType in [True, False]:
                            prob = om.Problem()
                            comp = QuadraticComp(
                                add_linearize=add_linearize,
                                add_apply_linear=add_apply_linear,
                                add_solve_nonlinear=add_solve_nonlinear,
                                add_solve_linear=add_solve_linear,
                                add_guess_nonlinear=add_guess_nonlinear,
                                pass_MethodType=pass_MethodType)

                            prob.model.add_subsystem('comp', comp, promotes=["*"])

                            prob.setup()
                            a = 1.0
                            b = 4.0
                            c = 2.0
                            prob.set_val('a', a)
                            prob.set_val('b', b)
                            prob.set_val('c', c)
                            prob.set_val('x', 1.0)
                            prob.run_model()

                            x_expected = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

                            assert_near_equal(prob.get_val('x'), x_expected, tolerance=1e-12, tol_type="rel")
                            # assert_check_partials(prob.check_partials(compact_print=True))#out_stream=None))
                            assert_check_partials(prob.check_partials(out_stream=None))

    def test_invalid_arguments(self):
        for add_linearize in [True, False]:
            add_apply_linear = not add_linearize
            for add_solve_nonlinear in [True, False]:
                for add_solve_linear in [True, False]:
                    for add_guess_nonlinear in [True, False]:
                        for pass_MethodType in [True, False]:
                            prob = om.Problem()
                            comp = QuadraticComp(
                                add_linearize=add_linearize,
                                add_apply_linear=add_apply_linear,
                                add_solve_nonlinear=add_solve_nonlinear,
                                add_solve_linear=add_solve_linear,
                                add_guess_nonlinear=add_guess_nonlinear,
                                pass_MethodType=pass_MethodType,
                                attempt_buggy_method_name=True)

                            prob.model.add_subsystem('comp', comp, promotes=["*"])

                            with self.assertRaises(ValueError):
                                prob.setup()


if __name__ == '__main__':
    unittest.main()
