import unittest

import numpy as np

import openmdao.func as omfunc
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

import openmdao.api as om
import openmdao.func_api as omf


scalar_funcs_unit_range = ['cos', 'cosh', 'sin', 'sinh', 'tan', 'tanh', 'exp', 'log', 'log10', 'sqrt', 'tan',
                           'arccos', 'arcsin', 'arctan']

scalar_inverse_hyperbolic_funcs = ['arccosh', 'arcsinh']


class TestFuncsFromNumpy(unittest.TestCase):

    def _test_func_derivs_cs(self, func_name, x, tol=1.0E-12):
        deriv_f = getattr(omfunc, f'd_{func_name}')(x)
        cs_deriv_f = getattr(omfunc, func_name)(x + (0 + 1.0E-40j)).imag / 1.0E-40
        assert_near_equal(deriv_f, cs_deriv_f, tolerance=tol)


    def test_scalar_derivs(self):

        x = np.random.random(1)
        y = 1 + np.random.random(1)

        for func_name in scalar_funcs_unit_range:
            print(f'testing {func_name}...', end='')
            self._test_func_derivs_cs(func_name, x)
            print('passed.')

        for func_name in scalar_inverse_hyperbolic_funcs:
            print(f'testing {func_name}...', end='')
            self._test_func_derivs_cs(func_name, y)
            print('passed.')

    def test_vector_derivs(self):

        x = np.random.random(100)
        y = 1 + np.random.random(100)

        for func_name in scalar_funcs_unit_range:
            print(f'testing {func_name}...', end='')
            self._test_func_derivs_cs(func_name, x)
            print('passed.')

        for func_name in scalar_inverse_hyperbolic_funcs:
            print(f'testing {func_name}...', end='')
            self._test_func_derivs_cs(func_name, y)
            print('passed.')

    def test_tensor_derivs(self):

        x = np.random.random((10, 10, 10))
        y = 1 + np.random.random((10, 10, 10))

        for func_name in scalar_funcs_unit_range:
            print(f'testing {func_name}...', end='')
            self._test_func_derivs_cs(func_name, x)
            print('passed.')

        for func_name in scalar_inverse_hyperbolic_funcs:
            print(f'testing {func_name}...', end='')
            self._test_func_derivs_cs(func_name, y)
            print('passed.')


class TestSum(unittest.TestCase):

    def test_sum_tensor(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(12,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            for AXIS in list(range(len(X_SHAPE))) + [None]:
                with self.subTest(f'sum of shape {X_SHAPE} along axis {AXIS}'):
                    if AXIS is None or len(X_SHAPE) == 1:
                        SUM_SHAPE = (1,)
                    else:
                        SUM_SHAPE = list(X_SHAPE)
                        SUM_SHAPE[AXIS] = 1

                    p = om.Problem()

                    def sum_wrap(x):
                        return omfunc.sum(x, axis=AXIS)

                    def dsum_wrap(x, J):
                        J['sum', 'x'] = omfunc.d_sum(x, axis=AXIS)

                    f = (omf.wrap(sum_wrap).add_input('x', shape=X_SHAPE, val=1.0)
                         .add_output('sum', shape=SUM_SHAPE, val=1.0)
                         .declare_partials(of='sum', wrt=('x',)))

                    p.model.add_subsystem('sum_comp', om.ExplicitFuncComp(f, compute_partials=dsum_wrap))

                    p.setup(force_alloc_complex=True)
                    p.set_val('sum_comp.x', np.random.random(X_SHAPE))

                    p.run_model()

                    assert_near_equal(p.get_val('sum_comp.sum'),
                                      np.asarray(np.sum(p.get_val('sum_comp.x'), axis=AXIS, keepdims=AXIS is not None)))

                    with np.printoptions(linewidth=1024):
                        cpd = p.check_partials(method='cs', out_stream=None)
                    assert_check_partials(cpd)
