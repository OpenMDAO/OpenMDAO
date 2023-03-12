import unittest

import numpy as np

import openmdao.api as om
import openmdao.func.cs_safe as cs_safe
import openmdao.func_api as omf
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

funcs_single_arg = ['abs', 'arctanh']
funcs_vector_arg = ['norm']
funcs_2_arg = ['arctan2']


class TestNorm(unittest.TestCase):

    def test_array(self):
        N = 10

        a = np.linspace(1, 5, N, dtype=complex)
        assert_near_equal(cs_safe.norm(a), np.linalg.norm(a))

        deriv_f = cs_safe.d_norm(a.real)
        cs_deriv_f = np.zeros((1, N))

        for i in range(N):
            a[i] += (0 + 1.0E-40j)
            cs_deriv_f[0, i] = cs_safe.norm(a).imag / 1.0E-40
            a[i] = a[i].real


        with np.printoptions(linewidth=1024):
            print(cs_deriv_f)
            print()
            print(deriv_f)

        assert_near_equal(deriv_f, cs_deriv_f, tolerance=1.0E-12)

    def test_tensor(self):

        for X_SHAPE in [(12,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            for AXIS in list(range(len(X_SHAPE))) + [None]:
                with self.subTest(f'sum of shape {X_SHAPE} along axis {AXIS}'):
                    if AXIS is None or len(X_SHAPE) == 1:
                        SUM_SHAPE = (1,)
                    else:
                        temp = list(X_SHAPE)
                        temp.pop(AXIS)
                        SUM_SHAPE = tuple(temp)

                    def norm_wrap(x):
                        return cs_safe.norm(x, axis=AXIS)

                    def dnorm_wrap(x, J):
                        J['norm', 'x'] = cs_safe.d_norm(x, axis=AXIS)

                    f = (omf.wrap(norm_wrap).add_input('x', shape=X_SHAPE, val=1.0)
                         .add_output('norm', shape=SUM_SHAPE, val=1.0)
                         .declare_partials(of='norm', wrt=('x',)))

                    p = om.Problem()
                    p.model.add_subsystem('norm_comp', om.ExplicitFuncComp(f, compute_partials=dnorm_wrap))

                    p.setup(force_alloc_complex=True)
                    p.set_val('norm_comp.x', np.random.random(X_SHAPE))

                    p.run_model()

                    assert_near_equal(p.get_val('norm_comp.norm'), np.asarray(np.linalg.norm(p.get_val('norm_comp.x'), axis=AXIS)))

                    with np.printoptions(linewidth=1024):
                        cpd = p.check_partials(method='cs', out_stream=None)
                    assert_check_partials(cpd)

    def test_vec(self):
        N = 10

        a = np.zeros((10, 1), dtype=complex)
        a[:, 0] = np.linspace(1, 5, N, dtype=complex)
        assert_near_equal(cs_safe.norm(a), np.linalg.norm(a))

        deriv_f = cs_safe.d_norm(a.real)
        cs_deriv_f = np.zeros((1, N))

        for i in range(N):
            a.ravel()[i] += (0 + 1.0E-40j)
            cs_deriv_f[0, i] = cs_safe.norm(a).imag / 1.0E-40
            a.ravel()[i] = a.ravel()[i].real

        with np.printoptions(linewidth=1024):
            print(cs_deriv_f)
            print()
            print(deriv_f)

        assert_near_equal(deriv_f, cs_deriv_f, tolerance=1.0E-12)

    def test_vec_axis_0(self):
        N = 10

        a = np.zeros((10, 1), dtype=complex)
        a[:, 0] = np.linspace(1, 5, N, dtype=complex)
        assert_near_equal(cs_safe.norm(a, axis=0), np.linalg.norm(a, axis=0))

        deriv_f = cs_safe.d_norm(a.real, axis=0)
        cs_deriv_f = np.zeros((N))

        for i in range(N):
            a.ravel()[i] += (0 + 1.0E-40j)
            cs_deriv_f[i] = cs_safe.norm(a, axis=0).imag / 1.0E-40
            a.ravel()[i] = a.ravel()[i].real

        with np.printoptions(linewidth=1024):
            print(cs_deriv_f)
            print(deriv_f)

        assert_near_equal(deriv_f, cs_deriv_f, tolerance=1.0E-12)

    def test_vec_axis_1(self):
        N = 10

        a = np.zeros((10, 1), dtype=complex)
        a[:, 0] = np.linspace(1, 5, N, dtype=complex)

        assert_near_equal(cs_safe.norm(a, axis=1), np.linalg.norm(a, axis=1))

        deriv_f = cs_safe.d_norm(a.real, axis=1)
        cs_deriv_f = np.zeros((N, N))

        for row in range(N):
            a[row, 0] += (0 + 1.0E-40j)
            cs_deriv_f[row, :] = cs_safe.norm(a, axis=1).imag / 1.0E-40
            a[row, 0] -= (0 + 1.0E-40j)

        with np.printoptions(linewidth=1024):
            print(cs_deriv_f)
            print(deriv_f)

        assert_near_equal(deriv_f, cs_deriv_f, tolerance=1.0E-12)


class TestOthers(unittest.TestCase):

    def _test_func_derivs_cs(self, func_name, x, y=None, tol=1.0E-12):
        f = getattr(cs_safe, f'd_{func_name}')
        cs_f = getattr(cs_safe, func_name)
        if y is None:
            deriv_f = f(x)
            cs_deriv_f = cs_f(x + (0 + 1.0E-40j)).imag / 1.0E-40
            assert_near_equal(deriv_f, cs_deriv_f, tolerance=tol)
        else:
            deriv_f = f(x, y)
            cs_deriv_f_x = cs_f(x + (0 + 1.0E-40j), y).imag / 1.0E-40
            cs_deriv_f_y = cs_f(x, y + (0 + 1.0E-40j)).imag / 1.0E-40
            assert_near_equal(deriv_f[0], cs_deriv_f_x, tolerance=tol)
            assert_near_equal(deriv_f[1], cs_deriv_f_y, tolerance=tol)

    def test_scalar_derivs(self):

        x = np.random.random(1)
        y = 1 + np.random.random(1)

        for func_name in funcs_single_arg:
            with self.subTest(msg=func_name):
                self._test_func_derivs_cs(func_name, x)

        for func_name in funcs_2_arg:
            with self.subTest(msg=func_name):
                self._test_func_derivs_cs(func_name, x, y)

    def test_vector_derivs(self):

        x = np.random.random(100)
        y = 1 + np.random.random(100)

        for func_name in funcs_single_arg:
            with self.subTest(msg=func_name):
                self._test_func_derivs_cs(func_name, x)

        for func_name in funcs_2_arg:
            with self.subTest(msg=func_name):
                self._test_func_derivs_cs(func_name, x, y)

    def test_tensor_derivs(self):

        x = np.random.random((10, 10, 10))
        y = 1 + np.random.random((10, 10, 10))

        for func_name in funcs_single_arg:
            with self.subTest(msg=func_name):
                self._test_func_derivs_cs(func_name, x)

        for func_name in funcs_2_arg:
            with self.subTest(msg=func_name):
                self._test_func_derivs_cs(func_name, x, y)



