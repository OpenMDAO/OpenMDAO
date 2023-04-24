import unittest

import numpy as np

import openmdao.math.jax as omath
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

import openmdao.api as om
import openmdao.func_api as omf


class TestActTanh(unittest.TestCase):

    def test_tanh_act(self):
        f = omath.act_tanh(6, mu=1.0E-5, z=6, a=-10, b=10)
        assert_near_equal(np.asarray(f), 0.0)

        f = omath.act_tanh(6, mu=1.0E-5, z=6, a=-10, b=0)
        assert_near_equal(np.asarray(f), -5.0)

        f = omath.act_tanh(-10, mu=1.0E-5, z=6, a=-10, b=0)
        assert_near_equal(np.asarray(f), -10)

        f = omath.act_tanh(10, mu=1.0E-5, z=6, a=-10, b=20)
        assert_near_equal(np.asarray(f), 20)

    def test_tanh_act_dense_derivs(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(12,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            with self.subTest(f'sum of shape {X_SHAPE}'):
                p = om.Problem()

                def f_wrap(x, mu, z, a, b):
                    return omath.act_tanh(x, mu, z, a, b)

                def df_wrap(x, mu, z, a, b, J):
                    df_dx, df_dmu, df_dz, df_da, df_db = omath.d_act_tanh(x, mu, z, a, b)
                    J['f', 'x'] = np.diagflat(df_dx)
                    J['f', 'mu'] = df_dmu
                    J['f', 'z'] = df_dz
                    J['f', 'a'] = df_da
                    J['f', 'b'] = df_db

                f = (omf.wrap(f_wrap).add_input('x', shape=X_SHAPE, val=1.0)
                     .add_output('f', shape=X_SHAPE, val=1.0)
                     .declare_partials(of='f', wrt=('x', 'mu', 'z', 'a', 'b')))

                p.model.add_subsystem('f_comp', om.ExplicitFuncComp(f, compute_partials=df_wrap))

                p.setup(force_alloc_complex=True)
                p.set_val('f_comp.x', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.mu', np.random.random(1,))
                p.set_val('f_comp.z', np.random.random(1,))
                p.set_val('f_comp.a', np.random.random(1,))
                p.set_val('f_comp.b', np.random.random(1,))

                p.run_model()

                with np.printoptions(linewidth=1024):
                    cpd = p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)

    def test_tanh_act_sparse_derivs(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(12,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            with self.subTest(f'sum of shape {X_SHAPE}'):
                p = om.Problem()

                ar = np.arange(np.prod(X_SHAPE), dtype=int)

                def f_wrap(x, mu, z, a, b):
                    return omath.act_tanh(x, mu, z, a, b)

                def df_wrap(x, mu, z, a, b, J):
                    df_dx, df_dmu, df_dz, df_da, df_db = omath.d_act_tanh(x, mu, z, a, b)
                    J['f', 'x'] = df_dx
                    J['f', 'mu'] = df_dmu
                    J['f', 'z'] = df_dz
                    J['f', 'a'] = df_da
                    J['f', 'b'] = df_db

                f = (omf.wrap(f_wrap).add_input('x', shape=X_SHAPE, val=1.0)
                     .add_input('z', shape=X_SHAPE, val=1.0)
                     .add_output('f', shape=X_SHAPE, val=1.0)
                     .declare_partials(of='f', wrt=('x', 'z'), rows=ar, cols=ar)
                     .declare_partials(of='f', wrt=('mu', 'a', 'b')))

                p.model.add_subsystem('f_comp', om.ExplicitFuncComp(f, compute_partials=df_wrap))

                p.setup(force_alloc_complex=True)
                p.set_val('f_comp.x', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.mu', 1.0E-5)
                p.set_val('f_comp.z', np.random.random(X_SHAPE))
                p.set_val('f_comp.a', np.random.random(1,))
                p.set_val('f_comp.b', np.random.random(1,))

                p.run_model()

                with np.printoptions(linewidth=1024):
                    cpd = p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)


class TestSmoothMax(unittest.TestCase):

    def test_smooth_max(self):
        x = np.linspace(0, 1, 1000)
        sin = np.sin(x)
        cos = np.cos(x)

        smax = omath.smooth_max(sin, cos, mu=1.0E-6)

        idxs_sgt = np.where(sin > cos)
        idxs_cgt = np.where(sin < cos)

        assert_near_equal(smax[idxs_sgt], sin[idxs_sgt])
        assert_near_equal(smax[idxs_cgt], cos[idxs_cgt])

    def test_smooth_max_dense_derivs(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(5,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            with self.subTest(f'sum of shape {X_SHAPE}'):
                p = om.Problem()

                def f_wrap(x, y, mu):
                    return omath.smooth_max(x, y, mu)

                def df_wrap(x, y, mu, J):
                    df_dx, df_dy, df_dmu = omath.d_smooth_max(x, y, mu, dx=True, dy=True, dmu=True)
                    J['f', 'x'] = np.diagflat(df_dx)
                    J['f', 'y'] = np.diagflat(df_dy)
                    J['f', 'mu'] = df_dmu

                f = (omf.wrap(f_wrap).add_input('x', shape=X_SHAPE, val=1.0)
                     .add_input('y', shape=X_SHAPE, val=1.0)
                     .add_input('mu', val=1.0)
                     .add_output('f', shape=X_SHAPE, val=1.0)
                     .declare_partials(of='f', wrt=('x', 'y', 'mu')))

                p.model.add_subsystem('f_comp', om.ExplicitFuncComp(f, compute_partials=df_wrap))

                p.setup(force_alloc_complex=True)
                p.set_val('f_comp.x', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.y', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.mu', np.random.random(1,))

                p.run_model()

                with np.printoptions(linewidth=1024):
                    cpd = p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)

    def test_smooth_max_sparse_derivs(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(5,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            with self.subTest(f'sum of shape {X_SHAPE}'):
                p = om.Problem()

                def df_wrap(x, y, mu, J):
                    df_dx, df_dy, df_dmu = omath.d_smooth_max(x, y, mu, dx=True, dy=True, dmu=True)
                    J['f', 'x'] = df_dx
                    J['f', 'y'] = df_dy
                    J['f', 'mu'] = df_dmu

                ar = np.arange(np.prod(X_SHAPE), dtype=int)

                f = (omf.wrap(omath.smooth_max).add_input('x', shape=X_SHAPE)
                     .add_input('y', shape=X_SHAPE)
                     .add_input('mu')
                     .add_output('f', shape=X_SHAPE, val=1.0)
                     .declare_partials(of='f', wrt=('x', 'y'), rows=ar, cols=ar)
                     .declare_partials(of='f', wrt='mu'))

                p.model.add_subsystem('f_comp', om.ExplicitFuncComp(f, compute_partials=df_wrap))

                p.setup(force_alloc_complex=True)
                p.set_val('f_comp.x', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.y', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.mu', np.random.random(1,))

                p.run_model()

                with np.printoptions(linewidth=1024):
                    cpd = p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)


class TestSmoothMin(unittest.TestCase):

    def test_smooth_min(self):
        x = np.linspace(0, 1, 1000)
        sin = np.sin(x)
        cos = np.cos(x)

        smin = omath.smooth_min(sin, cos, mu=1.0E-6)

        idxs_sgt = np.where(sin > cos)
        idxs_cgt = np.where(sin < cos)

        assert_near_equal(smin[idxs_sgt], cos[idxs_sgt])
        assert_near_equal(smin[idxs_cgt], sin[idxs_cgt])

    def test_smooth_min_dense_derivs(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(5,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            with self.subTest(f'sum of shape {X_SHAPE}'):
                p = om.Problem()

                def f_wrap(x, y, mu):
                    return omath.smooth_min(x, y, mu)

                def df_wrap(x, y, mu, J):
                    df_dx, df_dy, df_dmu = omath.d_smooth_min(x, y, mu, dx=True, dy=True, dmu=True)
                    J['f', 'x'] = np.diagflat(df_dx)
                    J['f', 'y'] = np.diagflat(df_dy)
                    J['f', 'mu'] = df_dmu

                f = (omf.wrap(f_wrap).add_input('x', shape=X_SHAPE, val=1.0)
                     .add_input('y', shape=X_SHAPE, val=1.0)
                     .add_input('mu', val=1.0)
                     .add_output('f', shape=X_SHAPE, val=1.0)
                     .declare_partials(of='f', wrt=('x', 'y', 'mu')))

                p.model.add_subsystem('f_comp', om.ExplicitFuncComp(f, compute_partials=df_wrap))

                p.setup(force_alloc_complex=True)
                p.set_val('f_comp.x', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.y', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.mu', np.random.random(1,))

                p.run_model()

                with np.printoptions(linewidth=1024):
                    cpd = p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)

    def test_smooth_min_sparse_derivs(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(5,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            with self.subTest(f'sum of shape {X_SHAPE}'):
                p = om.Problem()

                def df_wrap(x, y, mu, J):
                    df_dx, df_dy, df_dmu = omath.d_smooth_min(x, y, mu, dx=True, dy=True, dmu=True)
                    J['f', 'x'] = df_dx
                    J['f', 'y'] = df_dy
                    J['f', 'mu'] = df_dmu

                ar = np.arange(np.prod(X_SHAPE), dtype=int)

                f = (omf.wrap(omath.smooth_min).add_input('x', shape=X_SHAPE)
                     .add_input('y', shape=X_SHAPE)
                     .add_input('mu')
                     .add_output('f', shape=X_SHAPE, val=1.0)
                     .declare_partials(of='f', wrt=('x', 'y'), rows=ar, cols=ar)
                     .declare_partials(of='f', wrt='mu'))

                p.model.add_subsystem('f_comp', om.ExplicitFuncComp(f, compute_partials=df_wrap))

                p.setup(force_alloc_complex=True)
                p.set_val('f_comp.x', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.y', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.mu', np.random.random(1,))

                p.run_model()

                with np.printoptions(linewidth=1024):
                    cpd = p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)


class TestSmoothAbs(unittest.TestCase):

    def test_smooth_abs(self):
        x = np.linspace(-0.5, 0.5, 1000)

        sabs = omath.smooth_abs(x)
        abs = np.abs(x)

        idxs_compare = np.where(abs > 0.1)
        assert_near_equal(sabs[idxs_compare], abs[idxs_compare], tolerance=1.0E-9)

    def test_smooth_abs_dense_derivs(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(5,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            with self.subTest(f'sum of shape {X_SHAPE}'):
                p = om.Problem()

                def df_wrap(x, mu, J):
                    df_dx, df_dmu = omath.d_smooth_abs(x, mu, dx=True, dmu=True)
                    J['f', 'x'] = np.diagflat(df_dx)
                    J['f', 'mu'] = df_dmu

                f = (omf.wrap(omath.smooth_abs)
                     .add_input('x', shape=X_SHAPE)
                     .add_input('mu')
                     .add_output('f', shape=X_SHAPE)
                     .declare_partials(of='f', wrt=('x', 'mu')))

                p.model.add_subsystem('f_comp', om.ExplicitFuncComp(f, compute_partials=df_wrap))

                p.setup(force_alloc_complex=True)
                p.set_val('f_comp.x', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.mu', np.random.random(1,))

                p.run_model()

                with np.printoptions(linewidth=1024):
                    cpd = p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)

    def test_smooth_abs_sparse_derivs(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(5,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            with self.subTest(f'sum of shape {X_SHAPE}'):
                p = om.Problem()

                def df_wrap(x, mu, J):
                    df_dx, df_dmu = omath.d_smooth_abs(x, mu, dx=True, dmu=True)
                    J['f', 'x'] = df_dx
                    J['f', 'mu'] = df_dmu

                ar = np.arange(np.prod(X_SHAPE), dtype=int)

                f = (omf.wrap(omath.smooth_abs)
                     .add_input('x', shape=X_SHAPE)
                     .add_input('mu')
                     .add_output('f', shape=X_SHAPE, val=1.0)
                     .declare_partials(of='f', wrt='x', rows=ar, cols=ar)
                     .declare_partials(of='f', wrt='mu'))

                p.model.add_subsystem('f_comp', om.ExplicitFuncComp(f, compute_partials=df_wrap))

                p.setup(force_alloc_complex=True)
                p.set_val('f_comp.x', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.mu', np.random.random(1,))

                p.run_model()

                with np.printoptions(linewidth=1024):
                    cpd = p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)
                

if __name__ == '__main__':
    unittest.main()
