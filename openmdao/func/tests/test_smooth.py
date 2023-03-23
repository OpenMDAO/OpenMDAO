import unittest

import numpy as np

import openmdao.func as omfunc
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

import openmdao.api as om
import openmdao.func_api as omf


class TestActTanh(unittest.TestCase):

    def test_tanh_act_dense(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(12,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            with self.subTest(f'sum of shape {X_SHAPE}'):
                p = om.Problem()

                def f_wrap(x, mu, z, a, b):
                    return omfunc.act_tanh(x, mu, z, a, b)

                def df_wrap(x, mu, z, a, b, J):
                    df_dx, df_dmu, df_dz, df_da, df_db = omfunc.d_act_tanh(x, mu, z, a, b, sparse=False)
                    J['f', 'x'] = df_dx
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

                # TODO: Test outputs
                # assert_near_equal(p.get_val('sum_comp.sum'),
                #                   np.asarray(np.sum(p.get_val('sum_comp.x'), axis=AXIS, keepdims=AXIS is not None)))
                #
                with np.printoptions(linewidth=1024):
                    cpd = p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)

    def test_tanh_act_sparse(self):
        """ Test sum and its partials for a variety of input shapes and axis options. """
        for X_SHAPE in [(12,), (4, 5), (3, 2, 6), (2, 3, 2, 3), (4, 3, 2, 1, 5)]:
            with self.subTest(f'sum of shape {X_SHAPE}'):
                p = om.Problem()

                ar = np.arange(np.prod(X_SHAPE), dtype=int)

                def f_wrap(x, mu, z, a, b):
                    return omfunc.act_tanh(x, mu, z, a, b)

                def df_wrap(x, mu, z, a, b, J):
                    df_dx, df_dmu, df_dz, df_da, df_db = omfunc.d_act_tanh(x, mu, z, a, b, sparse=True)
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

                # comp = om.ExplicitFuncComp(omfunc.act_tanh,
                #                            compute_partials=lambda x, mu, z, a, b: omfunc.d_act_tanh(x, mu, z, a,
                #                                                                                      b, sparse=True))
                #
                # comp.add_input('x', shape=X_SHAPE)\
                #     .add_input('mu')\
                #     .add_input('z', shape=X_SHAPE)\
                #     .add_input('a')\
                #     .add_input('b')\
                #     .add_output('act_tanh')\
                #     .declare_partials(of='act_tanh', wrt=('x', 'z'), rows=ar, cols=ar)\
                #     .declare_partials(of='act_tanh', wrt=('mu', 'a', 'b'))

                # om.ExplicitFuncComp(f,
                #                     compute_partials=lambda x, mu, z, a, b: omfunc.d_act_tanh(x, mu, z, a, b, sparse=True),
                #                     inputs=[('x', dict(shape=X_SHAPE)),
                #                             'mu',
                #                             ('z', dict(shape=X_SHAPE)),
                #                             'a',
                #                             'b'],
                #                     outputs=([('act_tanh', dict(shape=X_SHAPE))]),
                #                     partials=({('act_tanh', ('x', 'z'): ...}))

                p.model.add_subsystem('f_comp', om.ExplicitFuncComp(f, compute_partials=df_wrap))

                p.setup(force_alloc_complex=True)
                p.set_val('f_comp.x', 10*np.random.random(X_SHAPE))
                p.set_val('f_comp.mu', np.random.random(1,))
                p.set_val('f_comp.z', np.random.random(X_SHAPE))
                p.set_val('f_comp.a', np.random.random(1,))
                p.set_val('f_comp.b', np.random.random(1,))

                p.run_model()

                # TODO: Test outputs
                # assert_near_equal(p.get_val('sum_comp.sum'),
                #                   np.asarray(np.sum(p.get_val('sum_comp.x'), axis=AXIS, keepdims=AXIS is not None)))
                #
                with np.printoptions(linewidth=1024):
                    cpd = p.check_partials(method='cs', out_stream=None)
                assert_check_partials(cpd)
