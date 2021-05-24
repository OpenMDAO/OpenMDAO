
# pylint: disable-msg=C0111,C0103

import unittest
import itertools
import numpy as np
import os

from openmdao.api import KrigingSurrogate
from openmdao.utils.assert_utils import assert_near_equal


def branin(x):
    y = (x[1] - (5.1 / (4. * np.pi ** 2.)) * x[0] ** 2. + 5. * x[0] / np.pi - 6.) ** 2. \
        + 10. * (1. - 1. / (8. * np.pi)) * np.cos(x[0]) + 10.
    return y


def branin_1d(x):
    return branin(np.array([x[0], 2.275]))


class TestKrigingSurrogate(unittest.TestCase):

    def test_1d_training(self):

        x = np.array([[0.0], [2.0], [3.0], [4.0], [6.0]])
        y = np.array([[branin_1d(case)] for case in x])
        surrogate = KrigingSurrogate(nugget=0., eval_rmse=True)
        surrogate.train(x, y)

        for x0, y0 in zip(x, y):
            mu, sigma = surrogate.predict(x0)
            assert_near_equal(mu, [y0], 1e-9)
            assert_near_equal(sigma, [[0]], 1e-5)

    def test_1d_predictor(self):
        x = np.array([[0.0], [2.0], [3.0], [4.0], [6.0]])
        y = np.array([[branin_1d(case)] for case in x])

        surrogate = KrigingSurrogate(eval_rmse=True)
        surrogate.train(x, y)

        new_x = np.array([3.5])
        mu, sigma = surrogate.predict(new_x)

        assert_near_equal(mu, [[branin_1d(new_x)]], 1e-1)
        assert_near_equal(sigma, [[0.07101449]], 1e-2)

    def test_1d_ill_conditioned(self):
        # Test for least squares solver utilization when ill-conditioned
        x = np.array([[case] for case in np.linspace(0., 1., 40)])
        y = np.sin(x)
        surrogate = KrigingSurrogate(eval_rmse=True)
        surrogate.train(x, y)
        new_x = np.array([0.5])
        mu, sigma = surrogate.predict(new_x)
        self.assertTrue(sigma < 1.e-5)
        assert_near_equal(mu, [[np.sin(0.5)]], 1e-5)

    def test_2d(self):

        x = np.array([[-2., 0.], [-0.5, 1.5], [1., 3.], [8.5, 4.5],
                      [-3.5, 6.], [4., 7.5], [-5., 9.], [5.5, 10.5],
                      [10., 12.], [7., 13.5], [2.5, 15.]])
        y = np.array([[branin(case)] for case in x])

        surrogate = KrigingSurrogate(nugget=0., eval_rmse=True)
        surrogate.train(x, y)

        for x0, y0 in zip(x, y):
            mu, sigma = surrogate.predict(x0)
            assert_near_equal(mu, [y0], 1e-9)
            assert_near_equal(sigma, [[0]], 1e-4)

        mu, sigma = surrogate.predict([5., 5.])

        assert_near_equal(mu, [[16.72]], 1e-1)
        assert_near_equal(sigma, [[15.27]], 1e-2)

    def test_no_training_data(self):
        surrogate = KrigingSurrogate()

        try:
            surrogate.predict([0., 1.])
        except RuntimeError as err:
            self.assertEqual(str(err),
                             "KrigingSurrogate has not been trained, "
                             "so no prediction can be made.")
        else:
            self.fail("RuntimeError Expected")

    def test_one_pt(self):
        surrogate = KrigingSurrogate()
        x = [[0.]]
        y = [[1.]]

        with self.assertRaises(ValueError) as cm:
            surrogate.train(x, y)

        self.assertEqual(str(cm.exception), 'KrigingSurrogate requires at least'
                                            ' 2 training points.')

    def test_vector_input(self):
        surrogate = KrigingSurrogate(nugget=0., eval_rmse=True)

        x = np.array([[0., 0., 0.], [1., 1., 1.]])
        y = np.array([[0.], [3.]])

        surrogate.train(x, y)

        for x0, y0 in zip(x, y):
            mu, sigma = surrogate.predict(x0)
            assert_near_equal(mu, [y0], 1e-9)
            assert_near_equal(sigma, [[0]], 1e-6)

    def test_vector_output(self):
        surrogate = KrigingSurrogate(nugget=0., eval_rmse=True)

        y = np.array([[0., 0.], [1., 1.], [2., 0.]])
        x = np.array([[0.], [2.], [4.]])

        surrogate.train(x, y)

        for x0, y0 in zip(x, y):
            mu, sigma = surrogate.predict(x0)
            assert_near_equal(mu, [y0], 1e-9)
            assert_near_equal(sigma, [[0, 0]], 1e-6)

    def test_scalar_derivs(self):
        surrogate = KrigingSurrogate(nugget=0.)

        x = np.array([[0.], [1.], [2.], [3.], [4.]])
        y = x.copy()

        surrogate.train(x, y)
        jac = surrogate.linearize(np.array([[0.]]))

        assert_near_equal(jac[0][0], 1., 5e-3)

    def test_vector_derivs(self):
        surrogate = KrigingSurrogate()
        n = 15
        x = np.array([[a, b] for a, b in
                      itertools.product(np.linspace(0, 1, n), repeat=2)])
        y = np.array([[a+b, a-b, a+2*b] for a, b in x])

        surrogate.train(x, y)
        jac = surrogate.linearize(np.array([[0.5, 0.5]]))
        assert_near_equal(jac, np.array([[1, 1], [1, -1], [1, 2]]), 6e-4)

    def test_cache(self):
        x = np.array([[-2., 0.], [-0.5, 1.5], [1., 3.], [8.5, 4.5],
                      [-3.5, 6.], [4., 7.5], [-5., 9.], [5.5, 10.5],
                      [10., 12.], [7., 13.5], [2.5, 15.]])
        y = np.array([[branin(case)] for case in x])

        surrogate_before = KrigingSurrogate(nugget=0., eval_rmse=True, training_cache='test_cache.npz')
        surrogate_before.train(x, y)

        surrogate = KrigingSurrogate(nugget=0., eval_rmse=True, training_cache='test_cache.npz')
        surrogate.train(x, y)

        for x0, y0 in zip(x, y):
            mu, sigma = surrogate.predict(x0)
            assert_near_equal(mu, [y0], 1e-9)
            assert_near_equal(sigma, [[0]], 1e-4)

        mu, sigma = surrogate.predict([5., 5.])

        assert_near_equal(mu, [[16.72]], 1e-1)
        assert_near_equal(sigma, [[15.27]], 1e-2)

        os.unlink('test_cache.npz')


if __name__ == "__main__":
    unittest.main()
