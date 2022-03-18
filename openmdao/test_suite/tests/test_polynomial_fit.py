"""Test the PolynominalFit component. """

import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.polynomial_fit import PolynomialFit
from openmdao.utils.assert_utils import assert_near_equal


class TestPolynomialFit(unittest.TestCase):

    def test_polynominal_fit(self):
        N_cp = 50
        N_predict = 1000

        p = om.Problem()

        fit = p.model.add_subsystem('fit', PolynomialFit(N_cp=N_cp, N_predict=N_predict))
        fit.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=10, iprint=2)
        fit.linear_solver = om.DirectSolver()

        p.setup()

        np.random.seed(0)

        X_cp = np.linspace(0, np.pi, N_cp)
        Y_cp = np.sin(X_cp) + (np.random.random(N_cp) - 0.5)*0.1

        X_true = np.linspace(0, np.pi, N_predict)
        Y_true = np.sin(X_true)

        p['fit.x_cp'] = X_cp
        p['fit.y_cp'] = Y_cp
        p['fit.x'] = X_true

        p.run_model()

        assert_near_equal(p['fit.y'], Y_true, 1e-1)


if __name__ == "__main__":
    unittest.main()
