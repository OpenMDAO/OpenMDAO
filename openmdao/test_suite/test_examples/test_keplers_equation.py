import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om


class TestKeplersEquation(unittest.TestCase):

    def test_result(self):
        import numpy as np
        from numpy.testing import assert_almost_equal

        import openmdao.api as om

        prob = om.Problem()

        bal = om.BalanceComp()

        bal.add_balance(name='E', val=0.0, units='rad', eq_units='rad', rhs_name='M')

        # Use M (mean anomaly) as the initial guess for E (eccentric anomaly)
        def guess_function(inputs, outputs, residuals):
            if np.abs(residuals['E']) > 1.0E-2:
                outputs['E'] = inputs['M']

        bal.options['guess_func'] = guess_function

        # ExecComp used to compute the LHS of Kepler's equation.
        lhs_comp = om.ExecComp('lhs=E - ecc * sin(E)',
                               lhs={'value': 0.0, 'units': 'rad'},
                               E={'value': 0.0, 'units': 'rad'},
                               ecc={'value': 0.0})

        prob.model.add_subsystem(name='balance', subsys=bal,
                                 promotes_inputs=['M'],
                                 promotes_outputs=['E'])

        prob.model.set_input_defaults('M', 85.0, units='deg')

        prob.model.add_subsystem(name='lhs_comp', subsys=lhs_comp,
                                 promotes_inputs=['E', 'ecc'])

        # Explicit connections
        prob.model.connect('lhs_comp.lhs', 'balance.lhs:E')

        # Set up solvers
        prob.model.linear_solver = om.DirectSolver()
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=2)

        prob.setup()

        prob.set_val('ecc', 0.6)

        prob.run_model()

        assert_almost_equal(np.degrees(prob.get_val('E')), 115.9, decimal=1)


if __name__ == "__main__":

    unittest.main()
