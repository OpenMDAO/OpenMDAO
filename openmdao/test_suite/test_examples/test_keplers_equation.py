from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp, BalanceComp, \
    ExecComp, DirectSolver, NewtonSolver, DenseJacobian

class TestKeplersEquation(unittest.TestCase):

    def test_result(self):
        import numpy as np
        from numpy.testing import assert_almost_equal

        from openmdao.api import Problem, Group, IndepVarComp, BalanceComp, \
            ExecComp, DirectSolver, NewtonSolver

        prob = Problem(model=Group())

        ivc = IndepVarComp()

        ivc.add_output(name='M',
                       val=0.0,
                       units='deg',
                       desc='Mean anomaly')

        ivc.add_output(name='ecc',
                       val=0.0,
                       units=None,
                       desc='orbit eccentricity')

        bal = BalanceComp()

        def guess_function(inputs, outputs):
            return inputs['M']

        bal.add_balance(name='E', val=0.0, units='rad', eq_units='rad', rhs_name='M',
                        guess_func=guess_function)

        # ExecComp used to compute the LHS of Kepler's equation.
        lhs_comp = ExecComp('lhs=E - ecc * sin(E)',
                            lhs={'value': 0.0, 'units': 'rad'},
                            E={'value': 0.0, 'units': 'rad'},
                            ecc={'value': 0.0})

        prob.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['M', 'ecc'])

        prob.model.add_subsystem(name='balance', subsys=bal,
                                 promotes_inputs=['M'],
                                 promotes_outputs=['E'])

        prob.model.add_subsystem(name='lhs_comp', subsys=lhs_comp,
                                 promotes_inputs=['E', 'ecc'])

        # Explicit connections
        prob.model.connect('lhs_comp.lhs', 'balance.lhs:E')

        # Setup solvers
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup()

        prob['M'] = 85.0
        prob['ecc'] = 0.6

        prob.run_model()

        assert_almost_equal(np.degrees(prob['E']), 115.9, decimal=1)

        print('E (deg) = ', np.degrees(prob['E'][0]))


if __name__ == "__main__":

    unittest.main()