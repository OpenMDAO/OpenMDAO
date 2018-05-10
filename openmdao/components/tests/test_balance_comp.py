from __future__ import print_function, division, absolute_import

import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from openmdao.api import Problem, Group, IndepVarComp, ExecComp, \
    NewtonSolver, DirectSolver
from openmdao.api import BalanceComp
from openmdao.utils.general_utils import printoptions


class TestBalanceComp(unittest.TestCase):


    def test_scalar_example(self):

        prob = Problem(model=Group())

        bal = BalanceComp()

        bal.add_balance('x', val=1.0)

        tgt = IndepVarComp(name='y_tgt', val=2)

        exec_comp = ExecComp('y=x**2')

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        # do one test in an unconverged state, to capture accuracy of partials
        prob.setup()

        prob['y_tgt'] = 100000 #set rhs and lhs to very different values. Trying to capture some derivatives wrt
        prob['exec.y'] = .001

        prob.run_model()

        cpd = prob.check_partials()

        for (of, wrt) in cpd['balance']:
            assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

        # set an actual solver, and re-setup. Then check derivatives at a converged point
        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver = NewtonSolver()

        prob.setup()

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        with printoptions(linewidth=1024):
            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_create_on_init(self):

        prob = Problem(model=Group())

        bal = BalanceComp('x', val=1.0)

        tgt = IndepVarComp(name='y_tgt', val=2)

        exec_comp = ExecComp('y=x**2')

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver = NewtonSolver()

        prob.setup()

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        with printoptions(linewidth=1024):
            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_vectorized(self):

        n = 100

        prob = Problem(model=Group())

        bal = BalanceComp()

        bal.add_balance('x', val=np.ones(n))

        tgt = IndepVarComp(name='y_tgt', val=4*np.ones(n))

        exec_comp = ExecComp('y=x**2', x={'value': np.ones(n)}, y={'value': np.ones(n)})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_vectorized_with_mult(self):

        n = 100

        prob = Problem(model=Group())

        bal = BalanceComp()

        bal.add_balance('x', val=np.ones(n), use_mult=True)

        tgt = IndepVarComp(name='y_tgt', val=4*np.ones(n))

        mult_ivc = IndepVarComp(name='mult', val=2.0*np.ones(n))

        exec_comp = ExecComp('y=x**2', x={'value': np.ones(n)}, y={'value': np.ones(n)})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        prob.model.add_subsystem(name='mult_comp', subsys=mult_ivc, promotes_outputs=['mult'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('mult', 'balance.mult:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_vectorized_with_default_mult(self):
        """
        solve:  2 * x**2 = 4
        expected solution:  x=sqrt(2)
        """

        n = 100

        prob = Problem(model=Group())

        bal = BalanceComp('x', val=np.ones(n), use_mult=True, mult_val=2.0)

        tgt = IndepVarComp(name='y_tgt', val=4 * np.ones(n))

        exec_comp = ExecComp('y=x**2', x={'value': np.ones(n)}, y={'value': np.ones(n)})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_scalar(self):

        n = 1

        prob = Problem(model=Group())

        bal = BalanceComp()

        bal.add_balance('x')

        tgt = IndepVarComp(name='y_tgt', val=4)

        exec_comp = ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_scalar_with_guess_func(self):

        n = 1

        prob = Problem(model=Group())

        bal = BalanceComp()

        bal.add_balance('x', guess_func=lambda inputs, resids: np.sqrt(inputs['rhs:x']))

        tgt = IndepVarComp(name='y_tgt', val=4)

        exec_comp = ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_scalar_with_guess_func_additional_input(self):

        n = 1

        prob = Problem(model=Group())

        bal = BalanceComp()

        bal.add_balance('x', guess_func=lambda inputs, resids: inputs['guess_x'])
        bal.add_input('guess_x', val=0.0)

        ivc = IndepVarComp()
        ivc.add_output(name='y_tgt', val=4)
        ivc.add_output(name='guess_x', val=2.5)

        exec_comp = ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['y_tgt', 'guess_x'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('guess_x', 'balance.guess_x')
        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_rhs_val(self):
        """ Test solution with a default RHS value and no connected RHS variable. """

        n = 1

        prob = Problem(model=Group())

        bal = BalanceComp('x', rhs_val=4.0)

        exec_comp = ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_scalar_with_mult(self):

        n = 1

        prob = Problem(model=Group())

        bal = BalanceComp()

        bal.add_balance('x', use_mult=True)

        tgt = IndepVarComp(name='y_tgt', val=4)

        mult_ivc = IndepVarComp(name='mult', val=2.0)

        exec_comp = ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        prob.model.add_subsystem(name='mult_comp', subsys=mult_ivc, promotes_outputs=['mult'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('mult', 'balance.mult:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_renamed_vars(self):

        n = 1

        prob = Problem(model=Group())

        bal = BalanceComp()

        bal.add_balance('x', use_mult=True, mult_name='MUL', lhs_name='XSQ', rhs_name='TARGETXSQ')

        tgt = IndepVarComp(name='y_tgt', val=4)

        mult_ivc = IndepVarComp(name='mult', val=2.0)

        exec_comp = ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        prob.model.add_subsystem(name='mult_comp', subsys=mult_ivc, promotes_outputs=['mult'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.TARGETXSQ')
        prob.model.connect('mult', 'balance.MUL')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.XSQ')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_feature_scalar(self):
        from numpy.testing import assert_almost_equal
        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, NewtonSolver, \
            DirectSolver, BalanceComp

        prob = Problem(model=Group())

        bal = BalanceComp()

        bal.add_balance('x', use_mult=True)

        tgt = IndepVarComp(name='y_tgt', val=4)

        mult_ivc = IndepVarComp(name='mult', val=2.0)

        exec_comp = ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        prob.model.add_subsystem(name='mult_comp', subsys=mult_ivc, promotes_outputs=['mult'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('mult', 'balance.mult:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup(check=False)

        # A reasonable initial guess to find the positive root.
        prob['balance.x'] = 1.0

        prob.run_model()

        print('x = ', prob['balance.x'])

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

    def test_feature_scalar_with_default_mult(self):
        from numpy.testing import assert_almost_equal
        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, NewtonSolver, \
            DirectSolver, BalanceComp

        prob = Problem(model=Group())

        bal = BalanceComp()

        bal.add_balance('x', use_mult=True, mult_val=2.0)

        tgt = IndepVarComp(name='y_tgt', val=4)

        exec_comp = ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup(check=False)

        # A reasonable initial guess to find the positive root.
        prob['balance.x'] = 1.0

        prob.run_model()

        print('x = ', prob['balance.x'])

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

    def test_feature_vector(self):
        import numpy as np
        from numpy.testing import assert_almost_equal

        from openmdao.api import Problem, Group, ExecComp, NewtonSolver, DirectSolver, BalanceComp

        n = 100

        prob = Problem(model=Group())

        exec_comp = ExecComp('y=b*x+c',
                             b={'value': np.random.uniform(0.01,100, size=n)},
                             c={'value': np.random.rand(n)},
                             x={'value': np.zeros(n)},
                             y={'value': np.ones(n)})

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=BalanceComp('x', val=np.ones(n)))

        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assembled_jac='dense')

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 100
        prob.model.nonlinear_solver.options['iprint'] = 0

        prob.setup(check=False)

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        b = prob['exec.b']
        c = prob['exec.c']

        assert_almost_equal(prob['balance.x'], -c/b, decimal=6)

        print('solution')
        print(prob['balance.x'])
        print('expected')
        print(-c/b)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
