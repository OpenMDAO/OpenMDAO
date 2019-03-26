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

        # Assert that normalization is happening
        assert_almost_equal(prob.model.balance._scale_factor, 1.0 / np.abs(2))

        with printoptions(linewidth=1024):
            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_create_on_init_no_normalization(self):

        prob = Problem(model=Group())

        bal = BalanceComp('x', val=1.0, normalize=False)

        tgt = IndepVarComp(name='y_tgt', val=1.5)

        exec_comp = ExecComp('y=x**2')

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver()
        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.nonlinear_solver.options['iprint'] = 2

        prob.setup()

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(1.5), decimal=7)

        assert_almost_equal(prob.model.balance._scale_factor, 1.0)

        with printoptions(linewidth=1024):
            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_vectorized(self):

        n = 100

        prob = Problem(model=Group(assembled_jac_type='dense'))

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

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_vectorized_no_normalization(self):

        n = 100

        prob = Problem(model=Group(assembled_jac_type='dense'))

        bal = BalanceComp()

        bal.add_balance('x', val=np.ones(n), normalize=False)

        tgt = IndepVarComp(name='y_tgt', val=1.7*np.ones(n))

        exec_comp = ExecComp('y=x**2', x={'value': np.ones(n)}, y={'value': np.ones(n)})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(1.7), decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_vectorized_with_mult(self):

        n = 100

        prob = Problem(model=Group(assembled_jac_type='dense'))

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

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

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

        prob = Problem(model=Group(assembled_jac_type='dense'))

        bal = BalanceComp('x', val=np.ones(n), use_mult=True, mult_val=2.0)

        tgt = IndepVarComp(name='y_tgt', val=4 * np.ones(n))

        exec_comp = ExecComp('y=x**2', x={'value': np.ones(n)}, y={'value': np.ones(n)})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

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

        prob = Problem(model=Group(assembled_jac_type='dense'))

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

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

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

        prob = Problem(model=Group(assembled_jac_type='dense'))

        bal = BalanceComp()

        bal.add_balance('x', guess_func=lambda inputs, outputs, resids: np.sqrt(inputs['rhs:x']))

        tgt = IndepVarComp(name='y_tgt', val=4)

        exec_comp = ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

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

        prob = Problem(model=Group(assembled_jac_type='dense'))

        bal = BalanceComp()

        bal.add_balance('x', guess_func=lambda inputs, outputs, resids: inputs['guess_x'])
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

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        with printoptions(linewidth=1024):

            cpd = prob.check_partials()

            for (of, wrt) in cpd['balance']:
                assert_almost_equal(cpd['balance'][of, wrt]['abs error'], 0.0, decimal=5)

    def test_scalar_guess_func_using_outputs(self):
        # Implicitly solve -(ax^2 + bx) = c using a BalanceComp.
        # For a=1, b=-4 and c=3, there are solutions at x=1 and x=3.

        # Verify that we can set the guess value (and target a solution) based on outputs.

        ind = IndepVarComp()
        ind.add_output('a', 1)
        ind.add_output('b', -4)
        ind.add_output('c', 3)

        lhs = ExecComp('lhs=-(a*x**2+b*x)')

        bal = BalanceComp()

        def guess_function(inputs, outputs, residuals):
            if outputs['x'] < 0:
                return 5.
            else:
                return 0.

        bal.add_balance(name='x', rhs_name='c', guess_func=guess_function)

        model = Group()

        model.add_subsystem('ind_comp', ind, promotes_outputs=['a', 'b', 'c'])
        model.add_subsystem('lhs_comp', lhs, promotes_inputs=['a', 'b', 'x'])
        model.add_subsystem('bal_comp', bal, promotes_inputs=['c'], promotes_outputs=['x'])

        model.connect('lhs_comp.lhs', 'bal_comp.lhs:x')

        model.linear_solver = DirectSolver()
        model.nonlinear_solver = NewtonSolver(maxiter=1000, iprint=0)

        prob = Problem(model)
        prob.setup()

        # initial value of 'x' less than zero, guess should steer us to solution of 3.
        prob['bal_comp.x'] = -1
        prob.run_model()
        assert_almost_equal(prob['bal_comp.x'], 3.0, decimal=7)

        # initial value of 'x' greater than zero, guess should steer us to solution of 1.
        prob['bal_comp.x'] = 99
        prob.run_model()
        assert_almost_equal(prob['bal_comp.x'], 1.0, decimal=7)

    def test_rhs_val(self):
        """ Test solution with a default RHS value and no connected RHS variable. """

        n = 1

        prob = Problem(model=Group(assembled_jac_type='dense'))

        bal = BalanceComp('x', rhs_val=4.0)

        exec_comp = ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

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

        prob = Problem(model=Group(assembled_jac_type='dense'))

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

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

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

        prob = Problem(model=Group(assembled_jac_type='dense'))

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

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

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

        prob = Problem(model=Group(assembled_jac_type='dense'))

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

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

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

        prob = Problem(model=Group(assembled_jac_type='dense'))

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

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

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

        prob = Problem(model=Group(assembled_jac_type='dense'))

        exec_comp = ExecComp('y=b*x+c',
                             b={'value': np.random.uniform(0.01,100, size=n)},
                             c={'value': np.random.rand(n)},
                             x={'value': np.zeros(n)},
                             y={'value': np.ones(n)})

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=BalanceComp('x', val=np.ones(n)))

        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = NewtonSolver(maxiter=100, iprint=0)

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
