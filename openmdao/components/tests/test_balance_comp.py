"""
Unit tests for the BalanceComp.
"""
import unittest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om
from openmdao.utils.assert_utils import assert_warning, assert_no_warning, assert_check_partials


class TestBalanceComp(unittest.TestCase):

    def test_scalar_example(self):

        prob = om.Problem()

        bal = om.BalanceComp()
        bal.add_balance('x', val=1.0)

        tgt = om.IndepVarComp(name='y_tgt', val=2)

        exec_comp = om.ExecComp('y=x**2')

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

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=2e-5, rtol=2e-5)

        # set an actual solver, and re-setup. Then check derivatives at a converged point
        prob.model.linear_solver = om.DirectSolver()
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        prob.setup()

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_balance_comp_with_units_kwarg_and_eq_units(self):

        prob = om.Problem()

        bal = om.BalanceComp()
        bal.add_balance('x', val=1.0, eq_units='m', units='m')

        tgt = om.IndepVarComp(name='y_tgt', val=2)

        exec_comp = om.ExecComp('y=x**2')

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        prob.model.add_subsystem(name='exec', subsys=exec_comp)
        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.setup()

        prob.run_model()
        meta = prob.model._var_abs2meta
        self.assertEqual(meta['output']['balance.x']['units'], 'm')
        self.assertEqual(meta['input']['balance.rhs:x']['units'], 'm')
        self.assertEqual(meta['input']['balance.lhs:x']['units'], 'm')

    def test_create_on_init(self):

        prob = om.Problem()

        bal = om.BalanceComp('x', val=1.0)

        tgt = om.IndepVarComp(name='y_tgt', val=2)

        exec_comp = om.ExecComp('y=x**2')

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        prob.model.add_subsystem(name='exec', subsys=exec_comp)
        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver()
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        prob.setup()

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        # Assert that normalization is happening
        assert_almost_equal(prob.model.balance._scale_factor, 1.0 / np.abs(2))

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_balance_comp_options_exclude_no_error(self):

        prob = om.Problem()

        bal = om.BalanceComp()
        bal.add_balance('x', val=1.0)

        prob.model.add_subsystem(name='balance', subsys=bal)

        recorder = om.SqliteRecorder('cases.sql')

        prob.model.add_recorder(recorder)

        prob.model.recording_options['record_inputs'] = True
        prob.model.recording_options['record_outputs'] = True
        prob.model.recording_options['record_residuals'] = True

        prob.setup()

        msg = ("Trying to record option 'guess_func' which cannot be pickled on system BalanceComp "
               "(balance). Set 'recordable' to False. Skipping recording options for this system.")

        with assert_no_warning(UserWarning, msg):
            prob.run_model()

        prob = om.Problem()

        bal = om.BalanceComp()
        bal.add_balance('x', val=1.0)

        prob.model.add_subsystem(name='balance', subsys=bal)

        recorder = om.SqliteRecorder('cases.sql')

        prob.model.add_recorder(recorder)

        bal.recording_options['options_excludes'] = ['guess_func']

        prob.setup()

        with assert_no_warning(UserWarning, msg):
            prob.run_model()

    def test_create_on_init_no_normalization(self):

        prob = om.Problem()

        bal = om.BalanceComp('x', val=1.0, normalize=False)

        tgt = om.IndepVarComp(name='y_tgt', val=1.5)

        exec_comp = om.ExecComp('y=x**2')

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver()
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=0)

        prob.setup()

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(1.5), decimal=7)

        assert_almost_equal(prob.model.balance._scale_factor, 1.0)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_vectorized(self):

        n = 100

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))

        bal = om.BalanceComp()

        bal.add_balance('x', val=np.ones(n))

        tgt = om.IndepVarComp(name='y_tgt', val=4*np.ones(n))

        exec_comp = om.ExecComp('y=x**2', x={'value': np.ones(n)}, y={'value': np.ones(n)})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=2e-5, rtol=2e-5)

    def test_vectorized_no_normalization(self):

        n = 100

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))

        bal = om.BalanceComp()

        bal.add_balance('x', val=np.ones(n), normalize=False)

        tgt = om.IndepVarComp(name='y_tgt', val=1.7*np.ones(n))

        exec_comp = om.ExecComp('y=x**2', x={'value': np.ones(n)}, y={'value': np.ones(n)})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(1.7), decimal=7)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_vectorized_with_mult(self):

        n = 100

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))

        bal = om.BalanceComp()

        bal.add_balance('x', val=np.ones(n), use_mult=True)

        tgt = om.IndepVarComp(name='y_tgt', val=4*np.ones(n))

        mult_ivc = om.IndepVarComp(name='mult', val=2.0*np.ones(n))

        exec_comp = om.ExecComp('y=x**2', x={'value': np.ones(n)}, y={'value': np.ones(n)})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        prob.model.add_subsystem(name='mult_comp', subsys=mult_ivc, promotes_outputs=['mult'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('mult', 'balance.mult:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_vectorized_with_default_mult(self):
        """
        solve:  2 * x**2 = 4
        expected solution:  x=sqrt(2)
        """

        n = 100

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))

        bal = om.BalanceComp('x', val=np.ones(n), use_mult=True, mult_val=2.0)

        tgt = om.IndepVarComp(name='y_tgt', val=4 * np.ones(n))

        exec_comp = om.ExecComp('y=x**2', x={'value': np.ones(n)}, y={'value': np.ones(n)})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_shape(self):
        n = 100

        bal = om.BalanceComp()
        bal.add_balance('x', shape=(n,))

        tgt = om.IndepVarComp(name='y_tgt', val=4*np.ones(n))

        exe = om.ExecComp('y=x**2', x=np.zeros(n), y=np.zeros(n))

        model = om.Group()

        model.add_subsystem('tgt', tgt, promotes_outputs=['y_tgt'])
        model.add_subsystem('exe', exe)
        model.add_subsystem('bal', bal)

        model.connect('y_tgt', 'bal.rhs:x')
        model.connect('bal.x', 'exe.x')
        model.connect('exe.y', 'bal.lhs:x')

        model.linear_solver = om.DirectSolver(assemble_jac=True)
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob = om.Problem(model)
        prob.setup()

        prob['bal.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['bal.x'], 2.0*np.ones(n), decimal=7)

    def test_complex_step(self):

        n = 1

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))

        bal = om.BalanceComp()

        bal.add_balance('x')

        tgt = om.IndepVarComp(name='y_tgt', val=4)

        exec_comp = om.ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup(force_alloc_complex=True)

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        with warnings.catch_warnings():
            warnings.filterwarnings(action="error", category=np.ComplexWarning)
            cpd = prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(cpd, atol=1e-10, rtol=1e-10)

    def test_scalar(self):

        n = 1

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))

        bal = om.BalanceComp()

        bal.add_balance('x')

        tgt = om.IndepVarComp(name='y_tgt', val=4)

        exec_comp = om.ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_scalar_with_guess_func(self):

        n = 1

        model=om.Group(assembled_jac_type='dense')

        def guess_function(inputs, outputs, residuals):
            outputs['x'] = np.sqrt(inputs['rhs:x'])

        bal = om.BalanceComp('x', guess_func=guess_function)  # test guess_func as kwarg

        tgt = om.IndepVarComp(name='y_tgt', val=4)

        exec_comp = om.ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        model.add_subsystem(name='exec', subsys=exec_comp)
        model.add_subsystem(name='balance', subsys=bal)

        model.connect('y_tgt', 'balance.rhs:x')
        model.connect('balance.x', 'exec.x')
        model.connect('exec.y', 'balance.lhs:x')

        model.linear_solver = om.DirectSolver(assemble_jac=True)
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob = om.Problem(model)
        prob.setup()

        prob['balance.x'] = np.random.rand(n)
        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        # should converge with no iteration due to the guess function
        self.assertEqual(model.nonlinear_solver._iter_count, 1)

        cpd = prob.check_partials(out_stream=None)
        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_scalar_with_guess_func_additional_input(self):

        model = om.Group(assembled_jac_type='dense')

        bal = om.BalanceComp()
        bal.add_balance('x')
        bal.add_input('guess_x', val=0.0)

        ivc = om.IndepVarComp()
        ivc.add_output(name='y_tgt', val=4)
        ivc.add_output(name='guess_x', val=2.5)

        exec_comp = om.ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['y_tgt', 'guess_x'])
        model.add_subsystem(name='exec', subsys=exec_comp)
        model.add_subsystem(name='balance', subsys=bal)

        model.connect('guess_x', 'balance.guess_x')
        model.connect('y_tgt', 'balance.rhs:x')
        model.connect('balance.x', 'exec.x')
        model.connect('exec.y', 'balance.lhs:x')

        model.linear_solver = om.DirectSolver(assemble_jac=True)
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob = om.Problem(model)
        prob.setup()

        # run problem without a guess function
        prob['balance.x'] = .5
        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        iters_no_guess = model.nonlinear_solver._iter_count

        # run problem with same initial value and a guess function
        def guess_function(inputs, outputs, resids):
            outputs['x'] = inputs['guess_x']

        bal.options['guess_func'] = guess_function

        prob['balance.x'] = .5
        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        iters_with_guess = model.nonlinear_solver._iter_count

        # verify it converges faster with the guess function
        self.assertTrue(iters_with_guess < iters_no_guess)

    def test_scalar_guess_func_using_outputs(self):

        model = om.Group()

        ind = om.IndepVarComp()
        ind.add_output('a', 1)
        ind.add_output('b', -4)
        ind.add_output('c', 3)

        lhs = om.ExecComp('lhs=-(a*x**2+b*x)')
        bal = om.BalanceComp(name='x', rhs_name='c')

        model.add_subsystem('ind_comp', ind, promotes_outputs=['a', 'b', 'c'])
        model.add_subsystem('lhs_comp', lhs, promotes_inputs=['a', 'b', 'x'])
        model.add_subsystem('bal_comp', bal, promotes_inputs=['c'], promotes_outputs=['x'])

        model.connect('lhs_comp.lhs', 'bal_comp.lhs:x')

        model.linear_solver = om.DirectSolver()
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        # first verify behavior of the balance comp without the guess function
        # at initial conditions x=5, x=0 and x=-1
        prob = om.Problem(model)
        prob.setup()

        # default solution with initial value of 5 is x=3.
        prob['x'] = 5
        prob.run_model()
        assert_almost_equal(prob['x'], 3.0, decimal=7)

        # default solution with initial value of 0 is x=1.
        prob['x'] = 0
        prob.run_model()
        assert_almost_equal(prob['x'], 1.0, decimal=7)

        # default solution with initial value of -1 is x=1.
        prob['x'] = -1
        prob.run_model()
        assert_almost_equal(prob['x'], 1.0, decimal=7)

        # now use a guess function that steers us to the x=3 solution only
        # if the initial value of x is less than zero
        def guess_function(inputs, outputs, residuals):
            if outputs['x'] < 0:
                outputs['x'] = 3.

        bal.options['guess_func'] = guess_function

        # solution with initial value of 5 is still x=3.
        prob['x'] = 5
        prob.run_model()
        assert_almost_equal(prob['x'], 3.0, decimal=7)

        # solution with initial value of 0 is still x=1.
        prob['x'] = 0
        prob.run_model()
        assert_almost_equal(prob['x'], 1.0, decimal=7)

        # solution with initial value of -1 is now x=3.
        prob['x'] = -1
        prob.run_model()
        assert_almost_equal(prob['x'], 3.0, decimal=7)

    def test_rhs_val(self):
        """ Test solution with a default RHS value and no connected RHS variable. """

        n = 1

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))

        bal = om.BalanceComp('x', rhs_val=4.0)

        exec_comp = om.ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], 2.0, decimal=7)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_scalar_with_mult(self):

        n = 1

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))

        bal = om.BalanceComp()

        bal.add_balance('x', use_mult=True)

        tgt = om.IndepVarComp(name='y_tgt', val=4)

        mult_ivc = om.IndepVarComp(name='mult', val=2.0)

        exec_comp = om.ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        prob.model.add_subsystem(name='mult_comp', subsys=mult_ivc, promotes_outputs=['mult'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('mult', 'balance.mult:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_renamed_vars(self):

        n = 1

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))

        bal = om.BalanceComp()

        bal.add_balance('x', use_mult=True, mult_name='MUL', lhs_name='XSQ', rhs_name='TARGETXSQ')

        tgt = om.IndepVarComp(name='y_tgt', val=4)

        mult_ivc = om.IndepVarComp(name='mult', val=2.0)

        exec_comp = om.ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        prob.model.add_subsystem(name='mult_comp', subsys=mult_ivc, promotes_outputs=['mult'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.TARGETXSQ')
        prob.model.connect('mult', 'balance.MUL')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.XSQ')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.rand(n)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_feature_scalar(self):
        from numpy.testing import assert_almost_equal
        import openmdao.api as om

        prob = om.Problem()

        bal = om.BalanceComp()
        bal.add_balance('x', use_mult=True)

        exec_comp = om.ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='exec', subsys=exec_comp)
        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob.set_val('balance.rhs:x', 4)
        prob.set_val('balance.mult:x', 2.)

        # A reasonable initial guess to find the positive root.
        prob['balance.x'] = 1.0

        prob.run_model()

        assert_almost_equal(prob.get_val('balance.x'), np.sqrt(2), decimal=7)

    def test_feature_scalar_with_default_mult(self):
        from numpy.testing import assert_almost_equal
        import openmdao.api as om

        prob = om.Problem()

        bal = om.BalanceComp()
        bal.add_balance('x', use_mult=True, mult_val=2.0)

        exec_comp = om.ExecComp('y=x**2', x={'value': 1}, y={'value': 1})

        prob.model.add_subsystem(name='exec', subsys=exec_comp)
        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob.set_val('balance.rhs:x', 4)

        # A reasonable initial guess to find the positive root.
        prob.set_val('balance.x', 1.0)

        prob.run_model()

        assert_almost_equal(prob.get_val('balance.x'), np.sqrt(2), decimal=7)

    def test_feature_vector(self):
        import numpy as np
        from numpy.testing import assert_almost_equal

        import openmdao.api as om

        n = 100

        prob = om.Problem()

        exec_comp = om.ExecComp('y=b*x+c',
                                b={'value': np.random.uniform(0.01, 100, size=n)},
                                c={'value': np.random.rand(n)},
                                x={'value': np.zeros(n)},
                                y={'value': np.ones(n)})

        prob.model.add_subsystem(name='exec', subsys=exec_comp)
        prob.model.add_subsystem(name='balance', subsys=om.BalanceComp('x', val=np.ones(n)))

        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob.set_val('balance.x', np.random.rand(n))

        prob.run_model()

        b = prob.get_val('exec.b')
        c = prob.get_val('exec.c')

        assert_almost_equal(prob.get_val('balance.x'), -c/b, decimal=6)
        assert_almost_equal(-c/b, prob.get_val('balance.x'), decimal=6)  # expected

    def test_specified_shape(self):
        shape = (3, 2, 4)

        prob = om.Problem(model=om.Group(assembled_jac_type='dense'))

        bal = om.BalanceComp()

        bal.add_balance('x', val=np.ones(shape), use_mult=True)

        tgt = om.IndepVarComp(name='y_tgt', val=4*np.ones(shape))

        mult_ivc = om.IndepVarComp(name='mult', val=2.0*np.ones(shape))

        exec_comp = om.ExecComp('y=x**2', x={'value': np.ones(shape)}, y={'value': np.ones(shape)})

        prob.model.add_subsystem(name='target', subsys=tgt, promotes_outputs=['y_tgt'])
        prob.model.add_subsystem(name='mult_comp', subsys=mult_ivc, promotes_outputs=['mult'])

        prob.model.add_subsystem(name='exec', subsys=exec_comp)

        prob.model.add_subsystem(name='balance', subsys=bal)

        prob.model.connect('y_tgt', 'balance.rhs:x')
        prob.model.connect('mult', 'balance.mult:x')
        prob.model.connect('balance.x', 'exec.x')
        prob.model.connect('exec.y', 'balance.lhs:x')

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=100, iprint=0)

        prob.setup()

        prob['balance.x'] = np.random.random(shape)

        prob.run_model()

        assert_almost_equal(prob['balance.x'], np.sqrt(2), decimal=7)

        cpd = prob.check_partials(out_stream=None)

        assert_check_partials(cpd, atol=1e-5, rtol=1e-5)

    def test_shape_from_rhs_val(self):
        p = om.Problem()
        init = np.ones((5, ))
        p.model.add_subsystem('bal', om.BalanceComp('x', rhs_val=init))

        # Bug was a size mismatch exception raised during setup.
        p.setup()

        self.assertTrue(p.get_val('bal.x').shape == init.shape)
        self.assertTrue(p.get_val('bal.lhs:x').shape == init.shape)
        self.assertTrue(p.get_val('bal.rhs:x').shape == init.shape)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
