""" Unit tests for AnalysisError with Pyoptsparse Driver."""

import unittest

from collections import defaultdict
from packaging.version import Version

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, parameterized_name, require_pyoptsparse
from openmdao.test_suite.components.paraboloid_invalid_region import Paraboloid

from openmdao.utils.mpi import MPI

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

from openmdao.drivers.pyoptsparse_driver import optlist, grad_drivers, pyoptsparse_version


do_not_test = {
    'NLPY_AUGLAG',  # requires nlpy to be built (raises pyOpt_error.Error)
    'NSGA2',        # PETSc segfault with Analysis Errors, NANs or not
    'PSQP',         # fails nominal with: (4) Maximum constraint value is less than or equal to tolerance
}

@use_tempdirs
@require_pyoptsparse()
class TestPyoptSparseAnalysisErrors(unittest.TestCase):

    # optimizer specific settings
    opt_settings = {
        'ALPSO': {
            'seed': 1.0 if pyoptsparse_version == Version('1.2') else 1
        },
        'IPOPT': {
            'file_print_level': 5
        },
        'SLSQP': {
            'ACC': 1e-9
        },
    }

    # some optimizers may not be able to find the solution within 1e-6
    tolerances = defaultdict(lambda: 1e-6)
    tolerances.update({
        'ALPSO': 1e-3,    # ALPSO gets a pretty bad answer, especially in v1.2
        'CONMIN': 1e-3,   # CONMIN gets a pretty bad answer
        'NSGA2': 1e-1,    # NSGA2 gets a really bad answer
    })

    # invalid range chosen to be on the nominal path of the optimizer
    invalid_range = defaultdict(lambda: {'x': (7.2, 10.2), 'y': (-50.0001, -40.)})
    invalid_range.update({
        'ParOpt': {'x': (4., 6.), 'y': (-4., -6.)},
    })

    expected_result_eval_errors = defaultdict(lambda: 0)
    expected_result_eval_errors.update({
        'CONMIN': None,  # CONMIN does not provide a return code and will just give a bad answer
        'ParOpt': None,  # ParOpt does not provide a return code and will just give a bad answer
    })

    expected_result_grad_errors = defaultdict(lambda: 0)
    expected_result_grad_errors.update({
        'CONMIN': None,  # CONMIN does not provide a return code and will just give a bad answer
        'IPOPT': -13,    # Invalid Number Detected (i.e. NaN)
        'SLSQP': 9,      # Iteration limit exceeded (will just keep trying?)
        'ParOpt': None,  # ParOpt does not provide a return code and will just give a bad answer
    })

    if pyoptsparse_version == Version('1.2'):
        # behavior is different on v1.2 (currently oldest supported version) for these optimizers
        expected_result_eval_errors.update({
            'SLSQP': None,  # SLSQP does not provide a return code and will return NaNs
        })
        expected_result_grad_errors.update({
            'IPOPT': None,  # IPOPT does not provide a return code and will just give a bad answer
            'SLSQP': None,  # SLSQP does not provide a return code and will return NaNs
        })

    def setup_problem(self, optimizer, func=None):
        # Paraboloid model with optional AnalysisErrors
        model = om.Group()

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])

        if func:
            invalid_range = self.invalid_range[optimizer]
            invalid_x = invalid_range['x']
            invalid_y = invalid_range['y']

            comp = model.add_subsystem('comp', Paraboloid(invalid_x, invalid_y, func),
                                       promotes=['*'])
        else:
            comp = model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)

        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.)

        # pyOptSparseDriver with selected optimizer
        driver = om.pyOptSparseDriver(optimizer=optimizer)
        if optimizer in self.opt_settings:
            driver.opt_settings = self.opt_settings[optimizer]
        driver.options['print_results'] = False
        driver.options['output_dir'] = None  # So output goes in current working directory
                                             # that was the location when this test was written

        # setup problem & initialize values
        prob = om.Problem(model, driver)
        prob.setup()

        prob.set_val('x', 50)
        prob.set_val('y', 50)

        return prob, comp

    def check_history(self, optimizer, err_count=None, func='eval'):
        """
        Check the optimizer output file for evaluation errors and successful optimization.
        """
        # make sure there was at least one AnalysisError raised
        if err_count is not None:
            self.assertGreater(err_count, 0, "There was no AnalysisError raised.")

        if optimizer == 'CONMIN':
            # check for NaN in CONMIN.out
            with open("CONMIN.out", encoding="utf-8") as f:
                CONMIN_history = f.readlines()

            if func == 'eval':
                nan_text = "OBJ =            NaN"
            else:
                nan_text = "GRADIENT OF OBJ\n  1)            NaN          NaN"
            errs = CONMIN_history.count(nan_text)

            if err_count is None:
                self.assertEqual(errs, 0,
                                f"Found {errs} unexpected {func} errors in CONMIN.out")
            else:
                self.assertGreater(errs, 0,
                                   f"Found {errs} {func} errors in CONMIN.out, expected {err_count}")

        elif optimizer == 'IPOPT':
            with open("IPOPT.out", encoding="utf-8") as f:
                IPOPT_history = f.read()

            if func == 'eval':
                # check for evaluation error messages in the IPOPT history file
                eval_msg = "Warning: Cutting back alpha due to evaluation error"
                errs = IPOPT_history.count(eval_msg)

                if err_count is None:
                    self.assertEqual(errs, 0,
                                     f"Found {errs} unexpected evaluation errors in IPOPT.out")
                else:
                    self.assertGreater(errs, 0,
                                       f"Found {errs} evaluation errors in IPOPT.out, expected {err_count}")

                # confirm that the optimization completed successfully
                self.assertTrue("EXIT: Optimal Solution Found."
                                in IPOPT_history)
            else:
                # confirm that the optimization failed due to invalid derivatives
                self.assertTrue("EXIT: Invalid number in NLP function or derivative detected."
                                in IPOPT_history)

        elif optimizer == 'SLSQP':
            # there is no information about evaluation/gradient errors in SLSQP.out
            pass

        elif optimizer == 'SNOPT':
            # check for evaluation error flags in the SNOPT history file
            with open("SNOPT_print.out", encoding="utf-8", errors='ignore') as f:
                SNOPT_history = f.readlines()

            itns = False
            errs = 0
            success = False
            for line in SNOPT_history:
                line = line.strip()
                # find the beginning of the Iterations section
                if line.startswith('Itns'):
                    itns = True
                    continue
                # count the number of iterations that encountered an evaluation error
                elif itns and line.endswith(' D'):
                    errs = errs + 1
                # confirm that the optimization completed successfully
                elif line.startswith('SNOPTC EXIT'):
                    success = line.endswith('finished successfully')
                    break

            self.assertTrue(success)

            if err_count is None:
                self.assertEqual(errs, 0,
                                 f"Found {errs} unexpected evaluation errors in SNOPT_print.out")
            else:
                self.assertEqual(errs, err_count,
                                 f"Found {errs} evaluation errors in SNOPT_print.out, expected {err_count}")

    @parameterized.expand(optlist - do_not_test, name_func=parameterized_name)
    def test_analysis_errors_eval(self, optimizer):
        #
        # first optimize without Analysis Errors
        #
        try:
            prob, comp = self.setup_problem(optimizer)
            failed = not prob.run_driver().success
        except ImportError as err:
            raise unittest.SkipTest(str(err))

        self.assertFalse(failed, "Nominal Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        tolerance = self.tolerances[optimizer]

        # make sure we got the right answer
        assert_near_equal(prob['x'], 7.166667, tolerance)
        assert_near_equal(prob['y'], -7.833334, tolerance)

        # check that there are no AnalysisError related messages in the history
        self.check_history(optimizer, err_count=None)

        # save the optimizer's path to the solution
        nominal_history = comp.eval_history

        #
        # Now try raising Analysis Errors in compute()
        #
        prob, comp = self.setup_problem(optimizer, func='compute')
        failed = not prob.run_driver().success

        expected_result = self.expected_result_eval_errors[optimizer]
        opt_inform = prob.driver.pyopt_solution.optInform

        if expected_result == 0:
            # we still expect the right answer
            self.assertFalse(failed,
                             "Optimization with AnalysisErrors failed, info = " +
                             str(prob.driver.pyopt_solution.optInform))

            assert_near_equal(prob['x'], 7.166667, tolerance)
            assert_near_equal(prob['y'], -7.833334, tolerance)

            # but it should take more iterations
            self.assertTrue(len(comp.eval_history) > len(nominal_history),
                            f"Iterations with analysis errors ({len(comp.eval_history)}) is "
                            f"not greater than nominal iterations ({len(nominal_history)})")

            # check that the optimizer output shows the optimizer handling the errors
            self.check_history(optimizer, err_count=len(comp.raised_eval_errors))

        elif expected_result is not None:
            # we expect the optimizer to return an error code
            self.assertEqual(opt_inform['value'], expected_result,
                             f"Optimization was expected to fail with code '{expected_result}'\n" +
                             str(prob.driver.pyopt_solution.optInform))
            self.assertTrue(failed,
                            f"Optimization was expected to fail with code '{expected_result}'\n" +
                            str(prob.driver.pyopt_solution.optInform))

            # check that the optimizer output shows the optimizer was unable to handle the errors
            self.check_history(optimizer, err_count=len(comp.raised_eval_errors), func='eval')

    @parameterized.expand(grad_drivers - do_not_test, name_func=parameterized_name)
    def test_analysis_errors_grad(self, optimizer):
        #
        # first optimize without Analysis Errors
        #
        try:
            prob, comp = self.setup_problem(optimizer)
            failed = not prob.run_driver().success
        except ImportError as err:
            raise unittest.SkipTest(str(err))

        self.assertFalse(failed, "Nominal Optimization failed, info = " +
                                 str(prob.driver.pyopt_solution.optInform))

        tolerance = self.tolerances[optimizer]

        # make sure we got the right answer
        assert_near_equal(prob['x'], 7.166667, tolerance)
        assert_near_equal(prob['y'], -7.833334, tolerance)

        # check that there are no AnalysisError related messages in the history
        self.check_history(optimizer, err_count=None)

        # save the optimizer's path to the solution
        nominal_history = comp.eval_history

        #
        # Now try raising Analysis Errors in compute_partials()
        #
        try:
            prob, comp = self.setup_problem(optimizer, func='compute_partials')
            failed = not prob.run_driver().success
        except ImportError as err:
            raise unittest.SkipTest(str(err))

        expected_result = self.expected_result_grad_errors[optimizer]
        opt_inform = prob.driver.pyopt_solution.optInform

        if expected_result == 0:
            # we still expect the right answer
            self.assertFalse(failed,
                             "Optimization with AnalysisErrors failed, info = " +
                             str(opt_inform))

            tolerance = self.tolerances[optimizer]
            assert_near_equal(prob['x'], 7.166667, tolerance)
            assert_near_equal(prob['y'], -7.833334, tolerance)

            # but it should take more iterations
            self.assertTrue(len(comp.eval_history) > len(nominal_history),
                            f"Iterations with analysis errors ({len(comp.eval_history)}) is "
                            f"not greater than nominal iterations ({len(nominal_history)})")

            # check that the optimizer output shows the optimizer handling the errors
            self.check_history(optimizer, err_count=len(comp.raised_grad_errors))

        elif expected_result is not None:
            # we expect the optimizer to return an error code
            self.assertEqual(opt_inform['value'], expected_result,
                             f"Optimization was expected to fail with code '{expected_result}'\n" +
                             str(opt_inform))
            self.assertTrue(failed,
                            f"Optimization was expected to fail with code '{expected_result}'\n" +
                            str(opt_inform))

            # check that the optimizer output shows the optimizer was unable to handle the errors
            self.check_history(optimizer, err_count=len(comp.raised_grad_errors), func='grad')


if __name__ == "__main__":
    unittest.main()
