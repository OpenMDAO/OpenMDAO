import unittest

import numpy as np

import openmdao.api as om

from openmdao.test_suite.components.sellar import SellarDerivativesGrouped

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from openmdao.recorders.view_cases import view_cases

try:
    import panel
except ImportError:
    panel = None

@use_tempdirs
@unittest.skipUnless(panel, "requires 'panel'")
class TestViewCases(unittest.TestCase):

    def setup_sellar_problem(self, problem_recorder_filename=None, driver_recorder_filename=None):
        prob = om.Problem()
        prob.model = SellarDerivativesGrouped()

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SLSQP"
        prob.driver.options['print_results'] = False
        prob.driver.opt_settings['ACC'] = 1e-13
        prob.set_solver_print(level=0)
        prob.model.add_constraint('con2', upper=0.0)
        prob.model.add_objective('obj')
        prob.model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 4.0]))
        prob.model.add_design_var('x')
        prob.model.add_constraint('con1', upper=0.0)

        if problem_recorder_filename:
            problem_recorder = om.SqliteRecorder(problem_recorder_filename)
            prob.model.add_recorder(problem_recorder)
        if driver_recorder_filename:
            driver_recorder = om.SqliteRecorder(driver_recorder_filename)
            prob.driver.add_recorder(driver_recorder)

        return prob


    @require_pyoptsparse('SLSQP')
    def test_viewing_problem_case_recorder_file(self):
        problem_recorder_filename = 'problem_history.db'
        prob = self.setup_sellar_problem(problem_recorder_filename=problem_recorder_filename)
        prob.setup(check=False, mode='rev')
        prob.run_driver()
        
        # just see if it has an exception
        view_cases(prob.get_outputs_dir() / problem_recorder_filename, show=False)

    @require_pyoptsparse('SLSQP')
    def test_viewing_driver_case_recorder_file(self):
        driver_recorder_filename = 'driver_history.db'
        prob = self.setup_sellar_problem(driver_recorder_filename=driver_recorder_filename)
        prob.setup(check=False, mode='rev')
        prob.run_driver()
        
        # just see if it has an exception
        view_cases(prob.get_outputs_dir() / driver_recorder_filename, show=False)


if __name__ == '__main__':
    unittest.main()
