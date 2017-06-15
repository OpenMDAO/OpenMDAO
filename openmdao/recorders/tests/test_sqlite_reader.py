""" Unit tests for the SqliteCaseReader. """
from __future__ import print_function

import errno
import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from openmdao.test_suite.components.sellar import SellarDerivatives


from openmdao.core.problem import Problem
from openmdao.recorders.sqlite_recorder import SqliteRecorder, format_version
from openmdao.recorders.case_reader import CaseReader

from openmdao.recorders.sqlite_reader import SqliteCaseReader

try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    pyOptSparseDriver = None

# Test that pyoptsparse SLSQP is a viable option
try:
    import pyoptsparse.pySLSQP.slsqp as slsqp
except ImportError:
    slsqp = None


optimizers = {'pyoptsparse': pyOptSparseDriver}
# optimizers = {'scipy': ScipyOptimizer,
#               'pyoptsparse': pyOptSparseDriver}




class TestSqliteCaseReader(unittest.TestCase):

    def setup_sellar_model(self):
        self.prob = Problem()
        self.prob.model = model = SellarDerivatives()

        optimizer = 'pyoptsparse'
        self.prob.driver = optimizers[optimizer]()

        self.prob.model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                   upper=np.array([10.0, 10.0]))
        self.prob.model.add_design_var('x', lower=0.0, upper=10.0)
        self.prob.model.add_objective('obj')
        self.prob.model.add_constraint('con1', upper=0.0)
        self.prob.model.add_constraint('con2', upper=0.0)
        self.prob.model.suppress_solver_output = True

        self.prob.setup(check=False)


    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        print('self.filename', self.filename)
        self.recorder = SqliteRecorder(self.filename)
        self.original_path = os.getcwd()
        os.chdir(self.dir)

    def tearDown(self):
        os.chdir(self.original_path)


        return # TODO_RECORDERS - remove this

        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_format_version(self):

        self.setup_sellar_model()

        self.prob.run_driver()

        self.prob.cleanup()  # closes recorders TODO_RECORDER: need to implement a cleanup

        cr = CaseReader(self.filename)
        self.assertEqual(cr.format_version, format_version,
                         msg='format version not read correctly')

    def test_reader_instantiates(self):
        """ Test that CaseReader returns an HDF5CaseReader. """

        self.setup_sellar_model()

        self.prob.run_driver()

        self.prob.cleanup()  # closes recorders TODO_RECORDER: need to implement a cleanup

        cr = CaseReader(self.filename)
        self.assertTrue(isinstance(cr, SqliteCaseReader), msg='CaseReader not'
                        ' returning the correct subclass.')

    def qqq_test_basic_sellar(self):
        """ Tests that the reader returns params correctly. """

        self.setup_sellar_model()

        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = True
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = True
        self.prob.driver.add_recorder(self.recorder)

        self.prob.run_driver()

        self.prob.cleanup()  # closes recorders TODO_RECORDER: need to implement a cleanup

        cr = CaseReader(self.filename)
        last_case = cr.get_case(-1)

        np.testing.assert_almost_equal(last_case.desvars['pz.z'], [ 1.9776389,  0.],
                              err_msg='Case reader gives '
                                  'incorrect Parameter value'
                                  ' for {0}'.format('pz.z'))
        np.testing.assert_almost_equal(last_case.desvars['px.x'], [ 0.0,],
                              err_msg='Case reader gives '
                                  'incorrect Parameter value'
                                  ' for {0}'.format('px.x'))

        # assert_rel_error(self, top['obj'], 3.1833940, 1e-5)



        print('last_case', last_case)
        last_case_id = cr.list_cases()[-1]
        n = cr.num_cases

        self.assertEqual(cr.num_cases, 6)

        print('num cases', n)




    def qqqtest_ConvergeDiverge(self):
        """ Tests that the reader returns params correctly. """
        cr = CaseReader(self.filename)
        last_case = cr.get_case(-1)

        print('last_case.desvars', last_case.desvars['pz.z'])
        np.testing.assert_almost_equal(last_case.desvars['pz.z'], [ 5.,  2.],
                              err_msg='Case reader gives '
                                  'incorrect Parameter value'
                                  ' for {0}'.format('pz.z'))

        print('last_case', last_case)
        last_case_id = cr.list_cases()[-1]
        n = cr.num_cases

        print('num cases', n)
        # with SqliteDict(self.filename, 'iterations', flag='r') as db:
        #     for key in db[last_case_id]['Parameters'].keys():
        #         val = db[last_case_id]['Parameters'][key]
        #         np.testing.assert_almost_equal(last_case.parameters[key], val,
        #                                        err_msg='Case reader gives '
        #                                            'incorrect Parameter value'
        #                                            ' for {0}'.format(key))





    # def test_root_derivs_array(self):
    #     prob = Problem()
    #     prob.root = SellarDerivativesGrouped()

    #     prob.driver = ScipyOptimizer()
    #     prob.driver.options['optimizer'] = 'SLSQP'
    #     prob.driver.options['tol'] = 1.0e-8
    #     prob.driver.options['disp'] = False

    #     prob.driver.add_desvar('z', lower=np.array([-10.0, 0.0]),
    #                          upper=np.array([10.0, 10.0]))
    #     prob.driver.add_desvar('x', lower=0.0, upper=10.0)

    #     prob.driver.add_objective('obj')
    #     prob.driver.add_constraint('con1', upper=0.0)
    #     prob.driver.add_constraint('con2', upper=0.0)

    #     prob.driver.add_recorder(self.recorder)
    #     self.recorder.options['record_metadata'] = False
    #     self.recorder.options['record_derivs'] = True
    #     prob.setup(check=False)

    #     prob.run()

    #     prob.cleanup()

    #     db = SqliteDict(self.filename, self.tablename_derivs, flag='r')
    #     J1 = db['rank0:SLSQP|1']['Derivatives']

    #     assert_rel_error(self, J1[0][0], 9.61001155, .00001)
    #     assert_rel_error(self, J1[0][1], 1.78448534, .00001)
    #     assert_rel_error(self, J1[0][2], 2.98061392, .00001)
    #     assert_rel_error(self, J1[1][0], -9.61002285, .00001)
    #     assert_rel_error(self, J1[1][1], -0.78449158, .00001)
    #     assert_rel_error(self, J1[1][2], -0.98061433, .00001)
    #     assert_rel_error(self, J1[2][0], 1.94989079, .00001)
    #     assert_rel_error(self, J1[2][1], 1.0775421, .00001)
    #     assert_rel_error(self, J1[2][2], 0.09692762, .00001)




    # def test_only_resids_recorded(self):
    #     prob = Problem()
    #     prob.root = ConvergeDiverge()
    #     prob.driver.add_recorder(self.recorder)
    #     self.recorder.options['record_params'] = False
    #     self.recorder.options['record_unknowns'] = False
    #     self.recorder.options['record_resids'] = True
    #     prob.setup(check=False)

    #     t0, t1 = run_problem(prob)
    #     prob.cleanup()  # closes recorders

    #     coordinate = [0, 'Driver', (1, )]

    #     expected_resids = [
    #         ("comp1.y1", 0.0),
    #         ("comp1.y2", 0.0),
    #         ("comp2.y1", 0.0),
    #         ("comp3.y1", 0.0),
    #         ("comp4.y1", 0.0),
    #         ("comp4.y2", 0.0),
    #         ("comp5.y1", 0.0),
    #         ("comp6.y1", 0.0),
    #         ("comp7.y1", 0.0),
    #         ("p.x", 0.0)
    #     ]

    #     self.assertIterationDataRecorded(((coordinate, (t0, t1), None, None, expected_resids),), self.eps)





# @unittest.skipIf(pyOptSparseDriver is None, 'pyOptSparse not available.')
# @unittest.skipIf(slsqp is None, 'pyOptSparse SLSQP not available.')
# class TestSqliteCaseReaderPyOptSparse(TestSqliteCaseReader):

#     def setUp(self):
#         _setup_test_case(self, record_params=True, record_metadata=True,
#                          record_derivs=True, record_resids=True,
#                          record_unknowns=True, optimizer='pyoptsparse')


if __name__ == "__main__":
    unittest.main()
