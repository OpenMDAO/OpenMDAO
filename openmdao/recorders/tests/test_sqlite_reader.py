""" Unit tests for the SqliteCaseReader. """
from __future__ import print_function

import errno
import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from sqlitedict import SqliteDict

# from openmdao.api import Problem, ScipyOptimizer, Group, \
#     IndepVarComp, CaseReader
# from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives


from openmdao.api import Problem, Group, IndepVarComp
from openmdao.recorders.sqlite_recorder import SqliteRecorder, format_version
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.case import Case
from openmdao.recorders.case_reader import CaseReader
from openmdao.recorders.base_case_reader import BaseCaseReader

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


def _setup_test_case(case, record_inputs=True, record_params=True,
                     record_metadata=True, optimizer='pyoptsparse'):

    case.dir = mkdtemp()
    case.filename = os.path.join(case.dir, "sqlite_test")
    case.recorder = SqliteRecorder(case.filename)

    prob = Problem()
    prob.model = model = SellarDerivatives()
    prob.driver.add_recorder(case.recorder)

    # prob.driver = optimizers[optimizer]()

    case.recorder.options['record_desvars'] = True
    case.recorder.options['record_responses'] = True
    case.recorder.options['record_objectives'] = True
    case.recorder.options['record_constraints'] = True

    model.add_design_var('z')
    model.add_objective('obj')
    model.add_constraint('con1', lower=0)
    model.suppress_solver_output = True

    prob.setup(check=False)
 
    case.original_path = os.getcwd()
    os.chdir(case.dir)

    prob.run_driver()

    prob.cleanup()  # closes recorders TODO_RECORDER: need to implement a cleanup

class TestSqliteCaseReader(unittest.TestCase):

    def setUp(self):
        _setup_test_case(self, record_inputs=True,record_params=True, record_metadata=True,
                         optimizer='scipy')

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
        print('self.filename', self.filename)
        cr = CaseReader(self.filename)
        self.assertEqual(cr.format_version, format_version,
                         msg='format version not read correctly')

    def test_reader_instantiates(self):
        """ Test that CaseReader returns an HDF5CaseReader. """
        cr = CaseReader(self.filename)
        self.assertTrue(isinstance(cr, SqliteCaseReader), msg='CaseReader not'
                        ' returning the correct subclass.')

    def test_params(self):
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
        # with SqliteDict(self.filename, 'iterations', flag='r') as db:
        #     for key in db[last_case_id]['Parameters'].keys():
        #         val = db[last_case_id]['Parameters'][key]
        #         np.testing.assert_almost_equal(last_case.parameters[key], val,
        #                                        err_msg='Case reader gives '
        #                                            'incorrect Parameter value'
        #                                            ' for {0}'.format(key))

# @unittest.skipIf(pyOptSparseDriver is None, 'pyOptSparse not available.')
# @unittest.skipIf(slsqp is None, 'pyOptSparse SLSQP not available.')
# class TestSqliteCaseReaderPyOptSparse(TestSqliteCaseReader):

#     def setUp(self):
#         _setup_test_case(self, record_params=True, record_metadata=True,
#                          record_derivs=True, record_resids=True,
#                          record_unknowns=True, optimizer='pyoptsparse')


if __name__ == "__main__":
    unittest.main()
