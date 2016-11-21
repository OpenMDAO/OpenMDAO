
from __future__ import division, print_function

import unittest

from openmdao.test_suite.update_test_general import CompTestCaseBase
from openmdao.test_suite.components.implicit_components     import TestImplCompNondLinear
from openmdao.test_suite.components.explicit_components     import TestExplCompNondLinear
from openmdao.api import DefaultVector, NewtonSolver, ScipyIterativeSolver
from openmdao.parallel_api import PETScVector

class CompTestCase(CompTestCaseBase):

    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_dense_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_dense_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c1_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c1_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c1_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c1_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c2_s1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c2_s2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c2_s2x1(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_TestImplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c2_s1x2(self):
        self.run_test(TestImplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_matvec_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_dense_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_implicit_sparse_coo_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_matvec_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_dense_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_DefaultVector_explicit_sparse_coo_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, DefaultVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_matvec_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_dense_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_implicit_sparse_coo_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'implicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_matvec_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'matvec', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_dense_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'dense', 'aij', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_array_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'array', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_sparse_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'sparse', 2, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v1_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 1, 2, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c1_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c1_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c1_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c1_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 1, (1, 2))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c2_s1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c2_s2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2,))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c2_s2x1(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (2, 1))


    def test_TestExplCompNondLinear_PETScVector_explicit_sparse_coo_aij_v2_c2_s1x2(self):
        self.run_test(TestExplCompNondLinear, PETScVector, 'explicit', 'sparse-coo', 'aij', 2, 2, (1, 2))


if __name__ == '__main__':
    unittest.main()
