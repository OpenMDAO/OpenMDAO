"""Temporary run file for the test components."""
from __future__ import division, print_function
import numpy

from six import iteritems
from six.moves import range
from collections import OrderedDict

import itertools
import unittest

from openmdao.test_suite.components.implicit_components \
    import TestImplCompNondLinear
from openmdao.test_suite.components.explicit_components \
    import TestExplCompNondLinear
from openmdao.test_suite.groups.group import TestGroupFlat
from openmdao.api import Problem
from openmdao.api import DefaultVector, NewtonSolver, ScipyIterativeSolver
from openmdao.api import GlobalJacobian, DenseMatrix, CooMatrix
from openmdao.parallel_api import PETScVector


class CompTestCase(unittest.TestCase):

    def test_comps(self):
        for key in itertools.product(
                [TestImplCompNondLinear, TestExplCompNondLinear],
                [DefaultVector, PETScVector],
                [DenseMatrix, CooMatrix],
                ['implicit', 'explicit'],
                ['matvec', 'dense'],
                range(1, 3),
                range(1, 3),
                [(1,), (2,), (2, 1), (1, 2)],
                ):
            Component = key[0]
            Vector = key[1]
            Matrix = key[2]
            connection_type = key[3]
            derivatives = key[4]
            num_var = key[5]
            num_sub = key[6]
            var_shape = key[7]

            print_str = ('%s %s %s %s %s %i-vars %i-comps %s' % (
                Component.__name__,
                Vector.__name__,
                Matrix.__name__,
                connection_type,
                derivatives,
                num_var, num_sub,
                str(var_shape),
            ))

            # print(print_str)

            group = TestGroupFlat(num_sub=num_sub, num_var=num_var,
                                  var_shape=var_shape,
                                  connection_type=connection_type,
                                  derivatives=derivatives,
                                  Component=Component,
                                  )
            prob = Problem(group).setup(Vector)

            if derivatives == 'dense':
                prob.root.jacobian = GlobalJacobian(Matrix=Matrix)

            prob.root.nl_solver = NewtonSolver(
                subsolvers={'linear': ScipyIterativeSolver(
                    maxiter=100,
                )}
            )
            prob.root.ln_solver = ScipyIterativeSolver(
                maxiter=200, atol=1e-10, rtol=1e-10)
            prob.root.suppress_solver_output = True

            fail, rele, abse = prob.run()
            if fail:
                self.fail('re %f ; ae %f ;  ' % (rele, abse) + print_str)

            # Setup for the 4 tests that follow
            size = numpy.prod(var_shape)
            work = prob.root._vectors['output']['']._clone()
            work.set_const(1.0)
            if Component == TestImplCompNondLinear:
                val = 1 - 0.01 + 0.01 * size * num_var * num_sub
            if Component == TestExplCompNondLinear:
                val = 1 - 0.01 * size * num_var * (num_sub - 1)

            # 1. fwd apply_linear test
            prob.root._vectors['output'][''].set_const(1.0)
            prob.root._apply_linear([''], 'fwd')
            prob.root._vectors['residual'][''].add_scal_vec(-val, work)
            self.assertAlmostEqual(
                prob.root._vectors['residual'][''].get_norm(), 0)

            # 2. rev apply_linear test
            prob.root._vectors['residual'][''].set_const(1.0)
            prob.root._apply_linear([''], 'rev')
            prob.root._vectors['output'][''].add_scal_vec(-val, work)
            self.assertAlmostEqual(
                prob.root._vectors['output'][''].get_norm(), 0)

            # 3. fwd solve_linear test
            prob.root._vectors['output'][''].set_const(0.0)
            prob.root._vectors['residual'][''].set_const(val)
            prob.root._solve_linear([''], 'fwd')
            prob.root._vectors['output'][''] -= work
            self.assertAlmostEqual(
                prob.root._vectors['output'][''].get_norm(), 0, delta=1e-2)

            # 4. rev solve_linear test
            prob.root._vectors['residual'][''].set_const(0.0)
            prob.root._vectors['output'][''].set_const(val)
            prob.root._solve_linear([''], 'rev')
            prob.root._vectors['residual'][''] -= work
            self.assertAlmostEqual(
                prob.root._vectors['residual'][''].get_norm(), 0, delta=1e-2)




if __name__ == '__main__':
    unittest.main()
