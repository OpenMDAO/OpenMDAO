"""This generates a module, test_general.py that runs all of the specified
test permutations as separate tests so they can run concurrently.
"""

from __future__ import division, print_function

import os
import itertools
import unittest
from collections import OrderedDict

from six import iteritems
from six.moves import range
import numpy

from openmdao.test_suite.components.implicit_components \
    import TestImplCompNondLinear
from openmdao.test_suite.components.explicit_components \
    import TestExplCompNondLinear
from openmdao.test_suite.groups.group import TestGroupFlat
from openmdao.api import Problem
from openmdao.api import DefaultVector, NewtonSolver, ScipyIterativeSolver
from openmdao.api import GlobalJacobian, DenseMatrix, CooMatrix
from openmdao.parallel_api import PETScVector


class CompTestCaseBase(unittest.TestCase):
    """The TestCase that actually runs all of the cases inherits from this."""
    def run_test(self, Component, Vector, connection_type, jacobian_type,
                 partial_type, num_var, num_comp, var_shape):

        print_str = ('%s %s %s %s %s %i-vars %i-comps %s' % (
            Component.__name__,
            Vector.__name__,
            connection_type,
            jacobian_type,
            partial_type,
            num_var, num_comp,
            str(var_shape),
        ))

        # print(print_str)

        group = TestGroupFlat(num_comp=num_comp, num_var=num_var,
                              var_shape=var_shape,
                              connection_type=connection_type,
                              jacobian_type=jacobian_type,
                              partial_type=partial_type,
                              Component=Component,
                              )
        prob = Problem(group).setup(Vector)

        if jacobian_type == 'dense':
            prob.root.jacobian = GlobalJacobian(Matrix=DenseMatrix)
        elif jacobian_type == 'sparse-coo':
            prob.root.jacobian = GlobalJacobian(Matrix=CooMatrix)

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
            val = 1 - 0.01 + 0.01 * size * num_var * num_comp
        if Component == TestExplCompNondLinear:
            val = 1 - 0.01 * size * num_var * (num_comp - 1)

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

#  this template is used to generate each test in the generated TestCase.
tst_template = """
    def test_{0}(self):
        self.run_test({1}, {2}, '{3}', '{4}', '{5}', {6}, {7}, {8})
"""

#  this is the template for the entire generated test module.
tstmod_template = """
from __future__ import division, print_function

import unittest

from openmdao.test_suite.update_test_general import CompTestCaseBase
from openmdao.test_suite.components.implicit_components \
    import TestImplCompNondLinear
from openmdao.test_suite.components.explicit_components \
    import TestExplCompNondLinear
from openmdao.api import DefaultVector, NewtonSolver, ScipyIterativeSolver
from openmdao.parallel_api import PETScVector

class CompTestCase(CompTestCaseBase):
{0}

if __name__ == '__main__':
    unittest.main()
"""

def tst_generator():
    """Specify all of your tests here."""
    counter = 0
    for key in itertools.product(
            [TestImplCompNondLinear, TestExplCompNondLinear],
            [DefaultVector, PETScVector],
            ['implicit', 'explicit'],
            ['matvec', 'dense', 'sparse-coo'],
            ['array', 'sparse', 'aij'],
            range(1, 3),
            range(1, 3),
            [(1,), (2,), (2, 1), (1, 2)],
            ):
        yield tst_template.format(counter, key[0].__name__, key[1].__name__,
                                    key[2], key[3], key[4], key[5],
                                    key[6], key[7])
        counter += 1

def generate_test():
    """Replace test_general.py with a new version based on the tests specified
    in tst_generator.
    """
    mydir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(mydir, 'test_general.py')
    with open(fname, 'w') as f:
        test_lines = [t for t in tst_generator()]
        modstr = tstmod_template.format('\n'.join(test_lines))
        f.write(modstr)


if __name__ == '__main__':
    generate_test()
