from __future__ import division, print_function

import itertools
import unittest
from six.moves import range
import numpy

from openmdao.test_suite.components.implicit_components \
    import TestImplCompNondLinear
from openmdao.test_suite.components.explicit_components \
    import TestExplCompNondLinear
from openmdao.test_suite.groups.group import TestGroupFlat
from openmdao.api import Problem
from openmdao.api import DefaultVector, NewtonSolver, ScipyIterativeSolver
from openmdao.api import GlobalJacobian, DenseMatrix, CooMatrix, CsrMatrix

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from nose_parameterized import parameterized


def custom_name(testcase_func, param_num, param):
    return ''.join(('test_',
                    '_'.join(p.__name__ for p in param.args[:2]),
                    '_',
                    '_'.join(str(p) for p in param.args[2:]))
                   )


class CompTestCaseBase(unittest.TestCase):
    """The TestCase that actually runs all of the cases inherits from this."""

    @parameterized.expand(itertools.product(
        [TestImplCompNondLinear, TestExplCompNondLinear],
        [DefaultVector, PETScVector] if PETScVector else [DefaultVector],
        ['implicit', 'explicit'],
        [True, False],
        ['matvec', 'dense', 'sparse-coo', 'sparse-csr'],
        ['array', 'sparse', 'aij'],
        range(1, 3),
        range(1, 3),
        [(1,), (2,), (2, 1), (1, 2)],
    ), testcase_func_name=custom_name)
    def test_openmdao(self, component_class, vector_class, connection_type, global_jac, jacobian_type,
                      partial_type, num_var, num_comp, var_shape):

        group = TestGroupFlat(num_comp=num_comp, num_var=num_var,
                              var_shape=var_shape,
                              connection_type=connection_type,
                              jacobian_type=jacobian_type,
                              partial_type=partial_type,
                              component_class=component_class,
                              )
        prob = Problem(group).setup(vector_class, check=False)

        if global_jac:
            if jacobian_type == 'dense':
                prob.root.jacobian = GlobalJacobian(matrix_class=DenseMatrix)
            elif jacobian_type == 'sparse-coo':
                prob.root.jacobian = GlobalJacobian(matrix_class=CooMatrix)
            elif jacobian_type == 'sparse-csr':
                prob.root.jacobian = GlobalJacobian(matrix_class=CsrMatrix)

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
            self.fail('Problem run failed: re %f ; ae %f' % (rele, abse))

        # Setup for the 4 tests that follow
        size = numpy.prod(var_shape)
        work = prob.root._vectors['output']['']._clone()
        work.set_const(1.0)
        if component_class == TestImplCompNondLinear:
            val = 1 - 0.01 + 0.01 * size * num_var * num_comp
        elif component_class == TestExplCompNondLinear:
            val = 1 - 0.01 * size * num_var * (num_comp - 1)

        prob.root._apply_nonlinear()
        prob.root._linearize()

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
