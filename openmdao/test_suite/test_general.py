from __future__ import division, print_function

import itertools
import unittest
from six.moves import range
from six import PY3, iteritems
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
from collections import OrderedDict
from openmdao.devtools.testutil import assert_rel_error

TEST_PARAMS = (
    [TestImplCompNondLinear, TestExplCompNondLinear],  # component_class
    [DefaultVector, PETScVector] if PETScVector else [DefaultVector],  # vector_class
    ['implicit', 'explicit'],  # connection_type
    [True, False],  # global_jac
    ['matvec', 'dense', 'sparse-coo', 'sparse-csr'],  # jacobian_type
    ['array', 'sparse', 'aij'],  # partial_type
    range(1, 3),  # num_var
    range(1, 3),  # num_comp
    [(1,), (2,), (2, 1), (1, 2)],  # var_shape
)


def _nice_name(obj):
    if isinstance(obj, type):
        return obj.__name__
    elif isinstance(obj, dict):
        return str({_nice_name(k): _nice_name(v) for k, v in iteritems(obj)})
    return str(obj)


class GeneralProblem(object):
    def __init__(self, component_class, vector_class, connection_type, global_jac, jacobian_type,
                 partial_type, num_var, num_comp, var_shape):

        self.args = OrderedDict((
            ('component_class', component_class),
            ('vector_class', vector_class),
            ('connection_type', connection_type),
            ('global_jac', global_jac),
            ('jacobian_type', jacobian_type),
            ('partial_type', partial_type),
            ('num_var', num_var),
            ('num_comp', num_comp),
            ('var_shape', var_shape),
        ))

        self.name = '_'.join(
            '{0}_{1}'.format(key, _nice_name(value)) for key, value in iteritems(self.args)
        )

        self._run = False
        self._setup = False
        self._linearized = False
        self.problem = None
        self.expected_d_input = None
        self.expected_d_output = None
        self.value = 0

        self.solver_class = NewtonSolver
        self.solver_options = {'subsolvers':{'linear': ScipyIterativeSolver(
                maxiter=100,
            )}}

        self.linear_solver_class = ScipyIterativeSolver
        self.linear_solver_options = {'maxiter': 200,
                                      'atol': 1e-10,
                                      'rtol': 1e-10,
                                      }

    def setup(self):
        self._setup = True
        args = self.args
        group = TestGroupFlat(num_comp=args['num_comp'],
                              num_var=args['num_var'],
                              var_shape=args['var_shape'],
                              connection_type=args['connection_type'],
                              jacobian_type=args['jacobian_type'],
                              partial_type=args['partial_type'],
                              component_class=args['component_class'],
                              )

        self.problem = prob = Problem(group).setup(args['vector_class'], check=False)

        if args['global_jac']:
            jacobian_type = args['jacobian_type']
            if jacobian_type == 'dense':
                prob.root.jacobian = GlobalJacobian(matrix_class=DenseMatrix)
            elif jacobian_type == 'sparse-coo':
                prob.root.jacobian = GlobalJacobian(matrix_class=CooMatrix)
            elif jacobian_type == 'sparse-csr':
                prob.root.jacobian = GlobalJacobian(matrix_class=CsrMatrix)

        prob.root.ln_solver = self.linear_solver_class(**self.linear_solver_options)

        prob.root.nl_solver = self.solver_class(**self.solver_options)

        prob.root.suppress_solver_output = True

        self._run = False

        size = numpy.prod(args['var_shape'])
        self.expected_d_input = prob.root._vectors['output']['linear']._clone(initialize_views=True)
        self.expected_d_output = self.expected_d_input._clone(initialize_views=True)

        n = args['num_var']
        m = args['num_comp'] - 1
        d_value = 0.01 * size * (n * (n + 1)) / 2 * m
        if args['component_class'] == TestImplCompNondLinear:
            self.value = 1 + d_value
        elif args['component_class'] == TestExplCompNondLinear:
            self.value = 1 - d_value
        else:
            raise NotImplementedError()

        for name in self.expected_d_input:
            output_num = int(name.split('_')[-1])
            self.expected_d_input[name][:] = output_num + 1
            self.expected_d_output[name][:] = output_num + self.value

    def run(self):
        if not self._setup:
            self.setup()
        self._run = True
        return self.problem.run()

    def apply_linear_test(self, mode='fwd'):
        root = self.problem.root
        if not self._linearized:
            root._apply_nonlinear()
            root._linearize()
            self._linearized = True
        if mode == 'fwd':
            in_ = 'output'
            out = 'residual'
        elif mode == 'rev':
            in_ = 'residual'
            out = 'output'
        else:
            raise NotImplementedError('Mode must be "fwd" or "rev"')

        root._vectors[in_]['linear'].set_const(1.0)
        root._apply_linear(['linear'], mode)
        root._vectors[out]['linear'].add_scal_vec(-self.value, self.expected_d_input)
        return root._vectors[out]['linear'].get_norm()

    def solve_linear_test(self, input=None, mode='fwd'):
        root = self.problem.root
        if input is None:
            input = self.expected_d_output
        if not self._linearized:
            root._apply_nonlinear()
            root._linearize()
            self._linearized = True
        if mode == 'rev':
            in_ = 'output'
            out = 'residual'
        elif mode == 'fwd':
            in_ = 'residual'
            out = 'output'
        else:
            raise NotImplementedError('Mode must be "fwd" or "rev"')

        root._vectors[out]['linear'].set_const(0.0)
        root._vectors[in_]['linear'] = input
        root._solve_linear(['linear'], mode)
        return root._vectors[out]['linear']._data


def full_test_suite():
    for args in itertools.product(*TEST_PARAMS):
        yield (GeneralProblem(*args),)

# Needed for Nose
full_test_suite.__test__ = False


def _test_name(testcase_func, param_num, params):
    return '_'.join(('test', params.args[0].name))


class ParameterizedTestCases(unittest.TestCase):
    """The TestCase that actually runs all of the cases inherits from this."""

    @parameterized.expand(full_test_suite(),
                          testcase_func_name=_test_name)
    def test_openmdao(self, test):

        fail, rele, abse = test.run()
        if fail:
            self.fail('Problem run failed: re %f ; ae %f' % (rele, abse))

        assert_rel_error(self, test.solve_linear_test(mode='fwd'), test.expected_d_input._data, 1e-15)
        assert_rel_error(self, test.solve_linear_test(mode='rev'), test.expected_d_input._data, 1e-15)
        assert_rel_error(self, test.apply_linear_test(mode='fwd'), 0., 1e-15)
        assert_rel_error(self, test.apply_linear_test(mode='rev'), 0., 1e-15)
