from __future__ import division, print_function

import itertools
import unittest
from six.moves import range, zip
from six import PY3, iteritems, iterkeys, itervalues, string_types
import numpy
import collections

from openmdao.test_suite.groups.group import TestMeshGroup
from openmdao.test_suite.groups.cycle_group import CycleGroup
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

MESH_PARAMS = {
    'component_class': ['implicit', 'explicit'],
    'vector_class': [DefaultVector, PETScVector] if PETScVector else [DefaultVector],
    'connection_type': ['implicit', 'explicit'],
    'global_jac': [True, False],
    'jacobian_type': ['matvec', 'dense', 'sparse-coo', 'sparse-csr'],
    'partial_type': ['array', 'sparse', 'aij'],
    'num_var': [2, 1],
    'num_comp': [2, 1],
    'var_shape': [(1,), (2,), (2, 1), (1, 2)],
}

CYCLE_PARAMS = {
    'component_class': ['explicit'],
    'vector_class': [DefaultVector, PETScVector] if PETScVector else [DefaultVector],
    'connection_type': ['implicit', 'explicit'],
    'global_jac': [True, False],
    'jacobian_type': ['matvec', 'dense', 'sparse-coo', 'sparse-csr'],
    'partial_type': ['array', 'sparse', 'aij'],
    'num_comp': [2, 1],
}

GROUP_PARAMS = {
    'mesh': MESH_PARAMS,
    'cycle': CYCLE_PARAMS,
}

GROUP_CONSTRUCTORS = {
    'mesh': TestMeshGroup,
    'cycle': CycleGroup,
}

def _nice_name(obj):
    if isinstance(obj, type):
        return obj.__name__
    elif isinstance(obj, dict):
        return str({_nice_name(k): _nice_name(v) for k, v in iteritems(obj)})
    return str(obj)


def test_suite(**kwargs):
    groups = kwargs.pop('group_type', iterkeys(GROUP_PARAMS))

    if isinstance(groups, string_types):
        groups = (groups, )

    for group_type in groups:
        opts = {}
        default_opts = GROUP_PARAMS[group_type]

        for arg, default_val in iteritems(default_opts):
            if arg in kwargs:
                arg_value = kwargs.pop(arg)
                if arg_value == '*':
                    opts[arg] = default_val
                elif isinstance(arg_value, string_types) \
                        or not isinstance(arg_value, collections.Iterable):
                    # itertools.product expects iterables, so make 1-item tuple
                    opts[arg] = (arg_value,)
                else:
                    opts[arg] = arg_value
            else:
                # We're not asked to vary this parameter, so choose first item as default
                # Since we may use a generator (e.g. range), take the first value from the iterator
                # instead of indexing.
                for iter_val in default_val:
                    opts[arg] = iter_val
                    break

        if kwargs:
            raise ValueError('Unknown options given: {0}'.format(_nice_name(kwargs)))

        for options in _cartesian_dict_product(opts):
            yield (ParameterizedInstance(group_type, **options),)

def _cartesian_dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*itervalues(dicts)))

def full_test_suite():
    for group_type, params in iteritems(GROUP_PARAMS):
        for test_problem in test_suite(group_type=group_type, **params):
            yield test_problem

# Needed for Nose
full_test_suite.__test__ = False


def _test_name(testcase_func, param_num, params):
    return '_'.join(('test', params.args[0].name))


class ParameterizedInstance(object):
    def __init__(self, group_type, **kwargs):

        self._group_type = group_type
        self.args = kwargs.copy()

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

        group = GROUP_CONSTRUCTORS[self._group_type](**args)
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
        if args['component_class'] == 'implicit':
            self.value = 1 + d_value
        elif args['component_class'] == 'explicit':
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


class ParameterizedTestCases(unittest.TestCase):
    """The TestCase that actually runs all of the cases inherits from this."""

    # @parameterized.expand(full_test_suite(),
    #                       testcase_func_name=_test_name)
    @parameterized.expand(test_suite(group_type='mesh', **MESH_PARAMS),
                          testcase_func_name=_test_name)
    def test_openmdao(self, test):

        fail, rele, abse = test.run()
        if fail:
            self.fail('Problem run failed: re %f ; ae %f' % (rele, abse))

        assert_rel_error(self, test.solve_linear_test(mode='fwd'), test.expected_d_input._data, 1e-15)
        assert_rel_error(self, test.solve_linear_test(mode='rev'), test.expected_d_input._data, 1e-15)
        assert_rel_error(self, test.apply_linear_test(mode='fwd'), 0., 1e-15)
        assert_rel_error(self, test.apply_linear_test(mode='rev'), 0., 1e-15)
