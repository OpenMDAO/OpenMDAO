from __future__ import division, print_function

import itertools
from six.moves import range, zip
from six import PY3, iteritems, iterkeys, itervalues, string_types
import collections

from openmdao.test_suite.groups.cycle_group import CycleGroup
from openmdao.api import Problem
from openmdao.api import DefaultVector, NewtonSolver, ScipyIterativeSolver
from openmdao.api import GlobalJacobian, DenseMatrix, CooMatrix, CsrMatrix
from openmdao.devtools.testutil import assert_rel_error

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


CYCLE_PARAMS = {
    'component_class': ['explicit'],
    'vector_class': [DefaultVector, PETScVector] if PETScVector else [DefaultVector],
    'connection_type': ['implicit', 'explicit'],
    'global_jac': [True, False],
    'jacobian_type': ['matvec', 'dense', 'sparse-coo', 'sparse-csr'],
    'partial_type': ['array', 'sparse', 'aij'],
    'num_comp': [3, 2],
}

GROUP_PARAMS = {
    'cycle': CYCLE_PARAMS,
}

GROUP_CONSTRUCTORS = {
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
                    opts[arg] = (iter_val,)
                    break

        if kwargs:
            raise ValueError('Unknown options given: {0}'.format(_nice_name(kwargs)))

        for options in _cartesian_dict_product(opts):
            yield (ParameterizedInstance(group_type, **options),)


def _cartesian_dict_product(dicts):
    # From http://stackoverflow.com/a/5228294
    return (dict(zip(dicts, x)) for x in itertools.product(*itervalues(dicts)))


def full_test_suite():
    for group_type, params in iteritems(GROUP_PARAMS):
        for test_problem in test_suite(group_type=group_type, **params):
            yield test_problem


# Needed for nose
full_test_suite.__test__ = False
test_suite.__test__ = False


class ParameterizedInstance(object):
    def __init__(self, group_type, **kwargs):

        self._group_type = group_type
        self.args = kwargs.copy()

        self.name = '_'.join(
            '{0}_{1}'.format(key, _nice_name(value)) for key, value in iteritems(self.args)
        )

        self.problem = None
        self.expected_d_input = None
        self.expected_d_output = None
        self.value = 0

        self.solver_class = NewtonSolver
        self.solver_options = {
            'maxiter': 100
        }

        self.linear_solver_class = ScipyIterativeSolver
        self.linear_solver_options = {'maxiter': 200,
                                      'atol': 1e-10,
                                      'rtol': 1e-10,
                                      }

    def setup(self):
        args = self.args

        group = GROUP_CONSTRUCTORS[self._group_type](**args)
        self.problem = prob = Problem(group).setup(args['vector_class'], check=False)

        if args['global_jac']:
            jacobian_type = args['jacobian_type']
            if jacobian_type == 'dense':
                prob.model.jacobian = GlobalJacobian(matrix_class=DenseMatrix)
            elif jacobian_type == 'sparse-coo':
                prob.model.jacobian = GlobalJacobian(matrix_class=CooMatrix)
            elif jacobian_type == 'sparse-csr':
                prob.model.jacobian = GlobalJacobian(matrix_class=CsrMatrix)

        prob.model.ln_solver = self.linear_solver_class(**self.linear_solver_options)

        prob.model.nl_solver = self.solver_class(**self.solver_options)

        prob.model.suppress_solver_output = True

        fail, rele, abse = prob.run_model()
        if fail:
            raise RuntimeError('Problem run failed: re %f ; ae %f' % (rele, abse))

    def compute_totals(self, mode='fwd'):
        problem = self.problem

        if problem._mode != mode:
            problem.setup(check=False, mode=mode)
            fail, rele, abse = problem.run_model()
            if fail:
                raise RuntimeError('Problem run failed: re %f ; ae %f' % (rele, abse))

        root = problem.model
        of = root.total_of
        wrt = root.total_wrt

        totals = self.problem.compute_total_derivs(of, wrt)
        return totals
