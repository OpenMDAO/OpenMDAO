from __future__ import division, print_function

import itertools
from six.moves import zip
from six import iteritems, iterkeys, itervalues, string_types
import collections

from parameterized import parameterized
from unittest import SkipTest

from openmdao.core.problem import Problem
from openmdao.solvers.linear.scipy_iter_solver import ScipyKrylov
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.test_suite.groups.cycle_group import CycleGroup
from openmdao.vectors.default_vector import DefaultVector

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

MODELS = {
    'cycle': CycleGroup,
}


def _nice_name(obj):
    if isinstance(obj, type):
        return obj.__name__
    elif isinstance(obj, dict):
        return str({_nice_name(k): _nice_name(v) for k, v in iteritems(obj)})
    return str(obj)


def _test_suite(*args, **kwargs):
    """
    Generator for the parametric tests. If args is present, must only be the value '*',
    indicating running all available groups/parameters. Otherwise, use kwargs to set the options
    like so:
        arg=value will specify that option,
        arg='*' will vary over all default options,
        arg=(iterable) will iterate over the given options.
    Arguments that are not specified will have a reasonable default chosen.
    """
    full_suite = args and args[0] == '*'
    groups = kwargs.pop('group_type', iterkeys(MODELS))

    if isinstance(groups, string_types):
        groups = (groups, )

    for group_type in groups:
        opts = {}
        default_params = MODELS[group_type]().default_params

        if full_suite:
            opts.update(default_params)
            if kwargs:
                raise ValueError('Cannot specify "*" and kwargs')
        else:
            for arg, default_val in iteritems(default_params):
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


def _test_name(run_by_default):
    if run_by_default:
        test_prefix = 'test'
    else:
        test_prefix = 'ptest'

    def namer(testcase_fun, param_num, params):
        return '_'.join([
            test_prefix,
            str(params.args[0]),
        ])
    return namer


def parametric_suite(*args, **kwargs):
    """
    Decorator used for testing a range of different options for a particular
    ParametericTestGroup. If args is present, must only be the value '*',
    indicating running all available groups/parameters. Otherwise, use kwargs to set the options like so:
    arg=value will specify that option,
    arg='*' will vary over all default options,
    arg=iterable will iterate over the given options.
    Arguments that are not specified will have a reasonable default chosen."""
    run_by_default = kwargs.pop('run_by_default', False)
    test_cases = _test_suite(*args, **kwargs)
    return parameterized.expand(test_cases, testcase_func_name=_test_name(run_by_default))

# Needed for Nose
parametric_suite.__test__ = False


class ParameterizedInstance(object):
    """
    Parameterized Instance for a particular ParametricTestGroup. Typically not instantiated
    directly, but rather through the @parametric_suite decorator.

    Attributes
    ----------
    args : dict
        Dictionary of kwargs used to construct the instance.
    name : str
        A "friendly" name of the instance constructed from the kwargs used.
    problem : `Problem`
        Containing Problem for the instance.
    solver_class : `Solver`
        Non-linear solver to be instantiated at the problem level.
    solver_options : dict
        Options to pass into the constructor for `solver_class`.
    linear_solver_class : `Solver`
        Linear solver to be instantiated at the problem level.
    linear_solver_options : dict
        Options to pass into the constructor for `linear_solver_class`.
    """
    def __init__(self, group_type, **kwargs):

        self._group_type = group_type
        self.args = kwargs.copy()

        self.name = '_'.join(
            '{0}_{1}'.format(key, _nice_name(value)) for key, value in iteritems(self.args)
        )

        self.problem = None
        self.solver_class = NewtonSolver
        self.solver_options = {
            'maxiter': 100
        }

        self.linear_solver_class = ScipyKrylov
        self.linear_solver_options = {'maxiter': 200,
                                      'atol': 1e-10,
                                      'rtol': 1e-10,
                                      'assembled_jac': None,
                                      }

    def setup(self, check=False):
        """
        Creates the containing `Problem` and performs needed initializations.

        Parameters
        ----------
        check : bool
            If setup should run checks.
        """
        args = self.args

        group = MODELS[self._group_type](**args)

        local_vec_class = args.get('local_vector_class', 'default')
        if local_vec_class == 'default':
            vec_class = DefaultVector
        elif local_vec_class == 'petsc':
            vec_class = PETScVector
            if PETScVector is None:
                raise SkipTest('PETSc not available.')
        else:
            raise RuntimeError("Unrecognized local_vector_class '%s'" % local_vec_class)

        self.problem = prob = Problem(group)

        if args['assembled_jac']:
            jacobian_type = args.get('jacobian_type', 'dense')

            if jacobian_type == 'dense':
                self.linear_solver_options['assembled_jac'] = 'dense'
            elif jacobian_type == 'sparse-csc':
                self.linear_solver_options['assembled_jac'] = 'csc'
            else:
                raise RuntimeError("Invalid assembled_jac: '%s'." % jacobian_type)

        prob.model.linear_solver = self.linear_solver_class(**self.linear_solver_options)

        prob.model.nonlinear_solver = self.solver_class(**self.solver_options)

        prob.set_solver_print(level=0)

        prob.setup(check=check, local_vector_class=vec_class)

        fail, rele, abse = prob.run_model()
        if fail:
            raise RuntimeError('Problem run failed: re %f ; ae %f' % (rele, abse))

    def compute_totals(self, mode='fwd'):
        """
        Computes the total derivatives across the model.

        Parameters
        ----------
        mode : str
            Which mode to use for computing the total derivatives. Must be understood by
            `Problem.setup()`.

        Returns
        -------
        dict mapping (out,in) pairs to their associated total derivative.
        """
        problem = self.problem

        if problem._mode != mode:
            problem.setup(check=False, mode=mode)
            fail, rele, abse = problem.run_model()
            if fail:
                raise RuntimeError('Problem run failed: re %f ; ae %f' % (rele, abse))

        root = problem.model
        of = root.total_of
        wrt = root.total_wrt

        totals = self.problem.compute_totals(of, wrt)
        return totals

    def __str__(self):
        return self.name
