"""
Functions for making assertions about OpenMDAO Systems.
"""
import numpy as np

from openmdao.core.component import Component
from openmdao.core.group import Group
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian

from numpy.testing import assert_allclose


def assert_check_partials(data, atol=1e-6, rtol=1e-6):
    """
    Raise assertion if any entry from the return from check_partials is above a tolerance.

    Parameters
    ----------
    data : dict of dicts of dicts
            First key:
                is the component name;
            Second key:
                is the (output, input) tuple of strings;
            Third key:
                is one of ['rel error', 'abs error', 'magnitude', 'J_fd', 'J_fwd', 'J_rev'];

            For 'rel error', 'abs error', 'magnitude' the value is: A tuple containing norms for
                forward - fd, adjoint - fd, forward - adjoint.
            For 'J_fd', 'J_fwd', 'J_rev' the value is: A numpy array representing the computed
                Jacobian for the three different methods of computation.
    atol : float
        absolute error. Default is 1e-6.
    rtol : float
        relative error. Default is 1e-6.
    """
    desired = (0.0, 0.0, 0.0)
    for comp in data:
        for (var, wrt) in data[comp]:
            for error_type, tolerance in [('abs error', atol), ('rel error', rtol), ]:
                actual = data[comp][var, wrt][error_type]
                if not np.isnan(actual).any():
                    assert_allclose(actual, desired, atol=tolerance,
                                    err_msg='{0} error in partial of'
                                            ' {1} wrt {2} in component {3}'.format(error_type,
                                                                                   var, wrt, comp),
                                    verbose=True)


def assert_no_approx_partials(system, include_self=True, recurse=True):
    """
    Raise assertion error if any component within system is using approximated partials.

    Parameters
    ----------
    system : System
        The system under which to search for approximated partials.
    include_self : bool
        If True, include this system in the iteration.
    recurse : bool
        If True, iterate over the whole tree under this system.

    Raises
    ------
    AssertionError
        If a subsystem of group is found to be using approximated partials.

    """
    approximated_partials = {}
    msg = 'The following components use approximated partials:\n'
    for s in system.system_iter(include_self=include_self, recurse=recurse):
        if isinstance(s, Component):
            if s._approximated_partials:
                approximated_partials[s.pathname] = s._approximated_partials
                msg += '    ' + s.pathname + '\n'
                for partial in s._approximated_partials:
                    msg += '        of={0:12s}    wrt={1:12s}    method={2:2s}\n'.format(*partial)
    if approximated_partials:
        raise AssertionError(msg)


def assert_no_dict_jacobians(system, include_self=True, recurse=True):
    """
    Raise an assertion error if any Group within system is found to be using dictionary jacobians.

    Parameters
    ----------
    system : System
        The system under which to search for approximated partials.
    include_self : bool
        If True, include this system in the iteration.
    recurse : bool
        If True, iterate over the whole tree under this system.

    Raises
    ------
    AssertionError
        If a subsystem of group is found to be using approximated partials.

    """
    groups_with_dict_jacobians = []
    msg = 'The following groups use dictionary jacobians:\n'
    for s in system.system_iter(include_self=include_self, recurse=recurse, typ=Group):
        if isinstance(s.jacobian, DictionaryJacobian):
            groups_with_dict_jacobians.append(s.pathname)
    msg = msg + '\n'.join(['    ' + s for s in groups_with_dict_jacobians])
    if groups_with_dict_jacobians:
        raise AssertionError(msg)
