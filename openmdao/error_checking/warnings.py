import inspect
import sys
import warnings


class OpenMDAOWarning(UserWarning):
    """
    Base class for all OpenMDAO warnings.
    """
    name = 'warn_all'

    def __init__(self, msg, prefix=''):
        """
        Parameters
        ----------
        msg : str
            The message to be printed with this warning/exception.
        prefix : str
            A prifix printed before this message, often used to identify the path of the system in which
            it was raised.
        """
        self._str = f'{prefix}: {msg}' if prefix else msg

    def __str__(self):
        return self._str


class NoMPIWarning(OpenMDAOWarning):
    """
    Warning dealing with the non availability of MPI.
    """
    name = 'warn_no_mpi'

class ColoringWarning(OpenMDAOWarning):
    """
    Warning dealing with derivative coloring.
    """
    name = 'warn_coloring'

class InefficiencyWarning(OpenMDAOWarning):
    """
    Warning about user options which may impact memory or execution time efficiency.
    """
    name = 'warn_inefficiency'

class NoDistributedVectorWarning(OpenMDAOWarning):
    """
    Warning pertaining to the non availability of a distributed vector class.
    """
    name = 'warn_no_distributed_vector'

class DistributedComponentWarning(OpenMDAOWarning):
    """
    Warning specific to a distributed component.
    """
    name = 'warn_distributed_component'

class UnusedOptionWarning(OpenMDAOWarning):
    """
    Warning dealing with an unnecessary option or argument being provided.
    """
    name = 'warn_unused_option'

class PromotionWarning(OpenMDAOWarning):
    """
    Warning dealing with the promotion of an input or output.
    """
    name = 'warn_promotion'

class InputDefaultsWarning(OpenMDAOWarning):
    """
    Warning dealing with the use of set_input_defaults.
    """
    name = 'warn_input_defaults'

class _AllowedForN2Warning(OpenMDAOWarning):
    """
    Warning which is normally error except when disregarded in order to build the N2 diagram.
    """
    name = '_warn_allowed_for_n2'

class KrigingCacheWarning(OpenMDAOWarning):
    """
    Warning which is issued when the KrigingSurrogate fails to load the kriging surrogate cache
    during training.
    """
    name = 'warn_kriging_cache'


class ConnectUnitsToUnitlessWarning(OpenMDAOWarning):
    """
    Warning which is issued when a unitless output is connected to an input with units, or vice versa.
    """
    name = 'warn_connect_unitless'


class InvalidCheckWarning(OpenMDAOWarning):
    """
    Warning which is issued when an invalid check is requested during setup.
    """
    name = 'warn_invalid_check'


class NoRecordedDataWarning(OpenMDAOWarning):
    """
    Warning which is issued when requested data cannot be found in a recording.
    """
    name = 'warn_no_recorded_data'


class SingularJacWarning(OpenMDAOWarning):
    """
    Warning which is issued when requested data cannot be found in a recording.
    """
    name = 'warn_singular_jac'


class ApproxPartialsWarning(OpenMDAOWarning):
    """
    Warning issued when the approximated partials cannot be evaluated as expected.
    """
    name = 'warn_approx_partials'


class NoDerivativesWarning(OpenMDAOWarning):
    """
    Warning issued when derivatives are expected but not found.
    """
    name = 'warn_no_derivatives'


class SolverWarning(OpenMDAOWarning):
    """
    Warning base class for solver-related warnings.
    """
    name = 'warn_solver'

class SolverUncoveredStatesWarning(SolverWarning):
    """
    Warning issued when states in an implicit system are not associated with any solver.
    """
    name = 'warn_solver_uncovered_states'


class SolverStallWarning(SolverWarning):
    """
    Warning issued when a solver stalls multiple times due to bounds enforcement.
    """
    name = 'warn_solver_stalled'


class ForceAllocComplexWarning(OpenMDAOWarning):
    """
    Warning issued when check_partials is called with complex-step, but force_alloc_complex=False.
    """
    name = 'warn_solver_stalled'


_warnings = {_class.name: _class for _, _class in
             inspect.getmembers(sys.modules[__name__], inspect.isclass)}


def apply_warning_filter(name, action):
    """
    Apply the given filter action to the OpenMDAO warning category with the given name.

    Parameters
    ----------
    name : str
        The name property of the OpenMDAO warning category to be filter.
    action : str
        A valid Python filter action to be applied.
    """
    warnings.filterwarnings(action, category=_warnings[name])


def _make_warning_table():

    max_name_len = max([len(name) for name in _warnings])
    max_desc_len = max([len(' '.join(c.__doc__.split())) for c in _warnings.values()])
    
    name_header = "Option Name"
    desc_header = "Description"
    print(f'| {name_header:<{max_name_len}} | {desc_header:<{max_desc_len}} |')
    print(f'| {max_name_len*"-"} | {max_desc_len*"-"} |')

    for name, _class in _warnings.items():
        desc = ' '.join(_class.__doc__.split())
        print(f'| {name:<{max_name_len}} | {desc:<{max_desc_len}} |')



if __name__ == '__main__':
    # apply_filter_option('warn_no_mpi', 'error')
    # warnings.warn(NoMPIWarning('This is my message', 'prefix'), stacklevel=2)
    # print('done')
    print(len(_warnings))
    _make_warning_table()
