import inspect
import sys
import io
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
        self._str = f'{prefix}: {msg}' if prefix else f'{msg}'

    def __str__(self):
        return self._str


class AllowableSetupError(UserWarning):
    """
    A class of setup errors that are treated as warnings to allow the N2 to be built.
    """
    name = '_warn_allowable_setup_error'

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


class SetupWarning(OpenMDAOWarning):
    """
    Warning class for warnings that occur during setup.
    """
    name = 'warn_setup'


#
# Subclasses of SetupWarning
#


class InputDefaultsWarning(SetupWarning):
    """
    Warning dealing with the use of set_input_defaults.
    """
    name = 'warn_input_defaults'


class PromotionWarning(SetupWarning):
    """
    Warning dealing with the promotion of an input or output.
    """
    name = 'warn_promotion'


class UnitsWarning(OpenMDAOWarning):
    """
    Warning which is issued when a unitless output is connected to an input with units, or vice versa.
    """
    name = 'warn_units'


class ApproxPartialsWarning(OpenMDAOWarning):
    """
    Warning issued when the approximated partials cannot be evaluated as expected.
    """
    name = 'warn_approx_partials'


class MPIWarning(SetupWarning):
    """
    Warning dealing with the availability of MPI.
    """
    name = 'warn_mpi'


class ColoringWarning(SetupWarning):
    """
    Warning dealing with derivative coloring.
    """
    name = 'warn_coloring'


class DistributedComponentWarning(OpenMDAOWarning):
    """
    Warning specific to a distributed component.
    """
    name = 'warn_distributed_component'

#
# End SetupWarning subclasses
#


class SolverWarning(OpenMDAOWarning):
    """
    Warning base class for solver-related warnings.
    """
    name = 'warn_solver'


class SingularJacWarning(OpenMDAOWarning):
    """
    Warning which is issued when requested data cannot be found in a recording.
    """
    name = 'warn_singular_jac'


class UnusedOptionWarning(OpenMDAOWarning):
    """
    Warning dealing with an unnecessary option or argument being provided.
    """
    name = 'warn_unused_option'


class CaseRecorderWarning(OpenMDAOWarning):
    """
    Warning pertaining to case recording and reading.
    """
    name = 'warn_case_recorder'


class KrigingCacheWarning(OpenMDAOWarning):
    """
    Warning which is issued when the KrigingSurrogate fails to load the kriging surrogate cache
    during training.
    """
    name = 'warn_kriging_cache'


_warnings = {_class.name: _class for _, _class in
             inspect.getmembers(sys.modules[__name__], inspect.isclass)}


def _warn_simple_format(message, category, filename, lineno, file=None, line=None):
    return f'{filename}:{lineno}: {category.__name__}:{message}\n'


def issue_warning(w, stacklevel=2):
    """
    Display a simple warning message without the annoying extra line showing the warning call.

    Parameters
    ----------
    w : UserWarning
        A warning derived from OpenMDAOWarning or AllowableSetupError.
    stacklevel : int
        Number of levels up the stack to identify as the warning location.
    """
    old_format = warnings.formatwarning
    warnings.formatwarning = _warn_simple_format
    try:
        warnings.warn(w, stacklevel)
    finally:
        warnings.formatwarning = old_format


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


def _make_table():
    """
    Generate a markdown table of the warning options.

    Returns
    -------
    str
        A string representation of a markdown table of the warning options.
    """
    s = io.StringIO()
    max_name_len = max([len(name) for name in _warnings])
    max_desc_len = max([len(' '.join(c.__doc__.split())) for c in _warnings.values()])
    
    name_header = "Option Name"
    desc_header = "Description"
    print(f'| {name_header:<{max_name_len}} | {desc_header:<{max_desc_len}} |', file=s)
    print(f'| {max_name_len*"-"} | {max_desc_len*"-"} |', file=s)

    for name, _class in _warnings.items():
        desc = ' '.join(_class.__doc__.split())
        print(f'| {name:<{max_name_len}} | {desc:<{max_desc_len}} |', file=s)
    return s.getvalue()


if __name__ == '__main__':
    # print(len(_warnings))
    # print(_make_table())

    warnings.warn(SolverWarning('foo', prefix='bar'), stacklevel=2)
