import inspect
import sys
import io
import textwrap
import warnings


__all__ = ['filter_warnings', 'issue_warning', 'OpenMDAOWarning', 'AllowableSetupError',
           'SetupWarning', 'InputDefaultsWarning', 'ApproxPartialsWarning',
           'DistributedComponentWarning', 'CaseRecorderWarning', 'SingularJacWarning',
           'KrigingCacheWarning', 'PromotionWarning', 'UnusedOptionWarning', 'ColoringWarning',
           'MPIWarning', 'UnitsWarning', 'SolverWarning']


class OpenMDAOWarning(UserWarning):
    """
    Base class for all OpenMDAO warnings.
    """
    name = 'warn_all'
    action = 'always'

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
        self._str = f' [{prefix}]: {msg}' if prefix else f' {msg}'

    def __str__(self):
        return self._str


class AllowableSetupError(UserWarning):
    """
    A class of setup errors that are treated as warnings to allow the N2 to be built.
    """
    name = '_warn_allowable_setup_error'
    action = 'error'

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
        self._str = f' {prefix}: {msg}' if prefix else f' {msg}'

    def __str__(self):
        return self._str


class SetupWarning(OpenMDAOWarning):
    """
    Warning class for warnings that occur during setup.
    """
    name = 'warn_setup'
    action = 'always'


#
# Subclasses of SetupWarning
#


class InputDefaultsWarning(SetupWarning):
    """
    Warning dealing with the use of set_input_defaults.
    """
    name = 'warn_input_defaults'
    action = 'always'


class PromotionWarning(SetupWarning):
    """
    Warning dealing with the promotion of an input or output.
    """
    name = 'warn_promotion'
    action = 'always'

class UnitsWarning(OpenMDAOWarning):
    """
    Warning which is issued when a unitless output is connected to an input with units, or vice versa.
    """
    name = 'warn_units'
    action = 'always'


class ApproxPartialsWarning(OpenMDAOWarning):
    """
    Warning issued when the approximated partials cannot be evaluated as expected.
    """
    name = 'warn_approx_partials'
    action = 'always'


class MPIWarning(SetupWarning):
    """
    Warning dealing with the availability of MPI.
    """
    name = 'warn_mpi'
    action = 'always'


class ColoringWarning(SetupWarning):
    """
    Warning dealing with derivative coloring.
    """
    name = 'warn_coloring'
    action = 'always'


class DistributedComponentWarning(OpenMDAOWarning):
    """
    Warning specific to a distributed component.
    """
    name = 'warn_distributed_component'
    action = 'always'

#
# End SetupWarning subclasses
#


class SolverWarning(OpenMDAOWarning):
    """
    Warning base class for solver-related warnings.
    """
    name = 'warn_solver'
    action = 'always'


class SingularJacWarning(OpenMDAOWarning):
    """
    Warning which is issued when requested data cannot be found in a recording.
    """
    name = 'warn_singular_jac'
    action = 'once'


class UnusedOptionWarning(OpenMDAOWarning):
    """
    Warning dealing with an unnecessary option or argument being provided.
    """
    name = 'warn_unused_option'
    action = 'always'


class CaseRecorderWarning(OpenMDAOWarning):
    """
    Warning pertaining to case recording and reading.
    """
    name = 'warn_case_recorder'
    action = 'always'


class KrigingCacheWarning(OpenMDAOWarning):
    """
    Warning which is issued when the KrigingSurrogate fails to load the kriging surrogate cache
    during training.
    """
    name = 'warn_kriging_cache'
    action = 'always'


_warnings = {_class.name: _class for _, _class in
             inspect.getmembers(sys.modules[__name__], inspect.isclass) if issubclass(_class, Warning)}


def get_warning_defaults():
    """
    Return a dictionary of the default action of each warning type.
    Returns
    -------

    """
    defaults = {_class.name: _class.action for _, _class in
                inspect.getmembers(sys.modules[__name__], inspect.isclass)
                if issubclass(_class, Warning) and not _class.name.startswith('_')}
    return defaults


def filter_warnings(reset_to_defaults=False, **kwargs):
    """
    Apply the warning filters as given by warning_options.

    This is necessary when testing the filters, because Python resets the default filters
    before running each test.
    """
    _actions = ['warn', 'error', 'ignore', 'once', 'always', 'module', 'default']

    if reset_to_defaults:
        for w_name, w_class in _warnings.items():
            warnings.filterwarnings(w_class.action, category=w_class)

    for w_name, action in kwargs.items():
        _action = 'error' if action == 'raise' else action
        if w_name not in _warnings:
            valid = [key for key in _warnings.keys() if not key.startswith('_')]
            msg = f"The warning '{w_name}' is not a valid OpenMDAO warning. \n" \
                  f"Valid values are {valid}."
            raise ValueError("\n".join(textwrap.wrap(msg, width=80)))
        if action not in _actions:
            msg = f"The action '{action}' for warning '{w_name}' is not a valid action. \n" \
                  f"Must be one of {_actions}.  See Python warning documentation for more details."
            raise ValueError(msg)


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


# When we import OpenMDAO and load this module, set the default filters on these warnings.
filter_warnings(reset_to_defaults=True)


if __name__ == '__main__':
    # print(len(_warnings))
    # print(_make_table())
    issue_warning(SolverWarning('foo', prefix='my.comp'))
