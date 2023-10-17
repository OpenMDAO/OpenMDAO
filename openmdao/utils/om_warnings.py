"""
A module for OpenMDAO-specific warnings and associated functions.
"""

from functools import cmp_to_key
import inspect
import re
import sys
import io
import warnings


__all__ = ['issue_warning', 'reset_warnings', 'reset_warning_registry', '_warn_simple_format',
           'OpenMDAOWarning', 'SetupWarning', 'DistributedComponentWarning', 'CaseRecorderWarning',
           'CacheWarning', 'PromotionWarning', 'UnusedOptionWarning', 'DerivativesWarning',
           'MPIWarning', 'UnitsWarning', 'SolverWarning', 'DriverWarning', 'OMDeprecationWarning',
           'OMInvalidCheckDerivativesOptionsWarning']


class OpenMDAOWarning(UserWarning):
    """
    Base class for all OpenMDAO warnings.
    """

    name = 'warn_openmdao'
    filter = 'always'


class SetupWarning(OpenMDAOWarning):
    """
    Warning class for warnings that occur during setup.
    """

    name = 'warn_setup'
    filter = 'always'


class PromotionWarning(SetupWarning):
    """
    Warning dealing with the promotion of an input or output.
    """

    name = 'warn_promotion'
    filter = 'always'


class UnitsWarning(SetupWarning):
    """
    Warning which is issued when unitless variable is connected to a variable with units.
    """

    name = 'warn_units'
    filter = 'always'


class DerivativesWarning(OpenMDAOWarning):
    """
    Warning issued when the approximated partials or coloring cannot be evaluated as expected.
    """

    name = 'warn_derivatives'
    filter = 'always'


class MPIWarning(SetupWarning):
    """
    Warning dealing with the availability of MPI.
    """

    name = 'warn_mpi'
    filter = 'always'


class DistributedComponentWarning(SetupWarning):
    """
    Warning specific to a distributed component.
    """

    name = 'warn_distributed_component'
    filter = 'always'


class SolverWarning(OpenMDAOWarning):
    """
    Warning base class for solver-related warnings.
    """

    name = 'warn_solver'
    filter = 'always'


class DriverWarning(OpenMDAOWarning):
    """
    Warning which is issued during the execution of a driver.
    """

    name = 'warn_driver'
    filter = 'always'


class UnusedOptionWarning(OpenMDAOWarning):
    """
    Warning dealing with an unnecessary option or argument being provided.
    """

    name = 'warn_unused_option'
    filter = 'always'


class CaseRecorderWarning(OpenMDAOWarning):
    """
    Warning pertaining to case recording and reading.
    """

    name = 'warn_case_recorder'
    filter = 'always'


class CacheWarning(OpenMDAOWarning):
    """
    Warning which is issued when the a cache is invalid and needs to be rebuilt.
    """

    name = 'warn_cache'
    filter = 'always'


class OMDeprecationWarning(OpenMDAOWarning):
    """
    An OpenMDAO-specific deprecation warning that is noisy by default, unlike the Python one.
    """

    name = 'warn_deprecation'
    filter = 'once'


class OMInvalidCheckDerivativesOptionsWarning(OpenMDAOWarning):
    """
    An OpenMDAO-specific warning that defaults to raising an exception.

    It tells the user that they are checking derivatives with the exact same method
    used to computer the derivatives, including all the settings.
    """

    name = 'warn_invalid_check_derivatives'
    filter = 'error'


_warnings = [_class for _, _class in
             inspect.getmembers(sys.modules[__name__], inspect.isclass)
             if issubclass(_class, Warning)]


# The order of warnings matters in the Python warnings module! See
#    https://docs.python.org/3/library/warnings.html#the-warnings-filter
# reset_warnings processes the warnings classes in the order returned by inspect.getmembers.
# The order is alphabetized. But this causes problems.
# To solve this problem, we sort the warnings classes so that parents come before their children
def _compare_class_hierarchy(class1, class2):
    if issubclass(class1, class2):
        return 1
    elif issubclass(class2, class1):
        return -1
    else:
        return 0


_warnings = sorted(_warnings, key=cmp_to_key(_compare_class_hierarchy))


def reset_warnings():
    """
    Apply the default warning filter actions for the OpenMDAO-specific warnings.

    This is necessary when testing the filters, because Python resets the default filters
    before running each test.
    """
    for w_class in _warnings:
        warnings.filterwarnings(w_class.filter, category=w_class)


def issue_warning(msg, prefix='', stacklevel=2, category=OpenMDAOWarning):
    """
    Display a warning with the desired stack level and optional prefix.

    Ex.

    >>> om.issue_warning('some warning message', prefix=self.pathname, category=om.SetupWarning)

    Parameters
    ----------
    msg : str
        The warning message.
    prefix : str
        An optional prefix to be prepended to the warning message (usually the system path).
    stacklevel : int
        Number of levels up the stack to identify as the warning location.
    category : class
        The class of warning to be issued.
    """
    old_format = warnings.formatwarning
    warnings.formatwarning = _warn_simple_format
    _msg = f'{prefix}: {msg}' if prefix else msg
    try:
        warnings.warn(_msg, category=category, stacklevel=stacklevel)
    finally:
        warnings.formatwarning = old_format


def _make_table(superclass=OpenMDAOWarning):
    """
    Generate a markdown table of the warning options.

    Returns
    -------
    str
        A string representation of a markdown table of the warning options.
    """
    s = io.StringIO()
    max_name_len = max([len(_class.__name__) for _class in _warnings])
    max_desc_len = max([len(' '.join(c.__doc__.split())) for c in _warnings])

    name_header = "Warning Class"
    desc_header = "Description"
    print(f'| {name_header:<{max_name_len}} | {desc_header:<{max_desc_len}} |', file=s)
    print(f'| {max_name_len * "-"} | {max_desc_len * "-"} |', file=s)

    for _class in _warnings:
        if isinstance(_class, superclass) or issubclass(_class, superclass):
            desc = ' '.join(_class.__doc__.split())
            print(f'| {_class.__name__:<{max_name_len}} | {desc:<{max_desc_len}} |', file=s)
    return s.getvalue()


def warn_deprecation(msg):
    """
    Raise a warning and prints a deprecation message to stdout.

    Parameters
    ----------
    msg : str
        Message that will be printed to stdout.
    """
    # note, stack level 3 should take us back to original caller.
    issue_warning(msg, stacklevel=3, category=OMDeprecationWarning)


def _warn_simple_format(message, category, filename, lineno, file=None, line=None):
    """
    Provide a warning format for OpenMDAO warnings.

    Parameters
    ----------
    message : str
        The warning message
    category : class
        The warning class being issued.
    filename : str
        The filename from which the warning is issued.
    lineno : int
        The line number from which the warning is issued.
    file : str
        Ignored in this implementation.
    line : str
        The line of code causing the warning (ignored in this implementation)

    Returns
    -------
    str
        A formatted warning message.

    """
    return f'{filename}:{lineno}: {category.__name__}:{message}\n'


class reset_warning_registry(object):
    """
    Context manager which archives & clears warning registry for duration of context.

    From https://bugs.python.org/file40031/reset_warning_registry.py
    (https://bugs.python.org/issue21724, https://github.com/python/cpython/issues/65923)

    Parameters
    ----------
    pattern : regex pattern
        Causes manager to only reset modules whose names match pattern. defaults to ``".*"``.

    Attributes
    ----------
    _pattern : regex pattern
        Causes manager to only reset modules whose names match this pattern. defaults to ``".*"``.
    """

    #: regexp for filtering which modules are reset
    _pattern = None

    #: dict mapping module name -> old registry contents
    _backup = None

    def __init__(self, pattern=None):
        """
        Initialize all attributes.
        """
        self._pattern = re.compile(pattern or ".*")

    def __enter__(self):
        """
        Enter the runtime context related to this object.

        Returns
        -------
        reset_warning_registry
            This context manager.

        """
        # archive and clear the __warningregistry__ key for all modules
        # that match the 'reset' pattern.
        pattern = self._pattern
        backup = self._backup = {}
        for name, mod in list(sys.modules.items()):
            if pattern.match(name):
                reg = getattr(mod, "__warningregistry__", None)
                if reg:
                    backup[name] = reg.copy()
                    reg.clear()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the runtime context related to this object.

        Parameters
        ----------
        exc_type : Exception class
            The type of the exception.
        exc_value : Exception instance
            The exception instance raised.
        traceback : regex pattern
            Traceback object.
        """
        # restore warning registry from backup
        modules = sys.modules
        backup = self._backup
        for name, content in backup.items():
            mod = modules.get(name)
            if mod is None:
                continue
            reg = getattr(mod, "__warningregistry__", None)
            if reg is None:
                setattr(mod, "__warningregistry__", content)
            else:
                reg.clear()
                reg.update(content)

        # clear all registry entries that we didn't archive
        pattern = self._pattern
        for name, mod in list(modules.items()):
            if pattern.match(name) and name not in backup:
                reg = getattr(mod, "__warningregistry__", None)
                if reg:
                    reg.clear()


# When we import OpenMDAO and load this module, set the default filters on these warnings.
reset_warnings()
