"""
OpenMDAO custom error: AnalysisError.
"""
from openmdao.utils.om_warnings import _warn_simple_format, reset_warning_registry
import warnings


class AnalysisError(Exception):
    """
    Analysis Error.

    This exception indicates that a possibly recoverable numerical error occurred in an analysis
    code or a subsolver.

    Parameters
    ----------
    error : str
        Error message.
    location : None or inspect.currentframe()
        Inspect.currentframe of error being raised.
    msginfo : str
        Name of component that raise the AnalysisError.
    """

    def __init__(self, error, location=None, msginfo=None):
        """
        Initialize AnalysisError.
        """
        super().__init__(error)
        if location is not None:
            if hasattr(location, 'f_lineno'):
                # from inspect.currentframe()
                line_num = location.f_lineno
                file_name = location.f_code.co_filename
            else:
                # from inspect.getframeinfo(inspect.currentframe())
                line_num = location.lineno
                file_name = location.filename
            with reset_warning_registry():
                warnings.formatwarning = _warn_simple_format
                msg = (f"Analysis Error: {msginfo} Line {line_num} of file {file_name}")
                warnings.warn(msg, UserWarning, 2)
