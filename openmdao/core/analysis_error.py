"""
OpenMDAO custom error: AnalysisError.
"""
from openmdao.warnings import _warn_simple_format, reset_warning_registry
import warnings


class AnalysisError(Exception):
    """
    Analysis Error.

    This exception indicates that a possibly recoverable numerical error occurred in an analysis
    code or a subsolver.
    """

    def __init__(self, error, location=None, msginfo=None):
        """
        Initialize AnalysisError.

        Parameters
        ----------
        error : str
            Error message.
        location : None or inspect.currentframe()
            inspect.currentframe of error being raised.
        msginfo : str
            Name of component that raise the AnalysisError.
        """
        super().__init__(error)
        if location is not None:
            with reset_warning_registry():
                warnings.formatwarning = _warn_simple_format
                msg = (f"Analysis Error: {msginfo} Line {location.lineno} of file "
                       f"{location.filename}")
                warnings.warn(msg, UserWarning, 2)
