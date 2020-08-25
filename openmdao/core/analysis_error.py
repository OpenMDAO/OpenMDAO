"""
OpenMDAO custom error: AnalysisError.
"""
import inspect
from openmdao.utils.general_utils import simple_warning


class AnalysisError(Exception):
    """
    Analysis Error.

    This exception indicates that a possibly recoverable numerical error occurred in an analysis
    code or a subsolver.
    """

    def __init__(self, error, location=None):
        """
        Initialize AnalysisError.

        Parameters
        ----------
        error : str
            Error message.
        location : None or inspect.currentframe()
            inspect.currentframe of error being raised.
        """
        super(AnalysisError, self).__init__(error)
        if location is not None:
            simple_warning(f"Analysis Error: Line {location.lineno} of file {location.filename}")
