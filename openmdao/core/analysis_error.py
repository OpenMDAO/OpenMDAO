"""
OpenMDAO custom error: AnalysisError.
"""


class AnalysisError(Exception):
    """
    Analysis Error.

    This exception indicates that a possibly recoverable numerical error occurred in an analysis
    code or a subsolver.
    """

    pass
