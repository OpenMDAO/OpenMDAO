"""Some miscellaneous utility functions."""

import os
import warnings


def warn_deprecation(msg):
    """
    Raise a warning and prints a deprecation message to stdout.

    Parameters
    ----------
    msg : str
        Message that will be printed to stdout.
    """
    # Deprecation warnings need to be printed regardless of debug level
    warnings.simplefilter('always', DeprecationWarning)

    # note, stack level 3 should take us back to original caller.
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    warnings.simplefilter('ignore', DeprecationWarning)


def set_pyoptsparse_opt(optname):
    """For testing, sets the pyoptsparse optimizer using the given optimizer
    name.  This may be modified based on the value of
    OPENMDAO_FORCE_PYOPTSPARSE_OPT.  This can be used on systems that have
    SNOPT installed to force them to use SLSQP in order to mimic our test
    machines on travis and appveyor.

    Parameters
    ----------
    optname : str
        Name of pyoptsparse optimizer that is requested by the test.

    Returns
    -------
    object
        Pyoptsparse optimizer instance.
    str
        Pyoptsparse optimizer string
    """

    OPT = None
    OPTIMIZER = None
    force = os.environ.get('OPENMDAO_FORCE_PYOPTSPARSE_OPT')
    if force:
        optname = force

    try:
        from pyoptsparse import OPT
        try:
            OPT(optname)
            OPTIMIZER = optname
        except:
            if optname != 'SLSQP':
                try:
                    OPT('SLSQP')
                    OPTIMIZER = 'SLSQP'
                except:
                    pass
    except:
        pass

    return OPT, OPTIMIZER
