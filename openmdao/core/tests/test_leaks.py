
import gc
from contextlib import contextmanager

import openmdao.api as om
from openmdao.core.tests.test_coloring import run_opt
from openmdao.core.system import System
from openmdao.vectors.vector import Vector
from openmdao.solvers.solver import Solver
from openmdao.core.driver import Driver


@contextmanager
def report_leaks(classes=(object,), flags=gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL):
    gc.set_debug(flags)

    yield

    # Force a sweep
    print('Collecting garbage')
    gc.collect()
    print('Done')

    # Report on what was left
    for o in gc.garbage:
        if isinstance(o, classes):
            print('Retained: {} 0x{:x}'.format(o, id(o)))

    # Reset the debug flags before exiting to avoid dumping a lot
    # of extra information and making the example output more
    # confusing.
    gc.set_debug(0)


if __name__ == '__main__':
    with report_leaks((System, om.Problem, Vector, Driver, Solver)):
        for i in range(3):
            p_color = run_opt(om.pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False,
                            dynamic_total_coloring=True, partial_coloring=True)
            p_color = None

