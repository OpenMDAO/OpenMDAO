
import gc
from contextlib import contextmanager
from types import FunctionType, MethodType, CoroutineType, GeneratorType, FrameType
from collections import defaultdict

import openmdao.api as om
from openmdao.core.tests.test_coloring import run_opt
from openmdao.core.system import System
from openmdao.vectors.vector import Vector
from openmdao.solvers.solver import Solver
from openmdao.core.driver import Driver
from openmdao.jacobians.jacobian import Jacobian


# much of the following code was taken from https://pymotw.com/3/gc/


# Ignore references from local variables in this module, global
# variables, and from the garbage collector's bookkeeping.
REFERRERS_TO_IGNORE = [locals(), globals(), gc.garbage]

_om_classes = (System, om.Problem, Vector, Driver, Solver, MethodType)


def find_referring_objects(obj, classes=(object,)):
    for ref in gc.get_referrers(obj):
        if ref in REFERRERS_TO_IGNORE:
            continue
        if isinstance(ref, classes):
            yield ref
        elif isinstance(ref, dict):
            # An instance or other namespace dictionary
            for parent in find_referring_objects(ref, classes):
                yield parent


@contextmanager
def report_leaks(classes=(object,),
                 flags=gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL):
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
            for r in find_referring_objects(o):#, _om_classes):
                print("   referred to by: {} 0x{:x}".format(r, id(r)))

    # Reset the debug flags before exiting to avoid dumping a lot
    # of extra information and making the example output more
    # confusing.
    gc.set_debug(0)



if __name__ == '__main__':
    with report_leaks(_om_classes):
        for i in range(3):
            p_color = run_opt(om.pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False,
                            dynamic_total_coloring=True, partial_coloring=True)
            p_color = None

