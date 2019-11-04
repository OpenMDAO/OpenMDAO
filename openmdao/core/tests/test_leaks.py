
import unittest
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

try:
    from pyoptsparse import Optimization
except ImportError:
    Optimization = Driver

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized


# much of the following code was taken from https://pymotw.com/3/gc/


# Ignore references from local variables in this module, global
# variables, and from the garbage collector's bookkeeping.
REFERRERS_TO_IGNORE = [locals(), globals(), gc.garbage]

_om_classes = (System, om.Problem, Vector, Driver, Solver, MethodType, FunctionType,
               GeneratorType, CoroutineType, Optimization)


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
def record_leaks(classes=(object,),
                 flags=gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL):

    lst = []

    # Force a sweep
    gc.collect()

    gc.set_debug(flags)

    yield lst

    # Force a sweep
    gc.collect()

    # Report on what was left
    for o in gc.garbage:
        if isinstance(o, classes):
            rlist = []
            for r in find_referring_objects(o, classes):
                rlist.append(r)

            # only add 'o' to the list if there are objects referring to it
            if rlist:
                lst.append((o, rlist))

    # Reset the debug flags before exiting to avoid dumping a lot
    # of extra information and making the output more
    # confusing.
    gc.set_debug(0)


# class LeakTestPyoptSparseCase(unittest.TestCase):

#     ISOLATED = True

#     def test_leaks_pyoptsparse(self):

#         with record_leaks(_om_classes) as rec:
#             for i in range(3):
#                 p_color = run_opt(om.pyOptSparseDriver, 'auto', optimizer='SLSQP',
#                                   dynamic_total_coloring=True, partial_coloring=True)
#                 p_color = None

#         if len(rec) > 0:
#             msgs = []
#             for o, refs in rec:
#                 try:
#                     n = o.__name__
#                 except AttributeError:
#                     n = ''
#                 msgs.append('Retained: {} {}'.format(n, type(o)))

#                 for r in refs:
#                     try:
#                         nr = r.__name__
#                     except AttributeError:
#                         nr = ''
#                     msgs.append("   referred to by: {} {}".format(nr, type(r)))

#             self.fail('\n'.join(msgs))

#         # Force a sweep
#         gc.collect()


class LeakTestScipyCase(unittest.TestCase):

    ISOLATED = True

    def test_leaks_scipyopt(self):

        with record_leaks(_om_classes) as rec:
            for i in range(3):
                p_color = run_opt(om.ScipyOptimizer, 'auto', optimizer='SLSQP',
                                  dynamic_total_coloring=True, partial_coloring=True)
                p_color = None

        if len(rec) > 0:
            msgs = []
            for o, refs in rec:
                try:
                    n = o.__name__
                except AttributeError:
                    n = ''
                msgs.append('Retained: {} {}'.format(n, type(o)))

                for r in refs:
                    try:
                        nr = r.__name__
                    except AttributeError:
                        nr = ''
                    msgs.append("   referred to by: {} {}".format(nr, type(r)))

            self.fail('\n'.join(msgs))


if __name__ == '__main__':
    unittest.main()
