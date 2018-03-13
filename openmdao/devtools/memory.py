"""Various debugging functions."""

from __future__ import print_function

import sys
import os
import functools

try:
    import resource

    def max_mem_usage():
        """
        Returns
        -------
        The max memory used by this process and its children, in MB.
        """
        denom = 1024.
        if sys.platform == 'darwin':
            denom *= denom
        total = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / denom
        total += resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / denom
        return total

except ImportError:
    resource = None
    def max_mem_usage(fn):
        raise RuntimeError("The 'max_mem_usage' function requires the 'resource' package.")

try:
    import psutil

    def mem_usage(msg='', out=sys.stdout):
        """
        Returns
        -------
        The current memory used by this process (and it's children?), in MB.
        """
        denom = 1024. * 1024.
        p = psutil.Process(os.getpid())
        mem = p.memory_info().rss / denom
        if msg:
            print(msg,"%6.3f MB" % mem, file=out)
        return mem

    def diff_mem(fn):
        """
        This gives the difference in memory before and after the
        decorated function is called. Does not show output unless there is a memory increase.
        Requires psutil to be installed.
        """
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            startmem = mem_usage()
            ret = fn(*args, **kwargs)
            maxmem = mem_usage()
            diff = maxmem - startmem
            if diff > 0.0:
                if args and hasattr(args[0], 'pathname'):
                    name = args[0].pathname
                else:
                    name = str(args[0])
                print(name, "%s added %.0f KB (total: %6.3f MB)" %
                      (fn.__name__, diff * 1024., maxmem))
            return ret
        return wrapper

except ImportError:
    psutil = None
    def mem_usage(fn):
        raise RuntimeError("The 'mem_usage' function requires the 'psutil' package.  You can "
                           "install it using 'pip install psutil'.")
    def diff_mem(fn):
        raise RuntimeError("The 'diff_mem' function requires the 'psutil' package.  You can "
                           "install it using 'pip install psutil'.")


try:
    import objgraph

    def new_objects(fn):
        """
        This performs garbage collection before and after the function call and prints any
        new objects that have not been garbage collected after the function returns.  This
        MAY indicate a memory leak, but not necessarily.
        """
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start_objs = objgraph.typestats()
            start_objs['frame'] += 1
            start_objs['cell'] += 1
            ret = fn(*args, **kwargs)
            for obj, _, delta_objs in objgraph.growth(peak_stats=start_objs):
                print(str(fn), "added %s %+d" % (obj, delta_objs))
            return ret
        return wrapper

except ImportError:
    objgraph = None
    def new_objects(fn):
        raise RuntimeError("The 'new_objects' decorator requires the 'objgraph' package.  You can "
                           "install it using 'pip install objgraph'.")
