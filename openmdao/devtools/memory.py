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
    mem_usage = diff_mem = None
