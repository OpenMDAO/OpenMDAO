"""Various debugging functions."""

import sys
import os
import functools
import gc

try:
    import resource

    def max_mem_usage():
        """
        Return the maximum resident memory used by this process and its children so far.

        Returns
        -------
        The max resident memory used by this process and its children, in MB.
        """
        denom = 1024.
        if sys.platform == 'darwin':
            denom *= denom
        total = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / denom
        total += resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / denom
        return total

except ImportError:
    resource = None
    def max_mem_usage():
        raise RuntimeError("The 'max_mem_usage' function requires the 'resource' package.")

try:
    import psutil

    def mem_usage(msg='', out=sys.stdout, resident=True):
        """
        Display current resident or virtual memory usage.

        Parameters
        ----------
        msg : str
            String prepended to each reported memory usage.
        out : file-like
            Output will be sent to this stream.
        resident : bool
            If True, report resident memory usage, else virtual.

        Returns
        -------
        The current memory used by this process, in MB.
        """
        denom = 1024. * 1024.
        p = psutil.Process(os.getpid())
        if resident:
            mem = p.memory_info().rss / denom
        else:
            mem = p.memory_info().vms / denom
        if msg:
            print(msg,"%6.3f MB" % mem, file=out)
        return mem

    def diff_mem(fn):
        """
        Decorator that prints the difference in resident memory usage resulting from the call.

        Does not show output unless there is a memory increase. Requires psutil to be installed.

        Parameters
        ----------
        fn : function
            The function being decorated.

        Returns
        -------
        function
            The wrapper function.
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

    def check_iter_mem(niter, func, *args, **kwargs):
        """
        Run func niter times and collect info on memory usage.

        Parameters
        ----------
        niter : int
            Number of times to run func.
        func : function
            A function that takes no arguments.
        *args : tuple
            Positional args passed to func.
        **kwargs : dict
            Named args to be passed to func.
        """
        gc.collect()

        yield mem_usage()
        for i in range(niter):
            func(*args, **kwargs)
            gc.collect()
            yield mem_usage()

except ImportError:
    psutil = None
    def mem_usage(*args, **kwargs):
        raise RuntimeError("The 'mem_usage' function requires the 'psutil' package.  You can "
                           "install it using 'pip install psutil'.")
    def diff_mem(*args, **kwargs):
        raise RuntimeError("The 'diff_mem' function requires the 'psutil' package.  You can "
                           "install it using 'pip install psutil'.")
    def check_iter_mem(*args, **kwargs):
        raise RuntimeError("The 'check_iter_mem' function requires the 'psutil' package.  You can "
                           "install it using 'pip install psutil'.")


try:
    import objgraph

    def get_new_objects(lst, fn, *args, **kwargs):
        """
        Collect types and numbers of new objects left over after the given function is called.

        If lst is not empty after the call, this MAY indicate a memory leak, but not necessarily,
        since some functions are intended to create new objects for later use.

        Parameters
        ----------
        lst : list
            List used to collect objects and deltas.
        fn : function
            The function being checked for possible memory leaks.
        *args : tuple
            Positional args passed to fn.
        **kwargs : dict
            Named args to be passed to fn.

        Returns
        -------
        object
            The object returned by the call to fn.
        """
        gc.collect()
        start_objs = objgraph.typestats()
        start_objs['frame'] += 1
        start_objs['function'] += 1
        start_objs['builtin_function_or_method'] += 1
        start_objs['cell'] += 1
        ret = fn(*args, **kwargs)
        lst.extend([(str(o), delta) for o, _, delta in objgraph.growth(peak_stats=start_objs)])
        return ret


    def new_objects(fn):
        """
        A decorator that prints types and numbers of new objects left over after calling fn.

        Parameters
        ----------
        fn : function
            The function being checked for possible memory leaks.

        Returns
        -------
        function
            A wrapper for fn that reports possible memory leaks.
        """
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            lst = []
            ret = get_new_objects(lst, fn, *args, **kwargs)
            for obj, delta_objs in lst:
                print(str(fn), "added %s %+d" % (obj, delta_objs))
            return ret
        return wrapper


    def check_iter_leaks(niter, func, *args, **kwargs):
        """
        Run func niter times and collect info on new objects left over after each iteration.

        Parameters
        ----------
        niter : int
            Number of times to run func.
        func : function
            A function that takes no arguments.
        *args : tuple
            Positional args passed to func.
        **kwargs : dict
            Named args to be passed to func.

        Returns
        -------
        set
            set of tuples of the form (typename, count)
        """
        if niter < 2:
            raise RuntimeError("Must run the function at least twice, but niter={}".format(niter))
        iters = []
        gc.collect()
        start_objs = objgraph.typestats()
        if 'frame' in start_objs:
            start_objs['frame'] += 1
        start_objs['function'] += 1
        start_objs['builtin_function_or_method'] += 1
        start_objs['cell'] += 1
        for i in range(niter):
            func(*args, **kwargs)
            gc.collect()
            lst = [(str(o), delta) for o, _, delta in objgraph.growth(peak_stats=start_objs)]
            iters.append(lst)

        set1 = set(iters[-2])
        set2 = set(iters[-1])

        return set2 - set1


    def list_iter_leaks(leakset, out=sys.stdout):
        """
        Print any new objects left over after each call to the specified function.

        Parameters
        ----------
        leakset : set of tuples of the form (objtype, count)
            Output of check_iter_leaks.
        out : file-like
            Output stream.
        """
        if leakset:
            print("\nPossible leaked objects:", file=out)
            for objstr, deltas in leakset:
                print(objstr, deltas, file=out)
            print(file=out)
        else:
            print("\nNo possible memory leaks detected.\n", file=out)

except ImportError:
    objgraph = None
    def get_new_objects(*args, **kwargs):
        raise RuntimeError("The 'get_new_objects' function requires the 'objgraph' package.  "
                           "You can install it using 'pip install objgraph'.")

    def new_objects(*args, **kwargs):
        raise RuntimeError("The 'new_objects' decorator requires the 'objgraph' package.  You can "
                           "install it using 'pip install objgraph'.")

    def check_iter_leaks(*args, **kwargs):
        raise RuntimeError("The 'check_iter_leaks' function requires the 'objgraph' package.  "
                           "You can install it using 'pip install objgraph'.")

    def list_iter_leaks(*args, **kwargs):
        raise RuntimeError("The 'list_iter_leaks' function requires the 'objgraph' package.  "
                           "You can install it using 'pip install objgraph'.")

def plot_mem(mems, fname=None):
    """
    Plot memory usage.

    Parameters
    ----------
    mems : iter of float
        Iterator containing memory usage values.
    fname : str (optional)
        If specified, save the plot to this file.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    mems = list(mems)
    ax.plot(list(range(len(mems))), mems)
    ax.set(xlabel='Iterations', ylabel='Memory (MB)', title='Memory useage per iteration')
    ax.grid()
    if fname is not None:
        fig.savefig(fname)
    plt.show()

