
import time
from itertools import repeat
import numpy as np
import openmdao.api as om
from openmdao.utils.general_utils import shape2tuple, env_truthy
from openmdao.utils.mpi import MPI


debug = env_truthy('OM_DEBUG')

class ExplicitSleepComp(om.ExplicitComponent):
    """
    A component with a settable number of params and outputs and delays.

    Component does a simple passthrough, one variable at a time, so there will be some
    overhead for vector setitem/getitem calls.

    If run under MPI, the sleep times will be divided by the number of procs in the comm in
    order to simulate a component that splits up its work across its procs.
    """
    def __init__(self, nvars, compute_delay=0.001, compute_partials_delay=0.001,
                 var_default=1.0, add_var_kwargs=None, use_coloring=True, **kwargs):
        super().__init__(**kwargs)

        self.nvars = nvars
        self.add_var_kwargs = add_var_kwargs
        self.compute_delay = compute_delay
        self.compute_partials_delay = compute_partials_delay
        self.use_coloring = use_coloring
        self.inames = []
        self.onames = []
        self.varshape = ()  # default to scalar variables

        if add_var_kwargs is not None:
            if 'shape' in add_var_kwargs:
                self.varshape = shape2tuple(add_var_kwargs['shape'])
            if 'val' in add_var_kwargs:
                v = add_var_kwargs['val']
                if isinstance(v, float):
                    self.varshape = ()
                elif isinstance(v, np.ndarray):
                    self.varshape = v.shape
                else:
                    raise TypeError(f"ExplicitSleepComp doesn't work with discrete variables.")

    def setup(self):
        self.inames = []
        self.onames = []
        add_var_kwargs = self.add_var_kwargs if self.add_var_kwargs else {}
        for i in range(self.nvars):
            self.inames.append(f'i{i}')
            self.onames.append(f'o{i}')
            self.add_input(self.inames[-1], **add_var_kwargs)
            self.add_output(self.onames[-1], **add_var_kwargs)
            self.declare_partials('*', '*', method='cs')

        if self.use_coloring:
            self.declare_coloring(wrt='*', method='cs', perturb_size=1e-5,
                                  num_full_jacs=2, tol=1e-20)

    def compute(self, inputs, outputs):
        if debug: print(f"running {self.pathname}")
        self.sleep(self.compute_delay)
        outputs.set_val(inputs.asarray())

    def compute_partials(self, inputs, partials):
        if debug: print(f"compute partials for {self.pathname}")
        self.sleep(self.compute_partials_delay)
        val = np.eye(np.prod(self.varshape, dtype=int))
        for iname, oname in zip(self.inames, self.onames):
            partials[oname, iname] = val

    def sleep(self, delay):
        time.sleep(delay / self.comm.size)


# A few solver classes that will just run to whatever maxiter is set to
class FixedIterNLBGS(om.NonlinearBlockGS):
    def _iter_get_norm(self):
        return 1.0


class FixedIterLinearBGS(om.LinearBlockGS):
    def _iter_get_norm(self):
        return 1.0


class FixedIterNLJac(om.NonlinearBlockJac):
    def _iter_get_norm(self):
        return 1.0


class FixedIterLinBlockJac(om.LinearBlockJac):
    def _iter_get_norm(self):
        return 1.0


def expand_kwargs(**kwargs):
    # this expects all entries of kwargs to be iterators of the same length.
    # It yields one kwargs dict for each iter of all kwargs entries,
    # e.g., if kwargs is {'a': [1,2,3], 'b': [4,5,6]}, it yields
    # {'a': 1, 'b':4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}
    if not kwargs:
        return

    lstkwargs = {k: list(v) for k, v in kwargs.items()}
    for v in lstkwargs.values():
        nitems = len(v)
        break

    try:
        for idx in range(nitems):
            yield {k: v[idx] for k, v in lstkwargs.items()}
    except IndexError:
        raise IndexError("expand_kwargs() requires that all kwargs entries have the same number of "
                         "items.")


def dup_inst(n, klass, *kwargs):
    # yield n duplicate instances of klass constructed using kwargs
    for i in range(n):
        yield klass(**kwargs)


def inst_iter(class_iter, kwargs_iter):
    # yields an instance of klass for each kwargs in kwargs_iter
    for klass, kwargs in zip(class_iter, kwargs_iter):
        yield klass(**kwargs)


def sys_var_path_iter(ncomps, nvars, path=''):
    # yield system_path, input_path, output_path for each input/output pair using
    # the default naming scheme (Component is 'C?', inputs are 'i?' and outputs are 'o?').
    if path:
        path = path + '.'
    for c in range(ncomps):
        for v in range(nvars):
            cpath = f"{path}C{c}"
            yield cpath, f"{cpath}.i{v}", f"{cpath}.o{v}"


def group_from_iter(par, sysiter, proc_groups=None, max_procs=None, nliters=1, liniters=1):
    # return a group (ParallelGroup or Group based on 'par' arg) that contains
    # system instances from sysiter.  The group
    # will have (block jac/gs depending on 'par') linear and nl solvers with
    # maxiters of 'liniters' and 'nliters' respectively.
    g = om.ParallelGroup() if par else om.Group()
    if proc_groups is None:
        proc_groups = repeat(None)
    if max_procs is None:
        max_procs = repeat(None)
    for i, (inst, pgrp, mx) in enumerate(zip(sysiter, proc_groups, max_procs)):
        g.add_subsystem(f"C{i}", inst, proc_group=pgrp, max_procs=mx)

    if par:
        g.nonlinear_solver = FixedIterNLJac(maxiter=nliters)
        g.linear_solver = FixedIterLinBlockJac(maxiter=liniters)
    else:
        g.nonlinear_solver = FixedIterNLBGS(maxiter=nliters)
        g.linear_solver = FixedIterLinearBGS(maxiter=liniters)

    return g


def make_group(ncomps, nvars, delays, proc_groups, max_procs, nliters=2, liniters=1):
    compiter = inst_iter(repeat(ExplicitSleepComp, ncomps),
                         expand_kwargs(nvars=repeat(nvars, ncomps),
                                       use_coloring=repeat(True, ncomps),
                                       compute_delay=delays))
    return group_from_iter(True, compiter, proc_groups=proc_groups, max_procs=max_procs,
                           nliters=2, liniters=1)


if __name__ == '__main__':
    import sys
    from openmdao.utils.assert_utils import assert_check_totals
    from time import perf_counter

    start = perf_counter()

    nvars = 10
    ncomps = 10
    nruns = 1

    if 'eq' in sys.argv:
        delays=[.136]*ncomps
        proc_groups=[None]*ncomps
        max_procs=[None]*ncomps
        print("Running equal runtime components under parallel group")
    else:
        delays=[.01]*6 + [.1,.1,.1, 1.]
        proc_groups=['a']*9 + ['b']
        max_procs=[1]*9 + [None]
        print("Running mixed runtime components under parallel group")

    p = om.Problem()
    p.model.add_subsystem('par', make_group(ncomps, nvars,
                                            delays=delays, proc_groups=proc_groups,
                                            max_procs=max_procs))

    p.setup()

    with om.timing_context():
        for i in range(nruns):
            p.run_model()

    print("Total time:", perf_counter() - start)
