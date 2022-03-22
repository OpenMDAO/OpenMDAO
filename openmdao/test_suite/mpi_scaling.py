
import time
import numpy as np
import openmdao.api as om
from openmdao.utils.general_utils import shape2tuple


class ExplicitSleepComp(om.ExplicitComponent):
    """
    A component with a settable number of params and outputs and delays.

    Component does a simple passthrough, one variable at a time, so there will be some
    overhead for vector setitem/getitem calls.
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
        print(f"running {self.pathname}")
        time.sleep(self.compute_delay)
        for iname, oname in zip(self.inames, self.onames):
            outputs[oname] = inputs[iname]

    def compute_partials(self, inputs, partials):
        """
        Jacobian for Sellar discipline 1.
        """
        print(f"compute partials for {self.pathname}")
        time.sleep(self.compute_partials_delay)
        val = np.eye(np.product(self.varshape, dtype=int))
        for iname, oname in zip(self.inames, self.onames):
            partials[oname, iname] = val


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


def make_group(par, ncomps, compclass, nliters=1, liniters=1, **kwargs):
    g = om.ParallelGroup() if par else om.Group()
    for i in range(ncomps):
        g.add_subsystem(f"C{i}", compclass(**kwargs))

    if par:
        g.nonlinear_solver = FixedIterNLJac(maxiter=nliters)
        g.linear_solver = FixedIterLinBlockJac(maxiter=nliters)
    else:
        g.nonlinear_solver = FixedIterNLBGS(maxiter=nliters)
        g.linear_solver = FixedIterLinearBGS(maxiter=nliters)

    return g


def sys_var_path_iter(ncomps, nvars, path=''):
    if path:
        path = path + '.'
    for c in range(ncomps):
        for v in range(nvars):
            cpath = f"{path}C{c}"
            yield cpath, f"{cpath}.i{v}", f"{cpath}.o{v}"


if __name__ == '__main__':
    from openmdao.utils.assert_utils import assert_check_totals
    from time import perf_counter

    start = perf_counter()

    nruns = 5
    nvars = 10
    ncomps = 10

    p = om.Problem()
    p.model.add_subsystem('par', make_group(True, ncomps, ExplicitSleepComp, nliters=2, liniters=1,
                                            compute_delay=0.1, nvars=nvars, use_coloring=True))
    p.setup()

    for i in range(nruns):
        p.run_model()

    # ofs = []
    # wrts = []
    # for c, i, o in sys_var_path_iter(ncomps, nvars, 'par'):
    #     ofs.append(o)
    #     wrts.append(i)

    # assert_check_totals(p.check_totals(of=ofs, wrt=wrts, out_stream=None))

    print("Total time:", perf_counter() - start)