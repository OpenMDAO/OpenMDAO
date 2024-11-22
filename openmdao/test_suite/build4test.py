"""
Various functions to make it easier to build test models.
"""

import time
import numpy as np

from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.utils.array_utils import evenly_distrib_idxs


class FloatFactory(object):
    def __init__(self, shape=()):
        self.shape = shape
        self.size = np.prod(shape) if shape else 1

    def __call__(self, iotype, idx):
        if self.shape:
            if iotype == 'input':
                return np.ones(self.shape)
            else:
                return np.zeros(self.shape)
        else:
            if iotype == 'input':
                return float(1.0)
            else:
                return float(0.0)


class DynComp(ExplicitComponent):
    """
    A component with a settable number of params and outputs.
    """
    def __init__(self, ninputs, noutputs,
                 nl_sleep=0.001, ln_sleep=0.001,
                 var_factory=float, vf_args=()):
        super().__init__()

        self.ninputs = ninputs
        self.noutputs = noutputs
        self.var_factory = var_factory
        self.vf_args = vf_args
        self.nl_sleep = nl_sleep
        self.ln_sleep = ln_sleep

    def setup(self):
        for i in range(self.ninputs):
            self.add_input(f'i{i}', self.var_factory(*self.vf_args))

        for i in range(self.noutputs):
            self.add_output(f'o{i}', self.var_factory(*self.vf_args))

    def compute(self, inputs, outputs):
        time.sleep(self.nl_sleep)

    def compute_partials(self, inputs, partials):
        """
        Jacobian for Sellar discipline 1.
        """
        time.sleep(self.ln_sleep)


class DynComp2(ExplicitComponent):
    """
    A component with a settable number of params and outputs.
    """
    def __init__(self, ninputs, noutputs, mult, var_factory):
        super().__init__()

        self.ninputs = ninputs
        self.noutputs = noutputs
        self.mult = mult
        self.var_factory = var_factory

    def setup(self):
        for i in range(self.ninputs):
            self.add_input(f'i{i}', self.var_factory('input', i))

        for i in range(self.noutputs):
            self.add_output(f'o{i}', self.var_factory('output', i))

        self.declare_partials('*', '*',
                              rows=np.arange(self.var_factory.size),
                              cols=np.arange(self.var_factory.size))

    def compute(self, inputs, outputs):
        outputs.set_val(inputs.asarray() * self.mult)

    def compute_partials(self, inputs, partials):
        for o in range(self.noutputs):
            for i in range(self.ninputs):
                if o == i:
                    partials[f'o{o}', f'i{i}'] = self.mult


class DynComp2Factory(object):
    def __init__(self, ninputs=10, noutputs=10, mult=1.1, var_factory=FloatFactory()):
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.mult = mult
        self.var_factory = var_factory

    def __call__(self):
        return DynComp2(self.ninputs, self.noutputs, self.mult, self.var_factory)



def make_subtree(parent, nsubgroups, levels,
                 ncomps, ninputs, noutputs, nconns, var_factory=float):
    """Construct a system subtree under the given parent group."""

    if levels <= 0:
        return

    if levels == 1:  # add leaf nodes
        create_dyncomps(parent, ncomps, ninputs, noutputs, nconns,
                        var_factory=var_factory)
    else:  # add more subgroup levels
        for i in range(nsubgroups):
            g = parent.add_subsystem("G%d"%i, Group())
            make_subtree(g, nsubgroups, levels-1,
                         ncomps, ninputs, noutputs, nconns,
                         var_factory=var_factory)


def create_dyncomps(parent, ncomps, ninputs, noutputs, nconns,
                    var_factory=float):
    """Create a specified number of DynComps with a specified number
    of variables (ninputs and noutputs), and add them to the given parent
    and add the number of specified connections.
    """
    for i in range(ncomps):
        parent.add_subsystem("C%d" % i, DynComp(ninputs, noutputs, var_factory=var_factory))

        if i > 0:
            for j in range(nconns):
                parent.connect("C%d.o%d" % (i-1,j), "C%d.i%d" % (i, j))


def create_testcomps(ncomps, comp_factory):
    """
    Yield ncomps instances returned from the given component factory.
    """
    for i in range(ncomps):
        yield f"C{i}", comp_factory()


def connect_comps(model, complist, ninputs, noutputs):
    """
    Connect the components in complist.
    """
    for i in range(len(complist)):
        if i > 0:
            for j in range(min(noutputs, ninputs)):
                model.connect(f"{complist[i-1][1]}.o{j}", f"{complist[i][1]}.i{j}")


def recursive_split(parent, complist, nsplits, max_levels, level=0, path=None):
    """
    Add levels to the system tree while splitting up the current component list.

    Parameters
    ----------
    parent : Group
        The parent group.
    complist : list
        List of components to split up.
    nsplits : int
        Number of splits to make.
    max_levels : int
        Maximum number of levels in the system tree.
    level : int
        Current level in the system tree.
    path : list or None
        List of names of parents in the system tree.
    """
    if path is None:
        path = []

    sizes, offsets = evenly_distrib_idxs(nsplits, len(complist))
    subs = [complist[offset:offset + size] for size, offset in zip(sizes, offsets)]
    for subcomps in subs:
        if not subcomps:
            continue

        g = Group()
        gname = subcomps[0][0].replace('C', f'G_{level}_')
        parent.add_subsystem(gname, g)

        # if leaf node, add comps
        if level + 1 >= max_levels or len(subcomps) < nsplits:
            for i, subcomp in enumerate(subcomps):
                cname, _, comp = subcomp
                # set the pathname for connections later
                subcomp[1] = '.'.join(path + [gname, cname])
                g.add_subsystem(cname, comp)
        else:
            recursive_split(g, subcomps, nsplits, max_levels, level + 1, path + [gname])


def build_test_model(ncomps, ninputs, noutputs, splits_per_group, max_levels, shape):
    """
    Create a test problem that has design vars and responses and can compute derivatives.

    Parameters
    ----------
    ncomps : int
        Number of components in the model.
    ninputs : int
        Number of inputs to each component.
    noutputs : int
        Number of outputs from each component.
    splits_per_group : int
        Each Group has sub-Groups added based on the number of splits of the current list
        of components.
    max_levels : int
        Maximum number of levels in the system tree.
    shape : tuple
        Shape of the input and output variables.

    Returns
    -------
    Group
        The model.
    """
    model = Group()
    complist = [[name, None, comp] for name, comp in
                create_testcomps(ncomps, DynComp2Factory(ninputs, noutputs, 1.1,
                                                         FloatFactory(shape)))]
    recursive_split(model, complist, splits_per_group, max_levels)
    connect_comps(model, complist, ninputs, noutputs)
    for i in range(ninputs):
        model.add_design_var(f"{complist[0][1]}.i{i}")
    if noutputs > 0:
        model.add_objective(f"{complist[-1][1]}.o0", index=0, flat_indices=True)
    if noutputs > 1:
        for i in range(1, noutputs):
            model.add_constraint(f"{complist[-1][1]}.o{i}", lower=0.0)
    return model


if __name__ == '__main__':
    import sys
    from openmdao.core.problem import Problem
    from openmdao.devtools.debug import config_summary

    vec_size = 1000
    num_comps = 50
    pts = 2


    class SubGroup(Group):
        def setup(self):
            create_dyncomps(self, num_comps, 2, 2, 2,
                            var_factory=lambda: np.zeros(vec_size))
            cname = "C%d"%(num_comps-1)
            self.add_objective("%s.o0" % cname)
            self.add_constraint("%s.o1" % cname, lower=0.0)


    p = Problem()
    g = p.model

    if 'gmres' in sys.argv:
        from openmdao.solvers.linear.scipy_iter_solver import ScipyKrylov
        g.linear_solver = ScipyKrylov()

    g.add_subsystem("P", IndepVarComp('x', np.ones(vec_size)))

    g.add_design_var("P.x")

    par = g.add_subsystem("par", ParallelGroup())
    for pt in range(pts):
        ptname = "G%d"%pt
        ptg = par.add_subsystem(ptname, SubGroup())
        g.connect("P.x", "par.%s.C0.i0" % ptname)

    p.setup()
    p.final_setup()
    p.run_model()
    #
    from openmdao.devtools.memory import max_mem_usage
    print("mem:", max_mem_usage())

    config_summary(p)
