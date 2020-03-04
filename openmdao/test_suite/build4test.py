"""
Various functions to make it easier to build test models.
"""

import time
import numpy

from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.test_suite.components.exec_comp_for_test import ExecComp4Test

from openmdao.utils.mpi import MPI


class DynComp(ExplicitComponent):
    """
    A component with a settable number of params and outputs.
    """
    def __init__(self, ninputs, noutputs,
                 nl_sleep=0.001, ln_sleep=0.001,
                 var_factory=float, vf_args=()):
        super(DynComp, self).__init__()

        self.ninputs = ninputs
        self.noutputs = noutputs
        self.var_factory = var_factory
        self.vf_args = vf_args
        self.nl_sleep = nl_sleep
        self.ln_sleep = ln_sleep

    def setup(self):
        for i in range(self.ninputs):
            self.add_input('i%d'%i, self.var_factory(*self.vf_args))

        for i in range(self.noutputs):
            self.add_output("o%d"%i, self.var_factory(*self.vf_args))

    def compute(self, inputs, outputs):
        time.sleep(self.nl_sleep)

    def compute_partials(self, inputs, partials):
        """
        Jacobian for Sellar discipline 1.
        """
        time.sleep(self.ln_sleep)


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
                            var_factory=lambda: numpy.zeros(vec_size))
            cname = "C%d"%(num_comps-1)
            self.add_objective("%s.o0" % cname)
            self.add_constraint("%s.o1" % cname, lower=0.0)


    p = Problem()
    g = p.model

    if 'gmres' in sys.argv:
        from openmdao.solvers.linear.scipy_iter_solver import ScipyKrylov
        g.linear_solver = ScipyKrylov()

    g.add_subsystem("P", IndepVarComp('x', numpy.ones(vec_size)))

    g.add_design_var("P.x")

    par = g.add_subsystem("par", ParallelGroup())
    for pt in range(pts):
        ptname = "G%d"%pt
        ptg = par.add_subsystem(ptname, SubGroup())
        #create_dyncomps(ptg, num_comps, 2, 2, 2,
                            #var_factory=lambda: numpy.zeros(vec_size))
        g.connect("P.x", "par.%s.C0.i0" % ptname)

        #cname = ptname + '.' + "C%d"%(num_comps-1)
        #g.add_objective("par.%s.o0" % cname)
        #g.add_constraint("par.%s.o1" % cname, lower=0.0)

    p.setup()
    p.final_setup()
    p.run_model()
    #
    from openmdao.devtools.memory import max_mem_usage
    print("mem:", max_mem_usage())

    config_summary(p)
