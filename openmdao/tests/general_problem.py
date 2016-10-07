from __future__ import division
import numpy
import unittest

from openmdao.api import Problem, IndepVarComponent, ExplicitComponent, Group, PETScVector



class GeneralComp(ExplicitComponent):

    def initialize_variables(self):
        kwargs = self.global_kwargs
        icomp = kwargs['icomp']
        ncomp = kwargs['ncomp']
        use_var_sets = kwargs['use_var_sets']

        for ind in xrange(ncomp):
            if use_var_sets:
                var_set = ind
            else:
                var_set = 0

            if ind is not icomp:
                self.add_input('v%i' % ind)
            else:
                self.add_output('v%i' % ind, var_set=var_set)



class GeneralProblem(object):

    def __init__(self, ngroup_level, use_var_sets=False, parallel_groups=False):
        self.ncomp = ngroup_level[0]
        self.ngroup_level = ngroup_level[1:]

        ncomp = self.ncomp
        ngroup_level = self.ngroup_level
        nlevel = len(self.ngroup_level)

        ilevel = 0
        all_systems = []

        current_systems = []
        for icomp in xrange(ncomp):
            comp = GeneralComp('Comp-%i-%i' % (nlevel-ilevel, icomp),
                               icomp=icomp, ncomp=ncomp, promotes_all=True)
            current_systems.append(comp)
        all_systems.extend(current_systems[::-1])
        ilevel += 1

        for ngroup in ngroup_level:
            nsub = len(current_systems)
            nsub_group = int(numpy.floor(nsub/ngroup)) * numpy.ones(ngroup, int)
            nsub_group[-1] += nsub - numpy.sum(nsub_group)
            next_systems = []
            for igroup in xrange(ngroup):
                group = Group('Group-%i-%i' % (nlevel-ilevel, igroup))
                group._mpi_proc_allocator.parallel = parallel_groups
                ind1 = numpy.sum(nsub_group[:igroup])
                ind2 = numpy.sum(nsub_group[:igroup+1])
                for ind in xrange(ind1, ind2):
                    group.add_subsystem(current_systems[ind])
                next_systems.append(group)
            current_systems = next_systems
            all_systems.extend(current_systems[::-1])
            ilevel += 1

        self.root = current_systems[0]
        self.all_systems = all_systems

        self.root.kwargs['use_var_sets'] = use_var_sets
        self.problem = Problem(self.root).setup(PETScVector)

    def print_all(self):
        for sys in self.all_systems[::-1]:
            print sys.name, ':',
            for subsys in sys._subsystems_allprocs:
                print subsys.name,
            print
        print

if __name__ == '__main__':

    ngroup_level = [4,2,1]
    gp = GeneralProblem(ngroup_level)
    gp.print_all()
