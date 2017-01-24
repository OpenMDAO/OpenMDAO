from __future__ import division, print_function
import numpy
import unittest

from openmdao.api import Problem, IndepVarComp, ExplicitComponent, Group, DefaultVector
try:
    from openmdao.parallel_api import PETScVector
    vec_impl = PETScVector
except ImportError:
    vec_impl = DefaultVector


class GeneralComp(ExplicitComponent):

    def initialize_variables(self):
        kwargs = self.metadata
        icomp = kwargs['icomp']
        ncomp = kwargs['ncomp']
        use_var_sets = kwargs['use_var_sets']

        for ind in range(ncomp):
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
        current_sysnames = []
        for icomp in range(ncomp):
            comp = GeneralComp(icomp=icomp, ncomp=ncomp)
            current_systems.append(comp)
            current_sysnames.append('Comp-%i-%i' % (nlevel-ilevel, icomp))
        all_systems.extend(current_systems[::-1])
        ilevel += 1

        for ngroup in ngroup_level:
            nsub = len(current_systems)
            nsub_group = int(numpy.floor(nsub/ngroup)) * numpy.ones(ngroup, int)
            nsub_group[-1] += nsub - numpy.sum(nsub_group)
            next_systems = []
            next_sysnames = []
            for igroup in range(ngroup):
                group = Group()
                group._mpi_proc_allocator.parallel = parallel_groups
                ind1 = numpy.sum(nsub_group[:igroup])
                ind2 = numpy.sum(nsub_group[:igroup + 1])
                for ind in range(ind1, ind2):
                    if isinstance(current_systems[ind], Group):
                        promotes = None
                    else:
                        promotes = ['*']
                    group.add_subsystem(current_sysnames[ind], current_systems[ind],
                                        promotes=promotes)
                next_systems.append(group)
                next_sysnames.append('Group-%i-%i' % (nlevel-ilevel, igroup))
            current_systems = next_systems
            current_sysnames = next_sysnames
            all_systems.extend(current_systems[::-1])
            ilevel += 1

        self.model = current_systems[0]
        self.all_systems = all_systems

        self.model.metadata['use_var_sets'] = use_var_sets
        self.problem = Problem(self.model).setup(vec_impl)

    def print_all(self):
        for sys in self.all_systems[::-1]:
            print(sys.name, ':', end=' ')
            for subsys in sys._subsystems_allprocs:
                print(subsys.name, end=' ')
            print()
        print()

if __name__ == '__main__':

    ngroup_level = [4,2,1]
    gp = GeneralProblem(ngroup_level)
    gp.print_all()
