import unittest
import os
import importlib
import inspect
from os.path import dirname, abspath, splitext

from openmdao.utils.entry_points import list_installed, _filtered_ep_iter, _allowed_types, \
    compute_entry_points, _epgroup_bases
from openmdao.utils.file_utils import files_iter

from openmdao.api import Group, SurrogateModel
from openmdao.core.component import Component
from openmdao.core.driver import Driver
from openmdao.solvers.solver import LinearSolver, NonlinearSolver
from openmdao.recorders.case_recorder import CaseRecorder
from openmdao.recorders.base_case_reader import BaseCaseReader


_ep_bases = tuple(_epgroup_bases)


def _is_ep_class(c):
    try:
        return issubclass(c, _ep_bases)
    except:
        return False


class TestEntryPoints(unittest.TestCase):

    def test_list_installed(self):
        dct = list_installed()

    # test that all entry points are loadable and result in the correct type
    def test_ep_load(self):
        pass

    # test if all relevant classes have been registered as entry points
    def test_ep_registered(self):
        skip = set(['openmdao.surrogate_models.surrogate_model:MultiFiSurrogateModel'])

        # if mpi4py isn't installed, then the pyopstsparse_driver import will fail
        try:
            import mpi4py
            from pyoptsparse import Optimization
        except ImportError:
            skip.add('openmdao.drivers.pyoptsparse_driver:pyOptSparseDriver')

        # collect declared entry points for openmdao
        registered_eps = {}
        for epgroup in _allowed_types.values():
            registered_eps[epgroup] = reg = set()
            for ep, name, module, target in _filtered_ep_iter(epgroup, includes=['openmdao']):
                reg.add(str(ep).split('=', 1)[1].strip())

        found_eps = compute_entry_points('openmdao',
                                         dir_excludes=('test_suite', 'docs', 'docs_experiment'),
                                         outstream=None)

        for epgroup in _allowed_types.values():
            reg = registered_eps.get(epgroup, set())
            found = [f.split('=', 1)[1].strip() for f in found_eps.get(epgroup, [])]
            # exclude any private classes
            found = set(f for f in found if not f.rsplit(':', 1)[-1].startswith('_'))

            missing = sorted(found - reg - skip)
            extra = sorted(reg - found - skip)
            if missing:
                self.fail("For entry point group '{}', the following EPs are missing: {}.".format(epgroup, missing))
            if extra:
                self.fail("For entry point group '{}', the following extra EPs were found: {}.".format(epgroup, extra))

if __name__ == "__main__":
    unittest.main()
