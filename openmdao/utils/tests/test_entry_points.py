import unittest
import importlib
import types

from openmdao.utils.entry_points import list_installed, _filtered_ep_iter, _allowed_types, \
    compute_entry_points, _epgroup_bases, split_ep
from openmdao.utils.assert_utils import assert_no_warning

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
                reg.add(f"{module}:{target}")

        found_eps = compute_entry_points('openmdao',
                                         dir_excludes=('test_suite', 'docs'),
                                         outstream=None)

        for epgroup in _allowed_types.values():
            if epgroup == 'openmdao_report':
                continue
            reg = registered_eps.get(epgroup, set())
            found = [':'.join(split_ep(f)[1:]) for f in found_eps.get(epgroup, [])]
            # exclude any private classes
            found = set(f for f in found if not f.rsplit(':', 1)[-1].startswith('_'))

            missing = sorted(found - reg - skip)
            extra = sorted(reg - found - skip)
            if missing:
                self.fail(f"For entry point group '{epgroup}', the following EPs are missing: {sorted(missing)}.")
            if extra:
                self.fail(f"For entry point group '{epgroup}', the following extra EPs were found: {sorted(extra)}.")

        # check that all registered reports point to actual functions
        badep = set()
        for fullpath in registered_eps.get('openmdao_report', ()):
            modpath, _, funcname = fullpath.partition(':')
            try:
                mod = importlib.import_module(modpath)
            except ImportError:
                badep.add(fullpath)
                continue

            f = getattr(mod, funcname, None)
            if f is None or not isinstance(f, types.FunctionType):
                badep.add(fullpath)

        if badep:
            self.fail("For entry point group 'openmdao_report', the following EPs either couldn't "
                      f"be found or are not functions: {sorted(badep)}.")


class TestEntryPointsWarning(unittest.TestCase):
    ISOLATED = True

    def test_ep_warn(self):
        with assert_no_warning(Warning):
            list_installed()


if __name__ == "__main__":
    unittest.main()
