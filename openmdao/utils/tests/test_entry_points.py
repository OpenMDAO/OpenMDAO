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
        skip = {  # mostly deprecated stuff
            'openmdao.components.external_code_comp:ExternalCode',
            'openmdao.components.ks_comp:KSComponent',
            'openmdao.components.meta_model_structured_comp:MetaModelStructured',
            'openmdao.components.meta_model_unstructured_comp:MetaModel',
            'openmdao.components.meta_model_unstructured_comp:MetaModelUnStructured',
            'openmdao.components.multifi_meta_model_unstructured_comp:MultiFiMetaModel',
            'openmdao.components.multifi_meta_model_unstructured_comp:MultiFiMetaModelUnStructured',
            'openmdao.solvers.linear.petsc_ksp:PetscKSP',
            'openmdao.solvers.linear.scipy_iter_solver:ScipyIterativeSolver',
            'openmdao.drivers.scipy_optimizer:ScipyOptimizer',
            'openmdao.solvers.nonlinear.nonlinear_runonce:NonLinearRunOnce',
            'openmdao.surrogate_models.surrogate_model:MultiFiSurrogateModel',
            'openmdao.surrogate_models.kriging:FloatKrigingSurrogate',
        }
        # collect declared entry points for openmdao
        registered_eps = {}
        for epgroup in _allowed_types.values():
            registered_eps[epgroup] = reg = set()
            for ep, name, module, target in _filtered_ep_iter(epgroup, includes=['openmdao']):
                reg.add(str(ep).split('=', 1)[1].strip())

        found_eps = compute_entry_points('openmdao', outstream=None)

        for epgroup in _allowed_types.values():
            reg = registered_eps.get(epgroup, set())
            found = set(f.split('=', 1)[1].strip() for f in found_eps.get(epgroup, []))

            missing = sorted(found - reg - skip)
            extra = sorted(reg - found)
            if missing: 
                self.fail("For entry point group '{}', the following EPs are missing: {}.".format(epgroup, missing))
            if extra:
                self.fail("For entry point group '{}', the following extra EPs were found: {}.".format(epgroup, extra))

if __name__ == "__main__":
    unittest.main()
