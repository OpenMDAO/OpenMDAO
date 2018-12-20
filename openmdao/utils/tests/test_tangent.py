import sys
import os
import unittest

import numpy as np

import tangent

from openmdao.core.driver import Driver
from openmdao.api import Problem, IndepVarComp, \
    ExecComp, Group, ImplicitComponent, ExplicitComponent,ParallelGroup, BroydenSolver
from openmdao.vectors.vector import set_vec
from openmdao.devtools.debug import compute_approx_jac
from openmdao.utils.ad_tangent import _get_tangent_ad_func, _get_tangent_ad_jac, check_ad

from openmdao.utils.assert_utils import assert_rel_error

class Passthrough(ExplicitComponent):

    def __init__(self, size, *args, **kwargs):
        self.size = size
        super(Passthrough, self).__init__(*args, **kwargs)

    def setup(self):
        self.add_input('a', np.ones(self.size))
        self.add_input('b', np.ones(self.size))
        self.add_input('c', np.ones(self.size))

        self.add_output('x', np.ones(self.size))
        self.add_output('y', np.ones(self.size))
        self.add_output('z', np.ones(self.size))

        rel2meta = self._var_rel2meta

        for in_name, out_name in zip(self._var_rel_names['input'], self._var_rel_names['output']):

            meta = rel2meta[in_name]
            shape = meta['shape']
            size = np.prod(shape)
            row_col = np.arange(size, dtype=int)

            self.declare_partials(of=out_name, wrt=in_name,
                                  val=np.ones(size), rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        outputs = set_vec(inputs, outputs)


def get_harness(comp, name='comp', top=False):
    if top:
        p = Problem(comp)
    else:
        p = Problem()
        p.model.add_subsystem(name, comp)

    p.setup()
    return p, comp


class TangentTestCase(unittest.TestCase):

    def test_set_vec(self):
        p, comp = get_harness(Passthrough(size=5))
        p['comp.a'] = np.random.random(comp.size) + 1.0
        p['comp.b'] = np.random.random(comp.size) + 1.0
        p['comp.c'] = np.random.random(comp.size) + 1.0
        p.run_model()
        check_ad(comp)

    def test_optimize(self):
        self.fail("not tested")

    def test_aug_assign(self):
        self.fail("not tested")

    def test_slice_rhs(self):
        self.fail("not tested")

    def test_slice_lhs(self):
        self.fail("not tested")

    def test_call_on_nonfunc(self):
        # test when an instance with a __call__ method is called like a function
        self.fail("not tested")

    def test_attribute(self):
        # test handling of self.* attributes within the AD'd function
        self.fail("not tested")

    def test_dynamic_loop(self):
        self.fail("not tested")

    def test_subfunction(self):
        self.fail("not tested")

    def test_submethod(self):
        self.fail("not tested")

if __name__ == '__main__':
    unittest.main()
