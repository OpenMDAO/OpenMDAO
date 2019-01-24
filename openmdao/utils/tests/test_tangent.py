import sys
import os
import unittest

import numpy as np

import tangent

from openmdao.api import Problem, IndepVarComp, \
    ExecComp, Group, ImplicitComponent, ExplicitComponent
from openmdao.vectors.vector import set_vec
from openmdao.devtools.debug import compute_approx_jac
from openmdao.utils.ad_tangent import _get_tangent_ad_func, _get_tangent_ad_jac, check_tangent_ad

from openmdao.utils.assert_utils import assert_rel_error

class PassThrough(ExplicitComponent):

    def __init__(self, size, *args, **kwargs):
        self.size = size
        super(PassThrough, self).__init__(*args, **kwargs)

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


class Looper(ExplicitComponent):

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        Options are declared here because this class is intended to be subclassed by
        the end user. The `initialize` method is left available for user-defined options.
        """
        super(Looper, self)._declare_options()
        self.options.declare('names', [])

    def setup(self):

        size = 3
        self.options['names'] = ['a', 'b', 'c', 'd']

        # primary inputs and outputs
        for n in self.options['names']:
            self.add_input(n + '_in', val=np.ones(size))
            self.add_output(n + '_out', val=np.zeros(size))

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):

        insum = 0.0
        names = self.options['names']
        for n in names:
            insum = insum + inputs[n + '_in']

        for n in names:
            outputs[n + '_out'] = insum + 3.0 * inputs[n + '_in']


class ForCond(ExplicitComponent):
    def setup(self):
        self.n = 5
        self.add_input('x', np.ones(self.n))
        self.add_output('y', np.zeros(self.n))

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        n = self.n
        y = outputs['y']
        x = inputs['x']
        for i in range(n):
            if i == 0:
                y[i] = np.sin(x[i])
            else:
                y[i] = x[i] * 3.


class PassThroughOptBug(ExplicitComponent):
    def setup(self):
        self.size = size = 3
        self.i_var = 'x'
        self.o_var = 'y'
        self.add_input(self.i_var, np.ones(size))
        self.add_output(self.o_var, np.ones(size))

        row_col = np.arange(size)
        self.declare_partials(of=self.o_var, wrt=self.i_var,
                              val=np.ones(size), rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        outputs[self.o_var] = inputs[self.i_var]

    def compute_partials(self, inputs, J):
        pass


def get_harness(comp, name='comp', top=False):
    if top:
        p = Problem(comp)
    else:
        p = Problem()
        p.model.add_subsystem(name, comp)

    p.setup()
    p.final_setup()
    return p, comp


class TangentTestCase(unittest.TestCase):

    def test_set_vec(self):
        p, comp = get_harness(PassThrough(size=5))
        p['comp.a'] = np.random.random(comp.size) + 1.0
        p['comp.b'] = np.random.random(comp.size) + 1.0
        p['comp.c'] = np.random.random(comp.size) + 1.0
        p.run_model()
        check_tangent_ad(comp)

    def test_optimize_req_key_fwd(self):
        # make sure tangent doesn't optimize away variables needed as keys in __getitem__ calls
        p, comp = get_harness(PassThroughOptBug())
        p['comp.x'] = np.random.random(comp.size)
        p.run_model()
        check_tangent_ad(comp, mode='fwd', verbose=0)

    def test_optimize_req_key_rev(self):
        # make sure tangent doesn't optimize away variables needed as keys in __getitem__ calls
        p, comp = get_harness(PassThroughOptBug())
        p['comp.x'] = np.random.random(comp.size)
        p.run_model()
        check_tangent_ad(comp, mode='rev', verbose=0)

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

    def test_dynamic_loop_fwd(self):
        p, comp = get_harness(Looper())
        comp._inputs._data[:] = np.random.random(comp._inputs._data.size)
        p.run_model()
        check_tangent_ad(comp, mode='fwd')

    def test_dynamic_loop_rev(self):
        p, comp = get_harness(Looper())
        comp._inputs._data[:] = np.random.random(comp._inputs._data.size)
        p.run_model()
        check_tangent_ad(comp, mode='rev')

    def test_loop_with_cond_fwd(self):
        p, comp = get_harness(ForCond())
        comp._inputs._data[:] = np.random.random(comp._inputs._data.size)
        p.run_model()
        check_tangent_ad(comp, verbose=2, mode='fwd')

    def test_loop_with_cond_rev(self):
        p, comp = get_harness(ForCond())
        comp._inputs._data[:] = np.random.random(comp._inputs._data.size)
        p.run_model()
        check_tangent_ad(comp, verbose=2, mode='rev')

    def test_subfunction(self):
        self.fail("not tested")

    def test_submethod(self):
        self.fail("not tested")

if __name__ == '__main__':
    #unittest.main()

    # p, comp = get_harness(PassThroughOptBug())
    # p['comp.x'] = np.random.random(comp.size)
    # p.run_model()
    # check_tangent_ad(comp, mode='rev', verbose=2)

    p, comp = get_harness(ForCond())
    comp._inputs._data[:] = np.random.random(comp._inputs._data.size)
    p.run_model()
    # this test is failing because reaching definition analysis is killing assigns in the if statement
    # based on *earlier* lines, which seems bogus.
    check_tangent_ad(comp, verbose=2, mode='fwd')
