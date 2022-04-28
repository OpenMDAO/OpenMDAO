
import unittest
import time

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_check_partials, \
    assert_check_totals
from openmdao.test_suite.components.paraboloid import Paraboloid


class MyParaboloid(Paraboloid):
    """ Use matrix-vector product."""

    def setup_partials(self):
        pass

    def compute_partials(self, inputs, partials):
        """Analytical derivatives."""
        pass

    def compute_full_jacvec_product(self, inputs, dinputs, doutputs, mode):
        x = inputs['x'][0]
        y = inputs['y'][0]

        print("computing full JVP, dinput rels:", list(dinputs._names), "doutput rels", list(doutputs._names))

        if mode == 'fwd':
            return np.array([(2.0*x - 6.0 + y)*dinputs['x'][0] + (2.0*y + 8.0 + x)*dinputs['y'][0]])
        else:  # rev
            return np.array([(2.0*x - 6.0 + y)*doutputs['f_xy'][0], (2.0*y + 8.0 + x)*doutputs['f_xy'][0]])

    def compute_jacvec_product(self, inputs, dinputs, doutputs, mode):
        full = self.get_full_jacvec_product(inputs, dinputs, doutputs, mode)

        if mode == 'fwd':
            # if 'x' in dinputs:
            #     doutputs['f_xy'] += full['f_xy']
            # if 'y' in dinputs:
            #     doutputs['f_xy'] += full['f_xy']
            doutputs.iadd(full)

        elif mode == 'rev':
            # if 'x' in dinputs:
            #     dinputs['x'] += full['x']
            # if 'y' in dinputs:
            #     dinputs['y'] += full['y']
            dinputs.iadd(full)


def execute_model(mode):
    prob = om.Problem()
    model = prob.model

    model.add_subsystem('indeps', om.IndepVarComp('dv1', val=1.0))

    sub1 = model.add_subsystem('sub1', om.Group())
    sub1.add_subsystem('c1', om.ExecComp(exprs=['y = x']))

    sub2 = sub1.add_subsystem('sub2', om.Group())
    sub2.add_subsystem('comp', MyParaboloid())

    model.connect('indeps.dv1', ['sub1.c1.x', 'sub1.sub2.comp.x'])
    sub1.connect('c1.y', 'sub2.comp.y')

    model.add_design_var('indeps.dv1')
    model.add_constraint('sub1.sub2.comp.f_xy')

    prob.setup(mode=mode, force_alloc_complex=True)
    # prob.set_solver_print(level=0)

    prob['indeps.dv1'] = 2.

    prob.run_model()
    assert_check_partials(prob.check_partials(method='cs'))
    totals = prob.compute_totals()
    assert_check_totals(prob.check_totals(method='cs', out_stream=None))


class TestLinOpCaching(unittest.TestCase):

    def test_matrix_free_explicit_fwd(self):
        execute_model('fwd')

    def test_matrix_free_explicit_rev(self):
        execute_model('rev')

if __name__ == '__main__':
    execute_model('fwd')