import unittest

import openmdao.api as om
from openmdao.utils.relevance import _vars2systems
from openmdao.utils.assert_utils import assert_check_totals


class TestRelevance(unittest.TestCase):
    def test_vars2systems(self):
        names = ['abc.def.g', 'xyz.pdq.bbb', 'aaa.xxx', 'foobar.y']
        expected = {'abc', 'abc.def', 'xyz', 'xyz.pdq', 'aaa', 'foobar', ''}
        self.assertEqual(_vars2systems(names), expected)


class TestDerivsWithoutDVs(unittest.TestCase):
    def test_derivs_with_no_dvs(self):
        # this tests github issue #3037

        class DummyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('a', 1.)
                self.add_output('b', 1.)
            def compute(self, inputs, outputs):
                outputs['b'] = 2. * inputs['a']
            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode=='rev':
                    if 'a' in d_inputs:
                        if 'b' in d_outputs:
                            d_inputs['a'] += d_outputs['b'] * 2.

        prob = om.Problem()
        prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        prob.model.ivc.add_output('a', 1.)
        prob.model.add_subsystem('dummy', DummyComp(), promotes=['*'])

        ### when there's a objective/constraint but no design var, derivatives were zero because the
        ### derivative computation was skipped
        #prob.model.add_design_var('a', lower=0., upper=2.)
        #prob.model.add_constraint('b', lower=3.)
        prob.model.add_objective('b')

        prob.setup(mode='rev')
        prob.run_model()
        chk = prob.check_totals(of='b', wrt='a', show_only_incorrect=True)
        assert_check_totals(chk)
