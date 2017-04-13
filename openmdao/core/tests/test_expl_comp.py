"""Simple example demonstrating how to implement an explicit component."""
from __future__ import division

from six.moves import cStringIO

import unittest

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp
from openmdao.devtools.testutil import assert_rel_error


class TestExplCompSimpleCompute(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('length', val=1.)
        self.add_input('width', val=1.)
        self.add_output('area', val=1.)

    def compute(self, inputs, outputs):
        outputs['area'] = inputs['length'] * inputs['width']


class TestExplCompSimplePartial(TestExplCompSimpleCompute):

    def compute_partial_derivs(self, inputs, outputs, partials):
        partials['area', 'length'] = inputs['width']
        partials['area', 'width'] = inputs['length']


class TestExplCompSimpleJacVec(TestExplCompSimpleCompute):

    def compute_jacvec_product(self, inputs, outputs,
                               d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'area' in d_outputs:
                if 'length' in d_inputs:
                    d_outputs['area'] += inputs['width'] * d_inputs['length']
                if 'width' in d_inputs:
                    d_outputs['area'] += inputs['length'] * d_inputs['width']
        elif mode == 'rev':
            if 'area' in d_outputs:
                if 'length' in d_inputs:
                    d_inputs['length'] += inputs['width'] * d_outputs['area']
                if 'width' in d_inputs:
                    d_inputs['width'] += inputs['length'] * d_outputs['area']


class TestExplCompSimple(unittest.TestCase):

    def test_simple(self):
        comp = TestExplCompSimpleCompute()
        prob = Problem(model=comp)
        prob.setup(check=False)
        prob.run_model()

    def test_compute(self):
        group = Group()
        group.add_subsystem('comp1', IndepVarComp([('length', 1.0), ('width', 1.0)]))
        group.add_subsystem('comp2', TestExplCompSimplePartial())
        group.add_subsystem('comp3', TestExplCompSimpleJacVec())
        group.connect('comp1.length', 'comp2.length')
        group.connect('comp1.width', 'comp2.width')
        group.connect('comp1.length', 'comp3.length')
        group.connect('comp1.width', 'comp3.width')

        prob = Problem(model=group)
        prob.setup(check=False)

        prob['comp1.length'] = 3.
        prob['comp1.width'] = 2.
        prob.run_model()
        assert_rel_error(self, prob['comp2.area'], 6.)
        assert_rel_error(self, prob['comp3.area'], 6.)

        total_derivs = prob.compute_total_derivs(
            wrt=['comp1.length', 'comp1.width'],
            of=['comp2.area', 'comp3.area']
        )
        assert_rel_error(self, total_derivs['comp2.area', 'comp1.length'], 2.)
        assert_rel_error(self, total_derivs['comp3.area', 'comp1.length'], 2.)
        assert_rel_error(self, total_derivs['comp2.area', 'comp1.width'], 3.)
        assert_rel_error(self, total_derivs['comp3.area', 'comp1.width'], 3.)

        # Piggyback testing of list_states

        stream = cStringIO()
        prob.model.list_states(stream=stream)
        content = stream.getvalue()

        self.assertTrue('No states in model' in content)


if __name__ == '__main__':
    unittest.main()
