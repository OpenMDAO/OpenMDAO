
import sys
import unittest

from six import assertRaisesRegex, StringIO, assertRegex, iteritems

import numpy as np

import openmdao.api as om
from openmdao.core.system import get_relevant_vars
from openmdao.core.driver import Driver
from openmdao.utils.assert_utils import assert_rel_error, assert_warning
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives


def get_comp(size):
    return om.ExecComp('out = sum(3*x**2) + inp', x=np.ones(size), inp=0.0, out=0.0)


class SubProbComp(om.ExplicitComponent):
    def __init__(self, input_size, num_nodes):
        self.prob = None
        self.size = input_size
        self.num_nodes = num_nodes

    def _setup_subprob(self):
        self.prob = om.Problem(self.comm)
        model = self.prob.model
        model.add_subsystem('comp', get_comp(self.size))

    def setup(self):
        self._setup_subprob()

        self.add_input('x', np.zeros(self.size))
        self.add_input('inp', val=0.0)
        self.add_output('out', val=0.0)

    def compute(self, inputs, outputs):
        p = self.prob
        x = inputs['x']
        p['x'] = x
        for xi in range(x.size):
            inp = inputs['inp']
            for i in range(self.num_nodes):
                p['inp'] = inp
                p.run_model()
                inp = p['out']

        outputs['out'] = p['out']

    def compute_partials(self, inputs, partials):
        pass




class TestPComputeJacvecProd(unittest.TestCase):

    def _build_om_model(self, size):
        p = om.Problem()
        model = p.model
        indep = model.add_subsystem('indep', om.IndepVarComp('x', val=np.zeros(size)))
        indep.add_output('out', val=0.0)

        C1 = model.add_subsystem('C1', get_comp(size))
        C2 = model.add_subsystem('C2', get_comp(size))
        C3 = model.add_subsystem('C3', get_comp(size))

        model.connect('indep.x', ['C1.x', 'C2.x', 'C3.x'])
        model.connect('C1.out', 'C2.inp')
        model.connect('C2.out', 'C3.inp')

        return p

    def test_fwd(self):
        p = self._build_om_model(3)
        p.setup(mode='fwd')
        p['indep.x'] = np.random.random(3)
        p['indep.out'] = np.random.random(1)[0]
        p.run_model()

        J = p.compute_totals(of=['C3.out'], wrt=['indep.x', 'indep.out'], return_format='array')
        print(J)

    def test_rev(self):
        pass

