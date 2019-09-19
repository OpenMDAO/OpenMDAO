
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
    def __init__(self, input_size, num_nodes, mode, **kwargs):
        super(SubProbComp, self).__init__(**kwargs)
        self.prob = None
        self.size = input_size
        self.num_nodes = num_nodes
        self.mode = mode

    def _setup_subprob(self):
        self.prob = p = om.Problem(comm=self.comm)
        model = self.prob.model
        model.add_subsystem('comp', get_comp(self.size))
        p.setup()
        p.final_setup()

    def setup(self):
        self._setup_subprob()

        self.add_input('x', np.zeros(self.size))
        self.add_input('inp', val=0.0)
        self.add_output('out', val=0.0)

    def compute(self, inputs, outputs):
        p = self.prob
        x = inputs['x']
        p['comp.x'] = x
        inp = inputs['inp']
        for i in range(self.num_nodes):
            p['comp.inp'] = inp
            p.run_model()
            inp = p['comp.out']

        outputs['out'] = p['comp.out']

    def compute_partials_fwd(self, inputs, partials):
        p = self.prob
        x = inputs['x']
        p['comp.x'] = x
        p['comp.inp'] = inputs['inp']
        rhs = np.zeros(x.size + 1)
        seed = {'x': rhs[:x.size], 'inp': rhs[x.size:]}
        p.run_model()
        for rhs_i in range(rhs.size):
            rhs[:] = 0.0
            rhs[rhs_i] = 1.0
            for i in range(self.num_nodes):
                inp = p['comp.out']
                jvp = p.compute_jacvec_product(of=['out'], wrt=['x','inp'], 'fwd', seed)
                seed['inp'] = jvp['out']

            if rhs_i < x.size:
                partials['out', 'x'][0, rhs_i] = jvp['out']
            else:
                partials['out', 'inp'][0, 0] = jvp['out']

    def compute_partials_rev(self, inputs, partials):
        pass

    def compute_partials(self, inputs, partials):
        if self.mode == 'fwd':
            self._compute_partials_fwd(inputs, partials)
        else:
            self._compute_partials_rev(inputs, partials)




class TestPComputeJacvecProd(unittest.TestCase):

    def _build_om_model(self, size):
        p = om.Problem()
        model = p.model
        indep = model.add_subsystem('indep', om.IndepVarComp('x', val=np.zeros(size)))
        indep.add_output('inp', val=0.0)

        C1 = model.add_subsystem('C1', get_comp(size))
        C2 = model.add_subsystem('C2', get_comp(size))
        C3 = model.add_subsystem('C3', get_comp(size))

        model.connect('indep.x', ['C1.x', 'C2.x', 'C3.x'])
        model.connect('indep.inp', 'C1.inp')
        model.connect('C1.out', 'C2.inp')
        model.connect('C2.out', 'C3.inp')

        return p

    def test_fwd(self):
        p = self._build_om_model(5)
        p.setup(mode='fwd')
        p['indep.x'] = np.random.random(5)
        p['indep.inp'] = np.random.random(1)[0]
        p.final_setup()

        p2 = om.Problem()
        comp = p2.model.add_subsystem('comp', SubProbComp(input_size=5, num_nodes=3))
        p2.setup(mode='fwd')

        p2['comp.x'] = p['indep.x']
        p2['comp.inp'] = p['indep.inp']

        p.run_model()

        J = p.compute_totals(of=['C3.out'], wrt=['indep.x', 'indep.inp'], return_format='array')
        print(J)


        p2.run_model()

        self.assertEqual(p['C3.out'], p2['comp.out'])


    def test_rev(self):
        pass

