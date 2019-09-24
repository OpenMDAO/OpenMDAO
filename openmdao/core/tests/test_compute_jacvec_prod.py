
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
    return om.ExecComp('out = sum(3*x**2) + inp', x=np.ones(size - 1), inp=0.0, out=0.0)


class SubProbComp(om.ExplicitComponent):
    """
    This component contains a sub-Problem with a component that will be solved over num_nodes
    points instead of creating num_nodes instances of that same component and connecting them
    together.
    """
    def __init__(self, input_size, num_nodes, mode, **kwargs):
        super(SubProbComp, self).__init__(**kwargs)
        self.prob = None
        self.size = input_size
        self.num_nodes = num_nodes
        self.mode = mode

    def _setup_subprob(self):
        self.prob = p = om.Problem(comm=self.comm)
        model = self.prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp('x', val=np.zeros(self.size - 1)))
        indep.add_output('inp', val=0.0)

        model.add_subsystem('comp', get_comp(self.size))

        model.connect('indep.x', 'comp.x')
        model.connect('indep.inp', 'comp.inp')

        p.setup()
        p.final_setup()

    def setup(self):
        self._setup_subprob()

        self.add_input('x', np.zeros(self.size - 1))
        self.add_input('inp', val=0.0)
        self.add_output('out', val=0.0)
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        p = self.prob
        p['indep.x'] = inputs['x']
        p['indep.inp'] = inputs['inp']
        inp = inputs['inp']
        for i in range(self.num_nodes):
            p['indep.inp'] = inp
            p.run_model()
            inp = p['comp.out']

        outputs['out'] = p['comp.out']

    def _compute_partials_fwd(self, inputs, partials):
        p = self.prob
        x = inputs['x']
        p['indep.x'] = x
        p['indep.inp'] = inputs['inp']

        seed = {'indep.x':np.zeros(x.size), 'indep.inp': np.zeros(1)}
        p.run_model()
        p.model._linearize(None)
        for rhsname in seed:
            for rhs_i in range(seed[rhsname].size):
                seed['indep.x'][:] = 0.0
                seed['indep.inp'][:] = 0.0
                seed[rhsname][rhs_i] = 1.0
                for i in range(self.num_nodes):
                    p.model._vectors['output']['linear'].set_const(0.0)
                    p.model._vectors['residual']['linear'].set_const(0.0)
                    jvp = p.compute_jacvec_product(of=['comp.out'], wrt=['indep.x','indep.inp'], mode='fwd', seed=seed)
                    seed['indep.inp'][:] = jvp['comp.out']

                if rhsname == 'indep.x':
                    partials[self.pathname + '.out', self.pathname +'.x'][0, rhs_i] = jvp[self.pathname + '.out']
                else:
                    partials[self.pathname + '.out', self.pathname + '.inp'][0, 0] = jvp[self.pathname + '.out']

    def _compute_partials_rev(self, inputs, partials):
        p = self.prob
        p['indep.x'] = inputs['x']
        p['indep.inp'] = inputs['inp']
        seed = {'comp.out': np.ones(1)}

        stack = []
        comp = p.model.comp
        comp._inputs['inp'] = inputs['inp']
        # store the inputs to each comp (the comp at each node point) by doing nonlinear solves
        # and storing what the inputs are for each node point.  We'll set these inputs back
        # later when we linearize about each node point.
        for i in range(self.num_nodes):
            stack.append(comp._inputs['inp'][0])
            comp._inputs['x'] = inputs['x']
            comp._solve_nonlinear()
            comp._inputs['inp'] = comp._outputs['out']

        for i in range(self.num_nodes):
            p.model._vectors['output']['linear'].set_const(0.0)
            p.model._vectors['residual']['linear'].set_const(0.0)
            comp._inputs['inp'] = stack.pop()
            comp._inputs['x'] = inputs['x']
            p.model._linearize(None)
            jvp = p.compute_jacvec_product(of=['comp.out'], wrt=['indep.x','indep.inp'], mode='rev', seed=seed)
            seed['comp.out'][:] = jvp['indep.inp']

            # all of the comp.x's are connected to indep.x, so we have to accumulate their
            # contributions together
            partials[self.pathname + '.out', self.pathname + '.x'] += jvp['indep.x']

            # this one doesn't get accumulated because each comp.inp contributes to the
            # previous comp's .out (or to indep.inp in the case of the first comp) only.
            # Note that we have to handle this explicitly here because normally in OpenMDAO
            # we accumulate derivatives when we do reverse transfers.  We can't do that
            # here because we only have one instance of our component, so instead of
            # accumulating into separate 'comp.out' variables for each comp instance,
            # we would be accumulating into a single comp.out variable, which would make
            # our derivative too big.
            partials[self.pathname + '.out', self.pathname + '.inp'] = jvp['indep.inp']

    def compute_partials(self, inputs, partials):
        if self.mode == 'fwd':
            self._compute_partials_fwd(inputs, partials)
        else:
            self._compute_partials_rev(inputs, partials)




class TestPComputeJacvecProd(unittest.TestCase):

    def _build_om_model(self, size):
        p = om.Problem()
        model = p.model
        indep = model.add_subsystem('indep', om.IndepVarComp('x', val=np.zeros(size - 1)))
        indep.add_output('inp', val=0.0)

        C1 = model.add_subsystem('C1', get_comp(size))
        C2 = model.add_subsystem('C2', get_comp(size))
        C3 = model.add_subsystem('C3', get_comp(size))

        model.connect('indep.x', ['C1.x', 'C2.x', 'C3.x'])
        model.connect('indep.inp', 'C1.inp')
        model.connect('C1.out', 'C2.inp')
        model.connect('C2.out', 'C3.inp')

        return p

    def _build_cjv_model(self, size, mode):
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x', val=np.zeros(size - 1)))
        indep.add_output('inp', val=0.0)

        comp = p.model.add_subsystem('comp', SubProbComp(input_size=size, num_nodes=3, mode=mode))
        p.model.connect('indep.x', 'comp.x')
        p.model.connect('indep.inp', 'comp.inp')

        #p.model.linear_solver = om.DirectSolver()
        p.setup(mode=mode)

        return p

    def test_fwd(self):
        size = 5
        p = self._build_om_model(size)
        p.setup(mode='fwd')
        p['indep.x'] = np.arange(size-1, dtype=float) + 1. #np.random.random(size - 1)
        p['indep.inp'] = np.array([7.])  #np.random.random(1)[0]
        p.final_setup()

        p2 = self._build_cjv_model(size, 'fwd')

        p2['indep.x'] = p['indep.x']
        p2['indep.inp'] = p['indep.inp']

        p2.run_model()
        J2 = p2.compute_totals(of=['comp.out'], wrt=['indep.x', 'indep.inp'], return_format='array')

        p.run_model()
        J = p.compute_totals(of=['C3.out'], wrt=['indep.x', 'indep.inp'], return_format='array')

        self.assertEqual(p['C3.out'], p2['comp.out'])
        np.testing.assert_allclose(J2, J)


    def test_rev(self):
        size = 5
        p = self._build_om_model(size)
        p.setup(mode='rev')
        p['indep.x'] = np.arange(size-1, dtype=float) + 1. #np.random.random(size - 1)
        p['indep.inp'] = np.array([7.])  #np.random.random(1)[0]
        p.final_setup()

        p2 = self._build_cjv_model(size, 'rev')

        p2['indep.x'] = p['indep.x']
        p2['indep.inp'] = p['indep.inp']

        p.run_model()
        J = p.compute_totals(of=['C3.out'], wrt=['indep.x', 'indep.inp'], return_format='array')

        p2.run_model()
        self.assertEqual(p['C3.out'], p2['comp.out'])

        J2 = p2.compute_totals(of=['comp.out'], wrt=['indep.x', 'indep.inp'], return_format='array')

        np.testing.assert_allclose(J2, J)

