import os
import unittest

import numpy as np

from openmdao.api import Problem, ExplicitComponent, IndepVarComp, ExecComp, ScipyOptimizeDriver
from openmdao.test_suite.components.sellar import SellarNoDerivatives

try:
    from pyxdsm.XDSM import XDSM
    from openmdao.devtools.xdsm_viewer.xdsm_writer import write_xdsm
except ImportError:
    XDSM = None

FILENAME = 'XDSM'


@unittest.skipUnless(XDSM, "XDSM is required.")
class TestXDSMViewer(unittest.TestCase):

    def test_pyxdsm_sellar(self):
        """Makes XDSM for the Sellar problem"""

        prob = Problem()
        prob.model = model = SellarNoDerivatives()
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]), indices=np.arange(2, dtype=int))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', equals=np.zeros(1))
        model.add_constraint('con2', upper=0.0)

        prob.setup(check=False)
        prob.final_setup()

        # no output checking, just make sure no exceptions raised
        write_xdsm(prob, filename=FILENAME+'0', show_browser=False)

    def test_pyxdsm_sellar_no_recurse(self):
        """Makes XDSM for the Sellar problem, with no recursion."""

        prob = Problem()
        prob.model = model = SellarNoDerivatives()
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]), indices=np.arange(2, dtype=int))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', equals=np.zeros(1))
        model.add_constraint('con2', upper=0.0)

        prob.setup(check=False)
        prob.final_setup()

        # no output checking, just make sure no exceptions raised
        write_xdsm(prob, filename=FILENAME+'1', show_browser=False, recurse=False)

    def test_pyxdsm_sphere(self):
        """
        Makes an XDSM of the Sphere test case. It also adds a design variable, constraint and
        objective.
        """
        class Rosenbrock(ExplicitComponent):

            def __init__(self, problem):
                super(Rosenbrock, self).__init__()
                self.problem = problem
                self.counter = 0

            def setup(self):
                self.add_input('x', np.array([1.5, 1.5]))
                self.add_output('f', 0.0)
                self.declare_partials('f', 'x', method='fd', form='central', step=1e-4)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = sum(x**2)

        x0 = np.array([1.2, 1.5])

        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp(problem=prob), promotes=['*'])
        indeps.add_output('x', list(x0))

        prob.model.add_subsystem('sphere', Rosenbrock(problem=prob), promotes=['*'])
        prob.model.add_subsystem('con', ExecComp('c=sum(x)', x=np.ones(2)), promotes=['*'])
        prob.driver = ScipyOptimizeDriver()
        prob.model.add_design_var('x')
        prob.model.add_objective('f')
        prob.model.add_constraint('c', lower=1.0)

        prob.setup(check=False)
        prob.final_setup()

        # no output checking, just make sure no exceptions raised
        write_xdsm(prob, filename=FILENAME+'2', show_browser=False)

    def test_xdsmjs(self):
        """Makes XDSMjs input file for the Sellar problem"""

        filename = 'xdsm'  # this name is needed for XDSMjs
        prob = Problem()
        prob.model = model = SellarNoDerivatives()
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]), indices=np.arange(2, dtype=int))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', equals=np.zeros(1))
        model.add_constraint('con2', upper=0.0)

        prob.setup(check=False)
        prob.final_setup()

        # no output checking, just make sure no exceptions raised
        write_xdsm(prob, filename=filename, out_format='json', subs=(), show_browser=False)

    def test_wrong_out_format(self):
        """Incorrect output format error."""

        filename = 'xdsm'  # this name is needed for XDSMjs
        prob = Problem()
        prob.model = SellarNoDerivatives()

        prob.setup(check=False)
        prob.final_setup()

        # no output checking, just make sure no exceptions raised
        with self.assertRaises(ValueError):
            write_xdsm(prob, filename=filename, out_format='jpg', subs=(), show_browser=False)

    def tearDown(self):
        """Set "clean_up" to False, if you want to inspect the output files."""
        clean_up = True

        def clean_file(fname):
            try:  # Try to clean up
                if os.path.exists(fname):
                    os.remove(fname)
            except Exception as e:
                pass

        if clean_up:
            nr_pyxdsm_tests = 3
            for ext in ('aux', 'log', 'pdf', 'tex', 'tikz'):
                for i in range(nr_pyxdsm_tests):
                    filename = '.'.join([FILENAME+str(i), ext])
                    clean_file(filename)

            clean_file('xdsm.json')


if __name__ == "__main__":
    unittest.main()