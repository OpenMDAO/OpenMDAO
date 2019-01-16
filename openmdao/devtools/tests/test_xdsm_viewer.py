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
        filename = FILENAME+'0'
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

        # Write output
        write_xdsm(prob, filename=filename, out_format='tex', show_browser=False)
        # Check if file was created
        self.assertTrue(os.path.isfile('.'.join([filename, 'tex'])))

    def test_pyxdsm_sellar_no_recurse(self):
        """Makes XDSM for the Sellar problem, with no recursion."""

        filename = FILENAME+'1'
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

        # Write output
        write_xdsm(prob, filename=filename, out_format='tex', show_browser=False, recurse=False)
        # Check if file was created
        self.assertTrue(os.path.isfile('.'.join([filename, 'tex'])))

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
        filename = FILENAME+'2'

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

        # Write output
        write_xdsm(prob, filename=filename, out_format='tex', show_browser=False)
        # Check if file was created
        self.assertTrue(os.path.isfile('.'.join([filename, 'tex'])))

    def test_xdsmjs(self):
        """
        Makes XDSMjs input file for the Sellar problem.

        Data is in a separate JSON file.
        """

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

        # Write output
        write_xdsm(prob, filename=filename, out_format='html', subs=(), show_browser=False,
                   embed_data=False)
        # Check if file was created
        self.assertTrue(os.path.isfile('.'.join([filename, 'html'])))

    def test_xdsmjs_embed_data(self):
        """
        Makes XDSMjs input file for the Sellar problem.

        Data is embedded into the HTML file.
        """

        filename = 'xdsm_embedded_data'  # this name is needed for XDSMjs
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

        # Write output
        write_xdsm(prob, filename=filename, out_format='html', subs=(), show_browser=False,
                   embed_data=True)
        # Check if file was created
        self.assertTrue(os.path.isfile('.'.join([filename, 'html'])))

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
        clean_up = False

        def clean_file(fname):
            try:  # Try to clean up
                if os.path.exists(fname):
                    os.remove(fname)
            except Exception as e:
                pass

        if clean_up:

            # clean-up of pyXDSM files
            nr_pyxdsm_tests = 3  # number of tests with pyXDSM
            for ext in ('aux', 'log', 'pdf', 'tex', 'tikz'):
                for i in range(nr_pyxdsm_tests):
                    filename = '.'.join([FILENAME+str(i), ext])
                    clean_file(filename)

            # clean-up of XDSMjs files
            for ext in ('json', 'html'):
                filename = '.'.join(['xdsm', ext])
                clean_file(filename)


if __name__ == "__main__":
    unittest.main()