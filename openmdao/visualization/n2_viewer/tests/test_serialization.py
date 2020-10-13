"""
This test assures that the use of nonstandard datatypes anywhere that they are allowed in a model
does not break the recording of the JSON data structures needed for the model viewer.
"""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.core.driver import Driver
from openmdao.utils.testing_utils import use_tempdirs


class BadOpt(object):
    def junk(self):
        pass


class NonSerComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', np.zeros((6, )),
                       tags=['a', 'c'])
        self.add_output('y', np.zeros((6, )),
                       tags=['a', 'b'])

        self.add_discrete_input('dx', [{('Discrete_i', BadOpt): (2, {((1, ), (2, )): 'stuff'})}])
        self.add_discrete_output('dy', [{('Discrete_o', BadOpt): (2, {((1, ), (2, )): 'stuff'})}])
        self.add_discrete_input('dcomplex', 3 + 5j)

    def initialize(self):
        self.options.declare('good', 'good_string')
        self.options.declare('bad', [{(1, BadOpt): (2, 3)}])
        self.options.declare('bad2', {((1, ), (2, )): 'stuff'})
        self.options.declare('nonrec', 3.0, recordable=False)
        self.options.declare('cx', 3 + 7j)


class NonSerIComp(om.ImplicitComponent):

    def setup(self):
        self.add_input('xx', np.zeros((6, )))
        self.add_output('yy', np.zeros((6, )))

        self.add_discrete_input('problem', None)

    def initialize(self):
        self.options.declare('good', 'good_string')
        self.options.declare('bad', [{(1, BadOpt): (2, 3)}])
        self.options.declare('bad2', {((1, ), (2, )): 'stuff'})
        self.options.declare('nonrec', 3.0, recordable=False)
        self.options.declare('problem')


class NonSerNL(om.NonlinearRunOnce):

    def _declare_options(self):
        super()._declare_options()
        self.options.declare('bad', [{(1, BadOpt): (2, 3)}])
        self.options.declare('bad2', {((1, ), (2, )): 'stuff'})
        self.options.declare('nonrec', 3.0, recordable=False)


class NonSerLN(om.LinearRunOnce):

    def _declare_options(self):
        super()._declare_options()
        self.options.declare('bad', [{(1, BadOpt): (2, 3)}])
        self.options.declare('bad2', {((1, ), (2, )): 'stuff'})
        self.options.declare('nonrec', 3.0, recordable=False)


class NonSerDriver(Driver):

    def _declare_options(self):
        super()._declare_options()
        self.options.declare('bad', [{(1, BadOpt): (2, 3)}])
        self.options.declare('bad2', {((1, ), (2, )): 'stuff'})
        self.options.declare('nonrec', 3.0, recordable=False)


@use_tempdirs
class TestSerialization(unittest.TestCase):

    def test_exhaustive_model(self):
        em_prob = om.Problem()
        em_model = em_prob.model
        em_model.add_subsystem('ns', NonSerComp())
        em_model.add_subsystem('nsi', NonSerIComp())
        em_prob.setup()
        em_prob.final_setup()

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('ns', NonSerComp())
        nsi = model.add_subsystem('nsi', NonSerIComp())

        nsi.options['problem'] = em_prob

        model.nonlinear_solver = NonSerNL()
        model.linear_solver = NonSerLN()
        nsi.nonlinear_solver = NonSerNL()
        nsi.linear_solver = NonSerLN()

        model.add_design_var('ns.x', indices=om.slicer[2:])
        model.add_design_var('ns.dx')
        model.add_constraint('ns.y', indices=om.slicer[2:])
        model.add_objective('nsi.yy', index=om.slicer[-1])

        prob.driver = NonSerDriver()

        prob.add_recorder(om.SqliteRecorder("cases1.sql"))
        prob.driver.add_recorder(om.SqliteRecorder("cases2.sql"))
        prob.model.add_recorder(om.SqliteRecorder("cases3.sql"))
        prob.model.nonlinear_solver.add_recorder(om.SqliteRecorder("cases4.sql"))

        prob.setup()
        prob.set_val('nsi.problem', em_prob)

        prob.run_model()

        cr = om.CaseReader("cases1.sql")

        dval = cr.problem_metadata['tree']['children'][1]['options']['bad']
        key, val = [(k, v) for k, v in dval[0].items()][0]
        self.assertTrue("(1, <class" in key)
        self.assertEqual(val, [2, 3])

        self.assertEqual(cr.problem_metadata['tree']['children'][1]['options']['bad2'],
                         {'((1,), (2,))': 'stuff'})
        self.assertEqual(cr.problem_metadata['tree']['children'][1]['options']['nonrec'],
                         'Not Recordable')
        self.assertEqual(cr.problem_metadata['tree']['children'][1]['options']['cx'],
                         '(3+7j)')

        dval = cr.problem_metadata['tree']['children'][1]['children'][4]['value']
        key, val = [(k, v) for k, v in dval[0].items()][0]
        self.assertTrue("Discrete_o', <class" in key)
        self.assertEqual(val, [2, {'((1,), (2,))': 'stuff'}])

        dval = cr.problem_metadata['tree']['children'][1]['children'][1]['value']
        key, val = [(k, v) for k, v in dval[0].items()][0]
        self.assertTrue("Discrete_i', <class" in key)
        self.assertEqual(val, [2, {'((1,), (2,))': 'stuff'}])

        self.assertEqual(cr.problem_metadata['tree']['children'][1]['children'][2]['value'],
                         '(3+5j)')


if __name__ == "__main__":
    unittest.main()
