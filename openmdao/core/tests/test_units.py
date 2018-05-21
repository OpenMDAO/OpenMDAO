""" Tests the ins and outs of automatic unit conversion in OpenMDAO."""

import unittest
import warnings

from six import iteritems

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp, DirectSolver
from openmdao.api import ExecComp
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.unit_conv import UnitConvGroup, SrcComp, TgtCompC, TgtCompF, \
    TgtCompK, SrcCompFD, TgtCompCFD, TgtCompFFD, TgtCompKFD, TgtCompFMulti


class SpeedComp(ExplicitComponent):
    """Simple speed computation from distance and time with unit conversations."""

    def setup(self):
        self.add_input('distance', val=1.0, units='km')
        self.add_input('time', val=1.0, units='h')
        self.add_output('speed', val=1.0, units='km/h')

    def compute(self, inputs, outputs):
        outputs['speed'] = inputs['distance'] / inputs['time']


class TestUnitConversion(unittest.TestCase):
    """ Testing automatic unit conversion."""

    def test_basic_dense_jac(self):
        """Test that output values and total derivatives are correct."""
        prob = Problem(model=UnitConvGroup(assembled_jac_type='dense'))

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        # Check the outputs after running to test the unit conversions
        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        # Check the total derivatives in forward mode
        wrt = ['px1.x1']
        of = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3', 'px1.x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3', 'px1.x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3', 'px1.x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3', 'px1.x1'][0][0], 1.0, 1e-6)

    def test_dangling_input_w_units(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('C1', ExecComp('y=x', x={'units': 'ft'}, y={'units': 'm'}))
        prob.setup()
        prob.run_model()
        # this test passes as long as it doesn't raise an exception

    def test_speed(self):
        from openmdao.api import Problem, Group, IndepVarComp, ExecComp
        from openmdao.core.tests.test_units import SpeedComp

        comp = IndepVarComp()
        comp.add_output('distance', val=1., units='m')
        comp.add_output('time', val=1., units='s')

        prob = Problem(model=Group())
        prob.model.add_subsystem('c1', comp)
        prob.model.add_subsystem('c2', SpeedComp())
        prob.model.add_subsystem('c3', ExecComp('f=speed',speed={'units': 'm/s'}))
        prob.model.connect('c1.distance', 'c2.distance')
        prob.model.connect('c1.time', 'c2.time')
        prob.model.connect('c2.speed', 'c3.speed')

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['c1.distance'], 1.)  # units: m
        assert_rel_error(self, prob['c2.distance'], 1.e-3)  # units: km

        assert_rel_error(self, prob['c1.time'], 1.)  # units: s
        assert_rel_error(self, prob['c2.time'], 1./3600.)  # units: h

        assert_rel_error(self, prob['c2.speed'], 3.6)  # units: km/h
        assert_rel_error(self, prob['c3.f'], 1.0)  # units: km/h

    def test_basic(self):
        """Test that output values and total derivatives are correct."""
        prob = Problem(model=UnitConvGroup())

        # Check the outputs after running to test the unit conversions
        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        # Check the total derivatives in forward mode
        wrt = ['px1.x1']
        of = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3', 'px1.x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3', 'px1.x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3', 'px1.x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3', 'px1.x1'][0][0], 1.0, 1e-6)

        # Make sure check partials handles conversion
        data = prob.check_partials()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-6)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-6)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-6)

    def test_basic_apply(self):
        """Test that output values and total derivatives are correct."""

        class SrcCompa(ExplicitComponent):
            """Source provides degrees Celsius."""

            def setup(self):
                self.add_input('x1', 100.0)
                self.add_output('x2', 100.0, units='degC')

            def compute(self, inputs, outputs):
                """ Pass through."""
                outputs['x2'] = inputs['x1']

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                """ Derivative is 1.0"""

                if mode == 'fwd':
                    d_outputs['x2'] += d_inputs['x1']
                else:
                    d_inputs['x1'] += d_outputs['x2']

        class TgtCompFa(ExplicitComponent):
            """Target expressed in degrees F."""

            def setup(self):
                self.add_input('x2', 100.0, units='degF')
                self.add_output('x3', 100.0)

            def compute(self, inputs, outputs):
                """ Pass through."""
                outputs['x3'] = inputs['x2']

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                """ Derivative is 1.0"""

                if mode == 'fwd':
                    d_outputs['x3'] += d_inputs['x2']
                else:
                    d_inputs['x2'] += d_outputs['x3']

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px1', IndepVarComp('x1', 100.0))
        model.add_subsystem('src', SrcCompa())
        model.add_subsystem('tgtF', TgtCompFa())

        model.connect('px1.x1', 'src.x1')
        model.connect('src.x2', 'tgtF.x2')

        # Check the outputs after running to test the unit conversions
        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)

        # Check the total derivatives in forward mode
        wrt = ['px1.x1']
        of = ['tgtF.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)

    def test_basic_fd_comps(self):

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.model.add_subsystem('src', SrcCompFD())
        prob.model.add_subsystem('tgtF', TgtCompFFD())
        prob.model.add_subsystem('tgtC', TgtCompCFD())
        prob.model.add_subsystem('tgtK', TgtCompKFD())
        prob.model.connect('x1', 'src.x1')
        prob.model.connect('src.x2', 'tgtF.x2')
        prob.model.connect('src.x2', 'tgtC.x2')
        prob.model.connect('src.x2', 'tgtK.x2')

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        indep_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_totals(of=unknown_list, wrt=indep_list, return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=unknown_list, wrt=indep_list, return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        prob.model.approx_totals(method='fd')
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=unknown_list, wrt=indep_list, return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        # Make sure check partials handles conversion
        data = prob.check_partials()

        for key1, val1 in iteritems(data):
            for key2, val2 in iteritems(val1):
                assert_rel_error(self, val2['abs error'][0], 0.0, 1e-6)
                assert_rel_error(self, val2['abs error'][1], 0.0, 1e-6)
                assert_rel_error(self, val2['abs error'][2], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][0], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][1], 0.0, 1e-6)
                assert_rel_error(self, val2['rel error'][2], 0.0, 1e-6)

    def test_bad_units(self):
        """Test error handling when invalid units are declared."""
        class Comp1(ExplicitComponent):
            def setup(self):
                self.add_input('x', 0.0, units='junk')

        class Comp2(ExplicitComponent):
            def setup(self):
                self.add_output('x', 0.0, units='junk')

        with self.assertRaises(Exception) as cm:
            prob = Problem(model=Comp1())
            prob.setup()
        expected_msg = "The units 'junk' are invalid"
        self.assertTrue(expected_msg in str(cm.exception))

        with self.assertRaises(Exception) as cm:
            prob = Problem(model=Comp2())
            prob.setup()
        expected_msg = "The units 'junk' are invalid"
        self.assertTrue(expected_msg in str(cm.exception))

    def test_add_unitless_output(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('indep', IndepVarComp('x', 0.0, units='unitless'))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.setup(check=False)
            self.assertEqual(str(w[-1].message),
                             "Output 'x' has units='unitless' but 'unitless' has "
                             "been deprecated. Use units=None "
                             "instead.  Note that connecting a unitless variable to "
                             "one with units is no longer an error, but will issue "
                             "a warning instead.")

    def test_add_unitless_input(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('C1', ExecComp('y=x', x={'units': 'unitless'}))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.setup(check=False)
        self.assertEqual(str(w[-1].message),
                         "Input 'x' has units='unitless' but 'unitless' has "
                         "been deprecated. Use units=None "
                         "instead.  Note that connecting a unitless variable to "
                         "one with units is no longer an error, but will issue "
                         "a warning instead.")

    def test_incompatible_units(self):
        """Test error handling when only one of src and tgt have units."""
        prob = Problem(model=Group())
        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0), promotes_outputs=['x1'])
        prob.model.add_subsystem('src', SrcComp(), promotes_inputs=['x1'])
        prob.model.add_subsystem('tgt', ExecComp('yy=xx', xx={'value': 0.0, 'units': 'unitless'}))
        prob.model.connect('src.x2', 'tgt.xx')

        msg = "Output 'src.x2' with units of 'degC' is connected to input 'tgt.xx' which has no units."
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.setup()
        self.assertEqual(str(w[-1].message), msg)

    def test_basic_implicit_conn(self):
        """Test units with all implicit connections."""
        prob = Problem(model=Group())
        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0), promotes_outputs=['x1'])
        prob.model.add_subsystem('src', SrcComp(), promotes_inputs=['x1'], promotes_outputs=['x2'])
        prob.model.add_subsystem('tgtF', TgtCompF(), promotes_inputs=['x2'])
        prob.model.add_subsystem('tgtC', TgtCompC(), promotes_inputs=['x2'])
        prob.model.add_subsystem('tgtK', TgtCompK(), promotes_inputs=['x2'])

        # Check the outputs after running to test the unit conversions
        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        # Check the total derivatives in forward mode
        wrt = ['x1']
        of = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['tgtF.x3', 'x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3', 'x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3', 'x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['tgtF.x3', 'x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3', 'x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3', 'x1'][0][0], 1.0, 1e-6)

    def test_basic_grouped(self):

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        sub1 = prob.model.add_subsystem('sub1', Group())
        sub2 = prob.model.add_subsystem('sub2', Group())

        sub1.add_subsystem('src', SrcComp())
        sub2.add_subsystem('tgtF', TgtCompF())
        sub2.add_subsystem('tgtC', TgtCompC())
        sub2.add_subsystem('tgtK', TgtCompK())

        prob.model.connect('x1', 'sub1.src.x1')
        prob.model.connect('sub1.src.x2', 'sub2.tgtF.x2')
        prob.model.connect('sub1.src.x2', 'sub2.tgtC.x2')
        prob.model.connect('sub1.src.x2', 'sub2.tgtK.x2')

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['sub1.src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['sub2.tgtK.x3'], 373.15, 1e-6)

        wrt = ['x1']
        of = ['sub2.tgtF.x3', 'sub2.tgtC.x3', 'sub2.tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_basic_grouped_bug_from_pycycle(self):

        prob = Problem()
        root = prob.model = Group()

        prob.model.add_subsystem('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        sub1 = prob.model.add_subsystem('sub1', Group(), promotes=['x2'])
        sub1.add_subsystem('src', SrcComp(), promotes=['x2'])
        root.add_subsystem('tgtF', TgtCompFMulti())
        root.add_subsystem('tgtC', TgtCompC())
        root.add_subsystem('tgtK', TgtCompK())

        prob.model.connect('x1', 'sub1.src.x1')
        prob.model.connect('x2', 'tgtF.x2')
        prob.model.connect('x2', 'tgtC.x2')
        prob.model.connect('x2', 'tgtK.x2')

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        wrt = ['x1']
        of = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    #def test_basic_grouped_grouped_implicit(self):

        #prob = Problem()
        #root = prob.model = Group()
        #sub1 = prob.model.add('sub1', Group(), promotes=['x2'])
        #sub2 = prob.model.add('sub2', Group(), promotes=['x2'])
        #sub1.add('src', SrcComp(), promotes = ['x2'])
        #sub2.add('tgtF', TgtCompFMulti(), promotes=['x2'])
        #sub2.add('tgtC', TgtCompC(), promotes=['x2'])
        #sub2.add('tgtK', TgtCompK(), promotes=['x2'])
        #prob.model.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.model.connect('x1', 'sub1.src.x1')

        #prob.setup(check=False)
        #prob.run()

        #assert_rel_error(self, prob['x2'], 100.0, 1e-6)
        #assert_rel_error(self, prob['sub2.tgtF.x3'], 212.0, 1e-6)
        #assert_rel_error(self, prob['sub2.tgtC.x3'], 100.0, 1e-6)
        #assert_rel_error(self, prob['sub2.tgtK.x3'], 373.15, 1e-6)

        #indep_list = ['x1']
        #unknown_list = ['sub2.tgtF.x3', 'sub2.tgtC.x3', 'sub2.tgtK.x3']
        #J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               #return_format='dict')

        #assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               #return_format='dict')

        #assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               #return_format='dict')

        #assert_rel_error(self, J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    #def test_apply_linear_adjoint(self):
        ## Make sure we can index into dinputs

        #class Attitude_Angular(Component):
            #""" Calculates angular velocity vector from the satellite's orientation
            #matrix and its derivative.
            #"""

            #def __init__(self, n=2):
                #super(Attitude_Angular, self).__init__()

                #self.n = n

                ## Inputs
                #self.add_param('O_BI', np.zeros((3, 3, n)), units="ft",
                               #desc="Rotation matrix from body-fixed frame to Earth-centered "
                               #"inertial frame over time")

                #self.add_param('Odot_BI', np.zeros((3, 3, n)), units="km",
                               #desc="First derivative of O_BI over time")

                ## Outputs
                #self.add_output('w_B', np.zeros((3, n)), units="1/s",
                                #desc="Angular velocity vector in body-fixed frame over time")

                #self.dw_dOdot = np.zeros((n, 3, 3, 3))
                #self.dw_dO = np.zeros((n, 3, 3, 3))

            #def solve_nonlinear(self, inputs, outputs, resids):
                #""" Calculate output. """

                #O_BI = inputs['O_BI']
                #Odot_BI = inputs['Odot_BI']
                #w_B = outputs['w_B']

                #for i in range(0, self.n):
                    #w_B[0, i] = np.dot(Odot_BI[2, :, i], O_BI[1, :, i])
                    #w_B[1, i] = np.dot(Odot_BI[0, :, i], O_BI[2, :, i])
                    #w_B[2, i] = np.dot(Odot_BI[1, :, i], O_BI[0, :, i])

            #def linearize(self, inputs, outputs, resids):
                #""" Calculate and save derivatives. (i.e., Jacobian) """

                #O_BI = inputs['O_BI']
                #Odot_BI = inputs['Odot_BI']

                #for i in range(0, self.n):
                    #self.dw_dOdot[i, 0, 2, :] = O_BI[1, :, i]
                    #self.dw_dO[i, 0, 1, :] = Odot_BI[2, :, i]

                    #self.dw_dOdot[i, 1, 0, :] = O_BI[2, :, i]
                    #self.dw_dO[i, 1, 2, :] = Odot_BI[0, :, i]

                    #self.dw_dOdot[i, 2, 1, :] = O_BI[0, :, i]
                    #self.dw_dO[i, 2, 0, :] = Odot_BI[1, :, i]

            #def apply_linear(self, inputs, outputs, dinputs, doutputs, dresids, mode):
                #""" Matrix-vector product with the Jacobian. """

                #dw_B = dresids['w_B']

                #if mode == 'fwd':
                    #for k in range(3):
                        #for i in range(3):
                            #for j in range(3):
                                #if 'O_BI' in dinputs:
                                    #dw_B[k, :] += self.dw_dO[:, k, i, j] * \
                                        #dinputs['O_BI'][i, j, :]
                                #if 'Odot_BI' in dinputs:
                                    #dw_B[k, :] += self.dw_dOdot[:, k, i, j] * \
                                        #dinputs['Odot_BI'][i, j, :]

                #else:

                    #for k in range(3):
                        #for i in range(3):
                            #for j in range(3):

                                #if 'O_BI' in dinputs:
                                    #dinputs['O_BI'][i, j, :] += self.dw_dO[:, k, i, j] * \
                                        #dw_B[k, :]

                                #if 'Odot_BI' in dinputs:
                                    #dinputs['Odot_BI'][i, j, :] -= -self.dw_dOdot[:, k, i, j] * \
                                        #dw_B[k, :]

        #prob = Problem()
        #root = prob.model = Group()
        #prob.model.add('comp', Attitude_Angular(n=5), promotes=['*'])
        #prob.model.add('p1', IndepVarComp('O_BI', np.ones((3, 3, 5))), promotes=['*'])
        #prob.model.add('p2', IndepVarComp('Odot_BI', np.ones((3, 3, 5))), promotes=['*'])

        #prob.setup(check=False)
        #prob.run()

        #indep_list = ['O_BI', 'Odot_BI']
        #unknown_list = ['w_B']
        #Jf = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                                #return_format='dict')

        #indep_list = ['O_BI', 'Odot_BI']
        #unknown_list = ['w_B']
        #Jr = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                                #return_format='dict')

        #for key, val in iteritems(Jr):
            #for key2 in val:
                #diff = abs(Jf[key][key2] - Jr[key][key2])
                #assert_rel_error(self, diff, 0.0, 1e-10)

    def test_incompatible_connections(self):

        class BadComp(ExplicitComponent):
            def setup(self):
                self.add_input('x2', 100.0, units='m')
                self.add_output('x3', 100.0)

        # Explicit Connection
        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('src', SrcComp())
        prob.model.add_subsystem('dest', BadComp())
        prob.model.connect('src.x2', 'dest.x2')
        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected_msg = "Output units of 'degC' for 'src.x2' are incompatible with input units of 'm' for 'dest.x2'."

        self.assertEqual(expected_msg, str(cm.exception))

        # Implicit Connection
        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('src', SrcComp(), promotes=['x2'])
        prob.model.add_subsystem('dest', BadComp(),promotes=['x2'])
        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)

        expected_msg = "Output units of 'degC' for 'src.x2' are incompatible with input units of 'm' for 'dest.x2'."

        self.assertEqual(expected_msg, str(cm.exception))

    #def test_nested_relevancy_base(self):

        ## This one actually has nothing to do with units, but it tests the
        ## "rest" of the problem that the others are testing, namely that
        ## outscope vars could sometimes cause a problem even absent any units.

        #prob = Problem()
        #root = prob.model = Group()
        #root.add('p1', IndepVarComp('xx', 3.0))
        #root.add('c1', ExecComp(['y1=0.5*x + 0.1*xx', 'y2=0.3*x - 1.0*xx']))
        #root.add('c2', ExecComp(['y=0.5*x']))
        #sub = root.add('sub', Group())
        #sub.add('cc1', ExecComp(['y=0.1*x1 + 0.01*x2']))
        #sub.add('cc2', ExecComp(['y=0.1*x']))

        #root.connect('p1.xx', 'c1.xx')
        #root.connect('c1.y1', 'c2.x')
        ##root.connect('c2.y', 'c1.x')
        #root.connect('c1.y2', 'sub.cc1.x1')
        #root.connect('sub.cc1.y', 'sub.cc2.x')
        #root.connect('sub.cc2.y', 'sub.cc1.x2')

        #root.nonlinear_solver = Newton()
        #root.linear_solver = ScipyKrylov()

        #sub.nonlinear_solver = Newton()
        #sub.linear_solver = DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('c1.y2')

        #prob.setup(check=False)

        #prob.run()

        ## Pollute the dpvec
        #sub.dpmat[None]['cc1.x1'] = 1e10

        ## Make sure we can calculate a good derivative in the presence of pollution

        #sub.linear_solver.rel_inputs = ['sub.cc2.x', 'sub.cc1.x2']
        #rhs_buf = {None : np.array([3.5, 1.7])}
        #sol_buf = sub.linear_solver.solve(rhs_buf, sub, mode='fwd')[None]
        #assert_rel_error(self, sol_buf[0], -3.52052052, 1e-3)
        #assert_rel_error(self, sol_buf[1], -2.05205205, 1e-3)

    #def test_nested_relevancy(self):

        ## This test is just to make sure that values in the dp vector from
        ## higher scopes aren't sitting there converting themselves during sub
        ## iterations.

        #prob = Problem()
        #root = prob.model = Group()
        #root.add('p1', IndepVarComp('xx', 3.0))
        #root.add('c1', ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2' : 'km'}))
        #root.add('c2', ExecComp(['y=0.5*x']))
        #sub = root.add('sub', Group())
        #sub.add('cc1', ExecComp(['y=1.01*x1 + 1.01*x2'], units={'x1' : 'nm'}))
        #sub.add('cc2', ExecComp(['y=1.01*x']))

        #root.connect('p1.xx', 'c1.xx')
        #root.connect('c1.y1', 'c2.x')
        #root.connect('c2.y', 'c1.x')
        #root.connect('c1.y2', 'sub.cc1.x1')
        #root.connect('sub.cc1.y', 'sub.cc2.x')
        #root.connect('sub.cc2.y', 'sub.cc1.x2')

        #root.nonlinear_solver = Newton()
        #root.nonlinear_solver.options['maxiter'] = 1
        #root.linear_solver = ScipyKrylov()
        #root.linear_solver.options['maxiter'] = 1

        #sub.nonlinear_solver = Newton()
        #sub.linear_solver = DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup(check=False)

        #prob.run()
        #self.assertTrue(not np.isnan(prob['sub.cc2.y']))

    #def test_nested_relevancy_adjoint(self):

        ## This test is just to make sure that values in the dp vector from
        ## higher scopes aren't sitting there converting themselves during sub
        ## iterations.

        #prob = Problem()
        #root = prob.model = Group()
        #root.add('p1', IndepVarComp('xx', 3.0))
        #root.add('c1', ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2' : 'km'}))
        #root.add('c2', ExecComp(['y=0.5*x']))
        #sub = root.add('sub', Group())
        #sub.add('cc1', ExecComp(['y=1.01*x1 + 1.01*x2'], units={'x1' : 'nm'}))
        #sub.add('cc2', ExecComp(['y=1.01*x']))

        #root.connect('p1.xx', 'c1.xx')
        #root.connect('c1.y1', 'c2.x')
        #root.connect('c2.y', 'c1.x')
        #root.connect('c1.y2', 'sub.cc1.x1')
        #root.connect('sub.cc1.y', 'sub.cc2.x')
        #root.connect('sub.cc2.y', 'sub.cc1.x2')

        #root.nonlinear_solver = Newton()
        #root.nonlinear_solver.options['maxiter'] = 1
        #root.linear_solver = ScipyKrylov()
        #root.linear_solver.options['maxiter'] = 1
        #root.linear_solver.options['mode'] = 'rev'

        #sub.nonlinear_solver = Newton()
        #sub.linear_solver = DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup(check=False)

        #prob.run()
        #self.assertTrue(not np.isnan(prob['sub.cc2.y']))

    #def test_nested_relevancy_adjoint_apply_linear(self):

        ## This test is just to make sure that values in the dp vector from
        ## higher scopes aren't sitting there converting themselves during sub
        ## iterations.

        #class TestComp(Component):

            #def __init__(self):
                #super(TestComp, self).__init__()

                ## Params
                #self.add_param('x1', 1.0, units='mm')
                #self.add_param('x2', 1.0)

                ## Unknowns
                #self.add_output('y', 1.0)

                #self.dx1count = 0
                #self.dx2count = 0

            #def solve_nonlinear(self, inputs, outputs, resids):
                #""" Doesn't do much. """
                #x1 = inputs['x1']
                #x2 = inputs['x2']
                #outputs['y'] = 1.01*(x1 + x2)

            #def apply_linear(self, inputs, outputs, dinputs, doutputs, dresids,
                             #mode):
                #"""Returns the product of the incoming vector with the Jacobian."""

                #if mode == 'fwd':
                    #if 'x1' in dinputs:
                        #dresids['y'] += 1.01*dinputs['x1']
                        #self.dx1count += 1
                    #if 'x2' in dinputs:
                        #dresids['y'] += 1.01*dinputs['x2']
                        #self.dx2count += 1

                #elif mode == 'rev':
                    #if 'x1' in dinputs:
                        #dinputs['x1'] = 1.01*dresids['y']
                        #self.dx1count += 1
                    #if 'x2' in dinputs:
                        #dinputs['x2'] = 1.01*dresids['y']
                        #self.dx2count += 1

        #prob = Problem()
        #root = prob.model = Group()
        #root.add('p1', IndepVarComp('xx', 3.0))
        #root.add('c1', ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2' : 'km'}))
        #root.add('c2', ExecComp(['y=0.5*x']))
        #sub = root.add('sub', Group())
        #sub.add('cc1', TestComp())
        #sub.add('cc2', ExecComp(['y=1.01*x']))

        #root.connect('p1.xx', 'c1.xx')
        #root.connect('c1.y1', 'c2.x')
        #root.connect('c2.y', 'c1.x')
        #root.connect('c1.y2', 'sub.cc1.x1')
        #root.connect('sub.cc1.y', 'sub.cc2.x')
        #root.connect('sub.cc2.y', 'sub.cc1.x2')

        #root.nonlinear_solver = Newton()
        #root.nonlinear_solver.options['maxiter'] = 1
        #root.linear_solver = ScipyKrylov()
        #root.linear_solver.options['maxiter'] = 1
        #root.linear_solver.options['mode'] = 'rev'

        #sub.nonlinear_solver = Newton()
        #sub.linear_solver = DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup(check=False)
        #prob.run()

        ## x1 deriv code should be called less if the dinputs vec only
        ## considers sub relevancy
        #self.assertTrue(sub.cc1.dx1count < sub.cc1.dx2count)

    #def test_nested_relevancy_gmres(self):

        ## This test is just to make sure that values in the dp vector from
        ## higher scopes aren't sitting there converting themselves during sub
        ## iterations.

        #prob = Problem()
        #root = prob.model = Group()
        #root.add('p1', IndepVarComp('xx', 3.0))
        #root.add('c1', ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2' : 'km'}))
        #root.add('c2', ExecComp(['y=0.5*x']))
        #sub = root.add('sub', Group())
        #sub.add('cc1', ExecComp(['y=1.01*x1 + 1.01*x2'], units={'x1' : 'fm'}))
        #sub.add('cc2', ExecComp(['y=1.01*x']))

        #root.connect('p1.xx', 'c1.xx')
        #root.connect('c1.y1', 'c2.x')
        #root.connect('c2.y', 'c1.x')
        #root.connect('c1.y2', 'sub.cc1.x1')
        #root.connect('sub.cc1.y', 'sub.cc2.x')
        #root.connect('sub.cc2.y', 'sub.cc1.x2')

        #root.nonlinear_solver = Newton()
        #root.nonlinear_solver.options['maxiter'] = 1
        #root.linear_solver = ScipyKrylov()
        #root.linear_solver.options['maxiter'] = 1

        #sub.nonlinear_solver = Newton()
        #sub.linear_solver = ScipyKrylov()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup(check=False)

        #prob.run()

        ## GMRES doesn't cause a successive build-up in the value of an out-of
        ## scope param, but the linear solver doesn't converge. We can test to
        ## make sure it does.
        #iter_count = sub.linear_solver.iter_count
        #self.assertTrue(iter_count < 20)
        #self.assertTrue(not np.isnan(prob['sub.cc2.y']))

    #def test_nested_relevancy_gmres_precon(self):

        ## Make sure preconditioners also work

        #prob = Problem()
        #root = prob.model = Group()
        #root.add('p1', IndepVarComp('xx', 3.0))
        #root.add('c1', ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2' : 'km'}))
        #root.add('c2', ExecComp(['y=0.5*x']))
        #sub = root.add('sub', Group())
        #sub.add('cc1', ExecComp(['y=1.01*x1 + 1.01*x2'], units={'x1' : 'fm'}))
        #sub.add('cc2', ExecComp(['y=1.01*x']))

        #root.connect('p1.xx', 'c1.xx')
        #root.connect('c1.y1', 'c2.x')
        #root.connect('c2.y', 'c1.x')
        #root.connect('c1.y2', 'sub.cc1.x1')
        #root.connect('sub.cc1.y', 'sub.cc2.x')
        #root.connect('sub.cc2.y', 'sub.cc1.x2')

        #root.nonlinear_solver = Newton()
        #root.nonlinear_solver.options['maxiter'] = 1
        #root.linear_solver = ScipyKrylov()
        #root.linear_solver.options['maxiter'] = 1

        #sub.nonlinear_solver = Newton()
        #sub.linear_solver = ScipyKrylov()
        #sub.linear_solver.precon = DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup(check=False)

        #prob.run()

        ## GMRES doesn't cause a successive build-up in the value of an out-of
        ## scope param, but the linear solver doesn't converge. We can test to
        ## make sure it does.
        #iter_count = sub.linear_solver.iter_count
        #self.assertTrue(iter_count < 20)
        #self.assertTrue(not np.isnan(prob['sub.cc2.y']))

#class PBOSrcComp(Component):

    #def __init__(self):
        #super(PBOSrcComp, self).__init__()

        #self.add_param('x1', 100.0)
        #self.add_output('x2', 100.0, units='degC', pass_by_obj=True)
        #self.deriv_options['type'] = 'fd'

    #def solve_nonlinear(self, inputs, outputs, resids):
        #""" No action."""
        #outputs['x2'] = inputs['x1']


#class PBOTgtCompF(Component):

    #def __init__(self):
        #super(PBOTgtCompF, self).__init__()

        #self.add_param('x2', 100.0, units='degF', pass_by_obj=True)
        #self.add_output('x3', 100.0)
        #self.deriv_options['type'] = 'fd'

    #def solve_nonlinear(self, inputs, outputs, resids):
        #""" No action."""
        #outputs['x3'] = inputs['x2']


#class TestUnitConversionPBO(unittest.TestCase):
    #""" Tests support for unit conversions on pass_by_obj connections."""

    #def test_basic(self):

        #prob = Problem()
        #prob.model = Group()
        #prob.model.add('src', PBOSrcComp())
        #prob.model.add('tgtF', PBOTgtCompF())
        #prob.model.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.model.connect('x1', 'src.x1')
        #prob.model.connect('src.x2', 'tgtF.x2')

        #prob.model.deriv_options['type'] = 'fd'

        #prob.setup(check=False)
        #prob.run()

        #assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        #assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)

        #indep_list = ['x1']
        #unknown_list = ['tgtF.x3']
        #J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)

        #stream = cStringIO()
        #conv = prob.model.list_unit_conv(stream=stream)
        #self.assertTrue((('src.x2', 'tgtF.x2'), ('degC', 'degF')) in conv)


    #def test_radian_bug(self):

        #class Src(Component):

            #def __init__(self):
                #super(Src, self).__init__()

                #self.add_output('x1', 180.0, units='deg')
                #self.add_output('x2', np.pi, units='rad')
                #self.add_output('x3', 2.0, units='m')
                #self.deriv_options['type'] = 'fd'

            #def solve_nonlinear(self, inputs, outputs, resids):
                #""" No action."""
                #pass


        #class Tgt(Component):

            #def __init__(self):
                #super(Tgt, self).__init__()

                #self.add_param('x1', 0.0, units='rad')
                #self.add_param('x2', 0.0, units='deg')
                #self.add_param('x3', 0.0, units='ft')
                #self.deriv_options['type'] = 'fd'

            #def solve_nonlinear(self, inputs, outputs, resids):
                #""" No action."""
                #pass

        #top = Problem()
        #root = top.root = Group()
        #root.add('src', Src())
        #root.add('tgt', Tgt())

        #root.connect('src.x1', 'tgt.x1')
        #root.connect('src.x2', 'tgt.x2')
        #root.connect('src.x3', 'tgt.x3')

        #top.setup(check=False)
        #top.run()

        #assert_rel_error(self, top['tgt.x1'], np.pi, 1e-6)
        #assert_rel_error(self, top['tgt.x2'], 180.0, 1e-6)
        #assert_rel_error(self, top['tgt.x3'], 2.0/0.3048, 1e-6)


if __name__ == "__main__":
    unittest.main()
