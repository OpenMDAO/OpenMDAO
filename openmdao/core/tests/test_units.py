""" Tests the ins and outs of automatic unit conversion in OpenMDAO."""

import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.test_suite.components.unit_conv import UnitConvGroup, SrcComp, TgtCompC, TgtCompF, \
    TgtCompK, SrcCompFD, TgtCompCFD, TgtCompFFD, TgtCompKFD, TgtCompFMulti


class SpeedComp(om.ExplicitComponent):
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
        prob = om.Problem(model=UnitConvGroup(assembled_jac_type='dense'))

        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

        # Check the outputs after running to test the unit conversions
        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_near_equal(prob['src.x2'], 100.0, 1e-6)
        assert_near_equal(prob['tgtF.x3'], 212.0, 1e-6)
        assert_near_equal(prob['tgtC.x3'], 100.0, 1e-6)
        assert_near_equal(prob['tgtK.x3'], 373.15, 1e-6)

        # Check the total derivatives in forward mode
        wrt = ['px1.x1']
        of = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_near_equal(J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3', 'px1.x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3', 'px1.x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_near_equal(J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3', 'px1.x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3', 'px1.x1'][0][0], 1.0, 1e-6)

    def test_dangling_input_w_units(self):
        prob = om.Problem()
        prob.model.add_subsystem('C1', om.ExecComp('y=x', x={'units': 'ft'}, y={'units': 'm'}))
        prob.setup()
        prob.run_model()
        # this test passes as long as it doesn't raise an exception

    def test_speed(self):
        import openmdao.api as om
        from openmdao.core.tests.test_units import SpeedComp

        prob = om.Problem()
        prob.model.add_subsystem('c1', SpeedComp())
        prob.model.add_subsystem('c2', om.ExecComp('f=speed',speed={'units': 'm/s'}))

        prob.model.set_input_defaults('c1.distance', val=1., units='m')
        prob.model.set_input_defaults('c1.time', val=1., units='s')

        prob.model.connect('c1.speed', 'c2.speed')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('c1.distance'), 1.e-3)  # units: km
        assert_near_equal(prob.get_val('c1.time'), 1./3600.)   # units: h
        assert_near_equal(prob.get_val('c1.speed'), 3.6)       # units: km/h

        assert_near_equal(prob.get_val('c2.f'), 1.0)           # units: m/s

    def test_basic(self):
        """Test that output values and total derivatives are correct."""
        prob = om.Problem(model=UnitConvGroup())

        # Check the outputs after running to test the unit conversions
        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_near_equal(prob['src.x2'], 100.0, 1e-6)
        assert_near_equal(prob['tgtF.x3'], 212.0, 1e-6)
        assert_near_equal(prob['tgtC.x3'], 100.0, 1e-6)
        assert_near_equal(prob['tgtK.x3'], 373.15, 1e-6)

        # Check the total derivatives in forward mode
        wrt = ['px1.x1']
        of = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_near_equal(J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3', 'px1.x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3', 'px1.x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_near_equal(J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3', 'px1.x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3', 'px1.x1'][0][0], 1.0, 1e-6)

        # Make sure check partials handles conversion
        data = prob.check_partials(out_stream=None)

        for key1, val1 in data.items():
            for key2, val2 in val1.items():
                assert_near_equal(val2['abs error'][0], 0.0, 1e-6)
                assert_near_equal(val2['rel error'][0], 0.0, 1e-6)

    def test_basic_apply(self):
        """Test that output values and total derivatives are correct."""

        class SrcCompa(om.ExplicitComponent):
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

        class TgtCompFa(om.ExplicitComponent):
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

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px1', om.IndepVarComp('x1', 100.0))
        model.add_subsystem('src', SrcCompa())
        model.add_subsystem('tgtF', TgtCompFa())

        model.connect('px1.x1', 'src.x1')
        model.connect('src.x2', 'tgtF.x2')

        # Check the outputs after running to test the unit conversions
        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_near_equal(prob['src.x2'], 100.0, 1e-6)
        assert_near_equal(prob['tgtF.x3'], 212.0, 1e-6)

        # Check the total derivatives in forward mode
        wrt = ['px1.x1']
        of = ['tgtF.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_near_equal(J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_near_equal(J['tgtF.x3', 'px1.x1'][0][0], 1.8, 1e-6)

    def test_basic_fd_comps(self):

        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0), promotes=['x1'])
        prob.model.add_subsystem('src', SrcCompFD())
        prob.model.add_subsystem('tgtF', TgtCompFFD())
        prob.model.add_subsystem('tgtC', TgtCompCFD())
        prob.model.add_subsystem('tgtK', TgtCompKFD())
        prob.model.connect('x1', 'src.x1')
        prob.model.connect('src.x2', 'tgtF.x2')
        prob.model.connect('src.x2', 'tgtC.x2')
        prob.model.connect('src.x2', 'tgtK.x2')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['src.x2'], 100.0, 1e-6)
        assert_near_equal(prob['tgtF.x3'], 212.0, 1e-6)
        assert_near_equal(prob['tgtC.x3'], 100.0, 1e-6)
        assert_near_equal(prob['tgtK.x3'], 373.15, 1e-6)

        indep_list = ['x1']
        unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_totals(of=unknown_list, wrt=indep_list, return_format='dict')

        assert_near_equal(J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=unknown_list, wrt=indep_list, return_format='dict')

        assert_near_equal(J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        prob.model.approx_totals(method='fd')
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=unknown_list, wrt=indep_list, return_format='dict')

        assert_near_equal(J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        # Make sure check partials handles conversion
        data = prob.check_partials(out_stream=None)

        for key1, val1 in data.items():
            for key2, val2 in val1.items():
                assert_near_equal(val2['abs error'][0], 0.0, 1e-6)
                assert_near_equal(val2['rel error'][0], 0.0, 1e-6)

    def test_bad_units(self):
        """Test error handling when invalid units are declared."""
        class Comp1(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', 0.0, units='junk')

        class Comp2(om.ExplicitComponent):
            def setup(self):
                self.add_output('x', 0.0, units='junk')

        with self.assertRaises(Exception) as cm:
            prob = om.Problem(model=Comp1())
            prob.setup()
        expected_msg = "The units 'junk' are invalid"
        self.assertTrue(expected_msg in str(cm.exception))

        with self.assertRaises(Exception) as cm:
            prob = om.Problem(model=Comp2())
            prob.setup()
        expected_msg = "The units 'junk' are invalid"
        self.assertTrue(expected_msg in str(cm.exception))

    def test_incompatible_units(self):
        """Test error handling when only one of src and tgt have units."""
        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0), promotes_outputs=['x1'])
        prob.model.add_subsystem('src', SrcComp(), promotes_inputs=['x1'])
        prob.model.add_subsystem('tgt', om.ExecComp('yy=xx', xx={'value': 0.0, 'units': None}))
        prob.model.connect('src.x2', 'tgt.xx')

        msg = "<model> <class Group>: Output 'src.x2' with units of 'degC' is connected to input 'tgt.xx' which has no units."

        with assert_warning(UserWarning, msg):
            prob.setup()

    def test_basic_implicit_conn(self):
        """Test units with all implicit connections."""
        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0), promotes_outputs=['x1'])
        prob.model.add_subsystem('src', SrcComp(), promotes_inputs=['x1'], promotes_outputs=['x2'])
        prob.model.add_subsystem('tgtF', TgtCompF(), promotes_inputs=['x2'])
        prob.model.add_subsystem('tgtC', TgtCompC(), promotes_inputs=['x2'])
        prob.model.add_subsystem('tgtK', TgtCompK(), promotes_inputs=['x2'])

        # Check the outputs after running to test the unit conversions
        prob.setup()
        prob.run_model()

        assert_near_equal(prob['x2'], 100.0, 1e-6)
        assert_near_equal(prob['tgtF.x3'], 212.0, 1e-6)
        assert_near_equal(prob['tgtC.x3'], 100.0, 1e-6)
        assert_near_equal(prob['tgtK.x3'], 373.15, 1e-6)

        # Check the total derivatives in forward mode
        wrt = ['x1']
        of = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_near_equal(J['tgtF.x3', 'x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3', 'x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3', 'x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        assert_near_equal(J['tgtF.x3', 'x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3', 'x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3', 'x1'][0][0], 1.0, 1e-6)

    def test_basic_grouped(self):

        prob = om.Problem()
        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0), promotes=['x1'])
        sub1 = prob.model.add_subsystem('sub1', om.Group())
        sub2 = prob.model.add_subsystem('sub2', om.Group())

        sub1.add_subsystem('src', SrcComp())
        sub2.add_subsystem('tgtF', TgtCompF())
        sub2.add_subsystem('tgtC', TgtCompC())
        sub2.add_subsystem('tgtK', TgtCompK())

        prob.model.connect('x1', 'sub1.src.x1')
        prob.model.connect('sub1.src.x2', 'sub2.tgtF.x2')
        prob.model.connect('sub1.src.x2', 'sub2.tgtC.x2')
        prob.model.connect('sub1.src.x2', 'sub2.tgtK.x2')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['sub1.src.x2'], 100.0, 1e-6)
        assert_near_equal(prob['sub2.tgtF.x3'], 212.0, 1e-6)
        assert_near_equal(prob['sub2.tgtC.x3'], 100.0, 1e-6)
        assert_near_equal(prob['sub2.tgtK.x3'], 373.15, 1e-6)

        wrt = ['x1']
        of = ['sub2.tgtF.x3', 'sub2.tgtC.x3', 'sub2.tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_near_equal(J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_near_equal(J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    def test_basic_grouped_bug_from_pycycle(self):

        prob = om.Problem()
        root = prob.model

        prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0), promotes=['x1'])
        sub1 = prob.model.add_subsystem('sub1', om.Group(), promotes=['x2'])
        sub1.add_subsystem('src', SrcComp(), promotes=['x2'])
        root.add_subsystem('tgtF', TgtCompFMulti())
        root.add_subsystem('tgtC', TgtCompC())
        root.add_subsystem('tgtK', TgtCompK())

        prob.model.connect('x1', 'sub1.src.x1')
        prob.model.connect('x2', 'tgtF.x2')
        prob.model.connect('x2', 'tgtC.x2')
        prob.model.connect('x2', 'tgtK.x2')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['x2'], 100.0, 1e-6)
        assert_near_equal(prob['tgtF.x3'], 212.0, 1e-6)
        assert_near_equal(prob['tgtC.x3'], 100.0, 1e-6)
        assert_near_equal(prob['tgtK.x3'], 373.15, 1e-6)

        wrt = ['x1']
        of = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_near_equal(J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        # Check the total derivatives in reverse mode
        prob.setup(check=False, mode='rev')
        prob.run_model()
        J = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_near_equal(J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        assert_near_equal(J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        assert_near_equal(J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    #def test_basic_grouped_grouped_implicit(self):

        #prob = om.Problem()
        #root = prob.model
        #sub1 = prob.model.add_subsystem('sub1', om.Group(), promotes=['x2'])
        #sub2 = prob.model.add_subsystem('sub2', om.Group(), promotes=['x2'])
        #sub1.add_subsystem('src', SrcComp(), promotes = ['x2'])
        #sub2.add_subsystem('tgtF', TgtCompFMulti(), promotes=['x2'])
        #sub2.add_subsystem('tgtC', TgtCompC(), promotes=['x2'])
        #sub2.add_subsystem('tgtK', TgtCompK(), promotes=['x2'])
        #prob.model.add_subsystem('px1', om.IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.model.connect('x1', 'sub1.src.x1')

        #prob.setup()
        #prob.run_model()

        #assert_near_equal(prob['x2'], 100.0, 1e-6)
        #assert_near_equal(prob['sub2.tgtF.x3'], 212.0, 1e-6)
        #assert_near_equal(prob['sub2.tgtC.x3'], 100.0, 1e-6)
        #assert_near_equal(prob['sub2.tgtK.x3'], 373.15, 1e-6)

        #indep_list = ['x1']
        #unknown_list = ['sub2.tgtF.x3', 'sub2.tgtC.x3', 'sub2.tgtK.x3']
        #J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               #return_format='dict')

        #assert_near_equal(J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_near_equal(J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_near_equal(J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               #return_format='dict')

        #assert_near_equal(J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_near_equal(J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_near_equal(J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               #return_format='dict')

        #assert_near_equal(J['sub2.tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_near_equal(J['sub2.tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_near_equal(J['sub2.tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    #def test_apply_linear_adjoint(self):
        ## Make sure we can index into dinputs

        #class Attitude_Angular(Component):
            #""" Calculates angular velocity vector from the satellite's orientation
            #matrix and its derivative.
            #"""

            #def __init__(self, n=2):
                #super().__init__()

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

        #prob = om.Problem()
        #root = prob.model
        #prob.model.add_subsystem('comp', Attitude_Angular(n=5), promotes=['*'])
        #prob.model.add_subsystem('p1', om.IndepVarComp('O_BI', np.ones((3, 3, 5))), promotes=['*'])
        #prob.model.add_subsystem('p2', om.IndepVarComp('Odot_BI', np.ones((3, 3, 5))), promotes=['*'])

        #prob.setup()
        #prob.run_model()

        #indep_list = ['O_BI', 'Odot_BI']
        #unknown_list = ['w_B']
        #Jf = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                                #return_format='dict')

        #indep_list = ['O_BI', 'Odot_BI']
        #unknown_list = ['w_B']
        #Jr = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                                #return_format='dict')

        #for key, val in Jr.items():
            #for key2 in val:
                #diff = abs(Jf[key][key2] - Jr[key][key2])
                #assert_near_equal(diff, 0.0, 1e-10)

    def test_incompatible_connections(self):

        class BadComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x2', 100.0, units='m')
                self.add_output('x3', 100.0)

        # Explicit Connection
        prob = om.Problem()
        prob.model.add_subsystem('src', SrcComp())
        prob.model.add_subsystem('dest', BadComp())
        prob.model.connect('src.x2', 'dest.x2')
        with self.assertRaises(Exception) as cm:
            prob.setup()

        expected_msg = "<model> <class Group>: Output units of 'degC' for 'src.x2' are incompatible with input units of 'm' for 'dest.x2'."

        self.assertEqual(expected_msg, str(cm.exception))

        # Implicit Connection
        prob = om.Problem()
        prob.model.add_subsystem('src', SrcComp(), promotes=['x2'])
        prob.model.add_subsystem('dest', BadComp(),promotes=['x2'])
        with self.assertRaises(Exception) as cm:
            prob.setup()

        expected_msg = "<model> <class Group>: Output units of 'degC' for 'src.x2' are incompatible with input units of 'm' for 'dest.x2'."

        self.assertEqual(expected_msg, str(cm.exception))

    #def test_nested_relevancy_base(self):

        ## This one actually has nothing to do with units, but it tests the
        ## "rest" of the problem that the others are testing, namely that
        ## outscope vars could sometimes cause a problem even absent any units.

        #prob = om.Problem()
        #root = prob.model
        #root.add_subsystem('p1', om.IndepVarComp('xx', 3.0))
        #root.add_subsystem('c1', om.ExecComp(['y1=0.5*x + 0.1*xx', 'y2=0.3*x - 1.0*xx']))
        #root.add_subsystem('c2', om.ExecComp(['y=0.5*x']))
        #sub = root.add_subsystem('sub', om.Group())
        #sub.add_subsystem('cc1', om.ExecComp(['y=0.1*x1 + 0.01*x2']))
        #sub.add_subsystem('cc2', om.ExecComp(['y=0.1*x']))

        #root.connect('p1.xx', 'c1.xx')
        #root.connect('c1.y1', 'c2.x')
        ##root.connect('c2.y', 'c1.x')
        #root.connect('c1.y2', 'sub.cc1.x1')
        #root.connect('sub.cc1.y', 'sub.cc2.x')
        #root.connect('sub.cc2.y', 'sub.cc1.x2')

        #root.nonlinear_solver = Newton()
        #root.linear_solver = ScipyKrylov()

        #sub.nonlinear_solver = Newton()
        #sub.linear_solver = om.DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('c1.y2')

        #prob.setup()

        #prob.run_model()

        ## Pollute the dpvec
        #sub.dpmat[None]['cc1.x1'] = 1e10

        ## Make sure we can calculate a good derivative in the presence of pollution

        #sub.linear_solver.rel_inputs = ['sub.cc2.x', 'sub.cc1.x2']
        #rhs_buf = {None : np.array([3.5, 1.7])}
        #sol_buf = sub.linear_solver.solve(rhs_buf, sub, mode='fwd')[None]
        #assert_near_equal(sol_buf[0], -3.52052052, 1e-3)
        #assert_near_equal(sol_buf[1], -2.05205205, 1e-3)

    #def test_nested_relevancy(self):

        ## This test is just to make sure that values in the dp vector from
        ## higher scopes aren't sitting there converting themselves during sub
        ## iterations.

        #prob = om.Problem()
        #root = prob.model
        #root.add_subsystem('p1', om.IndepVarComp('xx', 3.0))
        #root.add_subsystem('c1', om.ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2' : 'km'}))
        #root.add_subsystem('c2', om.ExecComp(['y=0.5*x']))
        #sub = root.add_subsystem('sub', om.Group())
        #sub.add_subsystem('cc1', om.ExecComp(['y=1.01*x1 + 1.01*x2'], units={'x1' : 'nm'}))
        #sub.add_subsystem('cc2', om.ExecComp(['y=1.01*x']))

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
        #sub.linear_solver = om.DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup()

        #prob.run_model()
        #self.assertTrue(not np.isnan(prob['sub.cc2.y']))

    #def test_nested_relevancy_adjoint(self):

        ## This test is just to make sure that values in the dp vector from
        ## higher scopes aren't sitting there converting themselves during sub
        ## iterations.

        #prob = om.Problem()
        #root = prob.model
        #root.add_subsystem('p1', om.IndepVarComp('xx', 3.0))
        #root.add_subsystem('c1', om.ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2' : 'km'}))
        #root.add_subsystem('c2', om.ExecComp(['y=0.5*x']))
        #sub = root.add_subsystem('sub', om.Group())
        #sub.add_subsystem('cc1', om.ExecComp(['y=1.01*x1 + 1.01*x2'], units={'x1' : 'nm'}))
        #sub.add_subsystem('cc2', om.ExecComp(['y=1.01*x']))

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
        #sub.linear_solver = om.DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup()

        #prob.run_model()
        #self.assertTrue(not np.isnan(prob['sub.cc2.y']))

    #def test_nested_relevancy_adjoint_apply_linear(self):

        ## This test is just to make sure that values in the dp vector from
        ## higher scopes aren't sitting there converting themselves during sub
        ## iterations.

        #class TestComp(Component):

            #def __init__(self):
                #super().__init__()

                ## Params
                #self.add_param('x1', 1.0, units='mm')
                #self.add_param('x2', 1.0)

                ## Unknowns
                #self.add_output('y', 1.0)

                #self.dx1count = 0
                #self.dx2count = 0

            #def solve_nonlinear(self, inputs, outputs, resids):
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

        #prob = om.Problem()
        #root = prob.model
        #root.add_subsystem('p1', om.IndepVarComp('xx', 3.0))
        #root.add_subsystem('c1', om.ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2' : 'km'}))
        #root.add_subsystem('c2', om.ExecComp(['y=0.5*x']))
        #sub = root.add_subsystem('sub', om.Group())
        #sub.add_subsystem('cc1', TestComp())
        #sub.add_subsystem('cc2', om.ExecComp(['y=1.01*x']))

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
        #sub.linear_solver = om.DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup()
        #prob.run_model()

        ## x1 deriv code should be called less if the dinputs vec only
        ## considers sub relevancy
        #self.assertTrue(sub.cc1.dx1count < sub.cc1.dx2count)

    #def test_nested_relevancy_gmres(self):

        ## This test is just to make sure that values in the dp vector from
        ## higher scopes aren't sitting there converting themselves during sub
        ## iterations.

        #prob = om.Problem()
        #root = prob.model
        #root.add_subsystem('p1', om.IndepVarComp('xx', 3.0))
        #root.add_subsystem('c1', om.ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2' : 'km'}))
        #root.add_subsystem('c2', om.ExecComp(['y=0.5*x']))
        #sub = root.add_subsystem('sub', om.Group())
        #sub.add_subsystem('cc1', om.ExecComp(['y=1.01*x1 + 1.01*x2'], units={'x1' : 'fm'}))
        #sub.add_subsystem('cc2', om.ExecComp(['y=1.01*x']))

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

        #prob.setup()

        #prob.run_model()

        ## GMRES doesn't cause a successive build-up in the value of an out-of
        ## scope param, but the linear solver doesn't converge. We can test to
        ## make sure it does.
        #iter_count = sub.linear_solver.iter_count
        #self.assertTrue(iter_count < 20)
        #self.assertTrue(not np.isnan(prob['sub.cc2.y']))

    #def test_nested_relevancy_gmres_precon(self):

        ## Make sure preconditioners also work

        #prob = om.Problem()
        #root = prob.model
        #root.add_subsystem('p1', om.IndepVarComp('xx', 3.0))
        #root.add_subsystem('c1', om.ExecComp(['y1=0.5*x + 1.0*xx', 'y2=0.3*x - 1.0*xx'], units={'y2' : 'km'}))
        #root.add_subsystem('c2', om.ExecComp(['y=0.5*x']))
        #sub = root.add_subsystem('sub', om.Group())
        #sub.add_subsystem('cc1', om.ExecComp(['y=1.01*x1 + 1.01*x2'], units={'x1' : 'fm'}))
        #sub.add_subsystem('cc2', om.ExecComp(['y=1.01*x']))

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
        #sub.linear_solver.precon = om.DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup()

        #prob.run_model()

        ## GMRES doesn't cause a successive build-up in the value of an out-of
        ## scope param, but the linear solver doesn't converge. We can test to
        ## make sure it does.
        #iter_count = sub.linear_solver.iter_count
        #self.assertTrue(iter_count < 20)
        #self.assertTrue(not np.isnan(prob['sub.cc2.y']))

    def test_promotes_equivalent_units(self):
        # multiple Group.set_input_defaults calls at same tree level with conflicting units args
        p = om.Problem()

        g1 = p.model.add_subsystem("G1", om.Group(), promotes_inputs=['x'])
        g1.add_subsystem("C1", om.ExecComp("y = 2. * x * z",
                                            x={'value': 5.0, 'units': 'm/s/s'},
                                            y={'value': 1.0, 'units': None},
                                            z={'value': 1.0, 'units': 'W'}),
                                            promotes_inputs=['x', 'z'])
        g1.add_subsystem("C2", om.ExecComp("y = 3. * x * z",
                                            x={'value': 5.0, 'units': 'm/s**2'},
                                            y={'value': 1.0, 'units': None},
                                            z={'value': 1.0, 'units': 'J/s'}),
                                            promotes_inputs=['x', 'z'])
        # converting m/s/s to m/s**2 is allowed
        p.setup()

    def test_promotes_non_equivalent_units(self):
        # multiple Group.set_input_defaults calls at same tree level with conflicting units args
        p = om.Problem()

        g1 = p.model.add_subsystem("G1", om.Group(), promotes_inputs=['x'])
        g1.add_subsystem("C1", om.ExecComp("y = 2. * x * z",
                                            x={'value': 5.0, 'units': 'J/s/s'},
                                            y={'value': 1.0, 'units': None},
                                            z={'value': 1.0, 'units': 'W'}),
                                            promotes_inputs=['x', 'z'])
        g1.add_subsystem("C2", om.ExecComp("y = 3. * x * z",
                                            x={'value': 5.0, 'units': 'm/s**2'},
                                            y={'value': 1.0, 'units': None},
                                            z={'value': 1.0, 'units': 'J/s'}),
                                            promotes_inputs=['x', 'z'])
        # trying to convert J/s/s to m/s**2 should cause Incompatible units TypeError exception
        with self.assertRaises(TypeError) as e:
            p.setup()
        self.assertEqual(str(e.exception), "Units 'm/s**2' and 'J/s**2' are incompatible.")

    def test_input_defaults_unit_compat(self):
        p = om.Problem()

        p.model.add_subsystem('comp', om.ExecComp('y=2*x', units='inch'))

        with self.assertRaises(ValueError) as cm:
            p.model.set_input_defaults('comp.x', val=2., units='in**2')

        msg = ("<class Group>: The units 'in**2' are invalid.")
        self.assertEqual(cm.exception.args[0], msg)


if __name__ == "__main__":
    unittest.main()
