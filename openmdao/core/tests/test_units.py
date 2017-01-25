""" Tests the ins and outs of automatic unit conversion in OpenMDAO."""

import unittest
from six import iteritems
from six.moves import cStringIO

import numpy as np

from openmdao.api import Problem
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.groups.unit_conversion_groups import UnitConvGroup


class TestUnitConversion(unittest.TestCase):
    """ Testing automatic unit conversion."""

    def test_basic(self):

        raise unittest.SkipTest('Unit Conversion not working robustly at present.')

        prob = Problem()
        prob.model = UnitConvGroup()

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        wrt = ['x1']
        of = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['tgtF.x3', 'x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3', 'x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3', 'x1'][0][0], 1.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')

        assert_rel_error(self, J['tgtF.x3', 'x1'][0][0], 1.8, 1e-6)
        assert_rel_error(self, J['tgtC.x3', 'x1'][0][0], 1.0, 1e-6)
        assert_rel_error(self, J['tgtK.x3', 'x1'][0][0], 1.0, 1e-6)

        # TODO - support FD

        #J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        ## Need to clean up after FD gradient call, so just rerun.
        #prob.run()

        ## Make sure check partials handles conversion
        #data = prob.check_partial_derivatives(out_stream=None)

        #for key1, val1 in iteritems(data):
            #for key2, val2 in iteritems(val1):
                #assert_rel_error(self, val2['abs error'][0], 0.0, 1e-6)
                #assert_rel_error(self, val2['abs error'][1], 0.0, 1e-6)
                #assert_rel_error(self, val2['abs error'][2], 0.0, 1e-6)
                #assert_rel_error(self, val2['rel error'][0], 0.0, 1e-6)
                #assert_rel_error(self, val2['rel error'][1], 0.0, 1e-6)
                #assert_rel_error(self, val2['rel error'][2], 0.0, 1e-6)

        #stream = cStringIO()
        #conv = prob.root.list_unit_conv(stream=stream)
        #self.assertTrue((('src.x2', 'tgtF.x2'), ('degC', 'degF')) in conv)
        #self.assertTrue((('src.x2', 'tgtK.x2'), ('degC', 'degK')) in conv)

    #def test_basic_force_fd_comps(self):

        #prob = Problem()
        #prob.root = Group()
        #prob.root.add('src', SrcComp())
        #prob.root.add('tgtF', TgtCompF())
        #prob.root.add('tgtC', TgtCompC())
        #prob.root.add('tgtK', TgtCompK())
        #prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.root.connect('x1', 'src.x1')
        #prob.root.connect('src.x2', 'tgtF.x2')
        #prob.root.connect('src.x2', 'tgtC.x2')
        #prob.root.connect('src.x2', 'tgtK.x2')

        #prob.root.src.deriv_options['type'] = 'fd'
        #prob.root.tgtF.deriv_options['type'] = 'fd'
        #prob.root.tgtC.deriv_options['type'] = 'fd'
        #prob.root.tgtK.deriv_options['type'] = 'fd'

        #prob.setup(check=False)
        #prob.run()

        #assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        #assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        #assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        #assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        ## Make sure we don't convert equal units
        #self.assertEqual(prob.root.params.metadata('tgtC.x2').get('unit_conv'),
                         #None)

        #indep_list = ['x1']
        #unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        #J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        ## Need to clean up after FD gradient call, so just rerun.
        #prob.run()

        ## Make sure check partials handles conversion
        #data = prob.check_partial_derivatives(out_stream=None)

        #for key1, val1 in iteritems(data):
            #for key2, val2 in iteritems(val1):
                #assert_rel_error(self, val2['abs error'][0], 0.0, 1e-6)
                #assert_rel_error(self, val2['abs error'][1], 0.0, 1e-6)
                #assert_rel_error(self, val2['abs error'][2], 0.0, 1e-6)
                #assert_rel_error(self, val2['rel error'][0], 0.0, 1e-6)
                #assert_rel_error(self, val2['rel error'][1], 0.0, 1e-6)
                #assert_rel_error(self, val2['rel error'][2], 0.0, 1e-6)

        #stream = cStringIO()
        #conv = prob.root.list_unit_conv(stream=stream)
        #self.assertTrue((('src.x2', 'tgtF.x2'), ('degC', 'degF')) in conv)
        #self.assertTrue((('src.x2', 'tgtK.x2'), ('degC', 'degK')) in conv)

    #def test_bad_units(self):

        #class Comp1(Component):
            #def __init__(self):
                #super(Comp1, self).__init__()
                #self.add_param('x', 0.0, unit='junk')

        #class Comp2(Component):
            #def __init__(self):
                #super(Comp2, self).__init__()
                #self.add_state('x', 0.0, unit='junk')

        #class Comp3(Component):
            #def __init__(self):
                #super(Comp3, self).__init__()
                #self.add_output('x', 0.0, unit='junk')

        #top = Problem()
        #root = top.root = Group()

        #with self.assertRaises(Exception) as cm:
            #root.add('comp', Comp1())

        #expected_msg = "Unit 'junk' is not a valid unit or combination of units."
        #self.assertTrue(expected_msg in str(cm.exception))

        #with self.assertRaises(Exception) as cm:
            #root.add('comp', Comp2())

        #expected_msg = "Unit 'junk' is not a valid unit or combination of units."
        #self.assertTrue(expected_msg in str(cm.exception))

        #with self.assertRaises(Exception) as cm:
            #root.add('comp', Comp3())

        #expected_msg = "Unit 'junk' is not a valid unit or combination of units."
        #self.assertTrue(expected_msg in str(cm.exception))

    #def test_list_unit_conversions_no_unit(self):

        #prob = Problem()
        #prob.root = Group()
        #prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.root.add('src', SrcComp())
        #prob.root.add('tgt', ExecComp('yy=xx', xx=0.0))
        #prob.root.connect('src.x2', 'tgt.xx')

        #prob.setup(check=False)
        #prob.run()

        #stream = cStringIO()
        #conv = prob.root.list_unit_conv(stream=stream)
        #self.assertTrue((('src.x2', 'tgt.xx'), ('degC', None)) in conv)

    #def test_basic_input_input(self):

        #prob = Problem()
        #prob.root = Group()
        #prob.root.add('src', SrcComp())
        #prob.root.add('tgtF', TgtCompF())
        #prob.root.add('tgtC', TgtCompC())
        #prob.root.add('tgtK', TgtCompK())
        #prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.root.connect('x1', 'src.x1')
        #prob.root.connect('src.x2', 'tgtC.x2')
        #prob.root.connect('tgtC.x2', 'tgtF.x2')
        #prob.root.connect('tgtC.x2', 'tgtK.x2')

        #prob.setup(check=False)
        #prob.run()

        #assert_rel_error(self, prob['src.x2'], 100.0, 1e-6)
        #assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        #assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        #assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        ## Make sure we don't convert equal units
        #self.assertEqual(prob.root.params.metadata('tgtC.x2').get('unit_conv'),
                         #None)

        #indep_list = ['x1']
        #unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        #J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    #def test_basic_implicit_conn(self):

        #prob = Problem()
        #prob.root = Group()
        #prob.root.add('src', SrcComp(), promotes=['x1', 'x2'])
        #prob.root.add('tgtF', TgtCompF(), promotes=['x2'])
        #prob.root.add('tgtC', TgtCompC(), promotes=['x2'])
        #prob.root.add('tgtK', TgtCompK(), promotes=['x2'])
        #prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])

        #prob.setup(check=False)
        #prob.run()

        #assert_rel_error(self, prob['x2'], 100.0, 1e-6)
        #assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        #assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        #assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        ## Make sure we don't convert equal units
        #self.assertEqual(prob.root.params.metadata('tgtC.x2').get('unit_conv'),
                         #None)

        #indep_list = ['x1']
        #unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        #J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    #def test_basic_grouped(self):

        #prob = Problem()
        #prob.root = Group()
        #sub1 = prob.root.add('sub1', Group())
        #sub2 = prob.root.add('sub2', Group())
        #sub1.add('src', SrcComp())
        #sub2.add('tgtF', TgtCompF())
        #sub2.add('tgtC', TgtCompC())
        #sub2.add('tgtK', TgtCompK())
        #prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.root.connect('x1', 'sub1.src.x1')
        #prob.root.connect('sub1.src.x2', 'sub2.tgtF.x2')
        #prob.root.connect('sub1.src.x2', 'sub2.tgtC.x2')
        #prob.root.connect('sub1.src.x2', 'sub2.tgtK.x2')

        #prob.setup(check=False)
        #prob.run()

        #assert_rel_error(self, prob['sub1.src.x2'], 100.0, 1e-6)
        #assert_rel_error(self, prob['sub2.tgtF.x3'], 212.0, 1e-6)
        #assert_rel_error(self, prob['sub2.tgtC.x3'], 100.0, 1e-6)
        #assert_rel_error(self, prob['sub2.tgtK.x3'], 373.15, 1e-6)

        ## Make sure we don't convert equal units
        #self.assertEqual(prob.root.sub2.params.metadata('tgtC.x2').get('unit_conv'),
                         #None)

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

        #stream = cStringIO()
        #conv = prob.root.sub1.list_unit_conv(stream=stream)
        #self.assertTrue(len(conv) == 0)


    #def test_list_unit_connections_sub(self):

        #prob = Problem()
        #prob.root = Group()
        #sub1 = prob.root.add('sub1', Group())
        #sub2 = prob.root.add('sub2', Group())
        #sub1.add('src', SrcComp())
        #sub1.add('tgtF', TgtCompF())
        #sub2.add('tgtC', TgtCompC())
        #sub2.add('tgtK', TgtCompK())
        #prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.root.connect('x1', 'sub1.src.x1')
        #prob.root.connect('sub1.src.x2', 'sub1.tgtF.x2')
        #prob.root.connect('sub1.src.x2', 'sub2.tgtC.x2')
        #prob.root.connect('sub1.src.x2', 'sub2.tgtK.x2')

        #prob.setup(check=False)
        #prob.run()

        #stream = cStringIO()
        #conv = prob.root.sub1.list_unit_conv(stream=stream)
        #self.assertTrue((('src.x2', 'tgtF.x2'), ('degC', 'degF')) in conv)

    #def test_basic_grouped_bug_from_pycycle(self):

        #prob = Problem()
        #root = prob.root = Group()
        #sub1 = prob.root.add('sub1', Group(), promotes=['x2'])
        #sub1.add('src', SrcComp(), promotes = ['x2'])
        #root.add('tgtF', TgtCompFMulti())
        #root.add('tgtC', TgtCompC())
        #root.add('tgtK', TgtCompK())
        #prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.root.connect('x1', 'sub1.src.x1')
        #prob.root.connect('x2', 'tgtF.x2')
        #prob.root.connect('x2', 'tgtC.x2')
        #prob.root.connect('x2', 'tgtK.x2')

        #prob.setup(check=False)
        #prob.run()

        #assert_rel_error(self, prob['x2'], 100.0, 1e-6)
        #assert_rel_error(self, prob['tgtF.x3'], 212.0, 1e-6)
        #assert_rel_error(self, prob['tgtC.x3'], 100.0, 1e-6)
        #assert_rel_error(self, prob['tgtK.x3'], 373.15, 1e-6)

        #indep_list = ['x1']
        #unknown_list = ['tgtF.x3', 'tgtC.x3', 'tgtK.x3']
        #J = prob.calc_gradient(indep_list, unknown_list, mode='fwd',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='rev',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

        #J = prob.calc_gradient(indep_list, unknown_list, mode='fd',
                               #return_format='dict')

        #assert_rel_error(self, J['tgtF.x3']['x1'][0][0], 1.8, 1e-6)
        #assert_rel_error(self, J['tgtC.x3']['x1'][0][0], 1.0, 1e-6)
        #assert_rel_error(self, J['tgtK.x3']['x1'][0][0], 1.0, 1e-6)

    #def test_basic_grouped_grouped_implicit(self):

        #prob = Problem()
        #root = prob.root = Group()
        #sub1 = prob.root.add('sub1', Group(), promotes=['x2'])
        #sub2 = prob.root.add('sub2', Group(), promotes=['x2'])
        #sub1.add('src', SrcComp(), promotes = ['x2'])
        #sub2.add('tgtF', TgtCompFMulti(), promotes=['x2'])
        #sub2.add('tgtC', TgtCompC(), promotes=['x2'])
        #sub2.add('tgtK', TgtCompK(), promotes=['x2'])
        #prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.root.connect('x1', 'sub1.src.x1')

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
        ## Make sure we can index into dparams

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

            #def solve_nonlinear(self, params, unknowns, resids):
                #""" Calculate output. """

                #O_BI = params['O_BI']
                #Odot_BI = params['Odot_BI']
                #w_B = unknowns['w_B']

                #for i in range(0, self.n):
                    #w_B[0, i] = np.dot(Odot_BI[2, :, i], O_BI[1, :, i])
                    #w_B[1, i] = np.dot(Odot_BI[0, :, i], O_BI[2, :, i])
                    #w_B[2, i] = np.dot(Odot_BI[1, :, i], O_BI[0, :, i])

            #def linearize(self, params, unknowns, resids):
                #""" Calculate and save derivatives. (i.e., Jacobian) """

                #O_BI = params['O_BI']
                #Odot_BI = params['Odot_BI']

                #for i in range(0, self.n):
                    #self.dw_dOdot[i, 0, 2, :] = O_BI[1, :, i]
                    #self.dw_dO[i, 0, 1, :] = Odot_BI[2, :, i]

                    #self.dw_dOdot[i, 1, 0, :] = O_BI[2, :, i]
                    #self.dw_dO[i, 1, 2, :] = Odot_BI[0, :, i]

                    #self.dw_dOdot[i, 2, 1, :] = O_BI[0, :, i]
                    #self.dw_dO[i, 2, 0, :] = Odot_BI[1, :, i]

            #def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
                #""" Matrix-vector product with the Jacobian. """

                #dw_B = dresids['w_B']

                #if mode == 'fwd':
                    #for k in range(3):
                        #for i in range(3):
                            #for j in range(3):
                                #if 'O_BI' in dparams:
                                    #dw_B[k, :] += self.dw_dO[:, k, i, j] * \
                                        #dparams['O_BI'][i, j, :]
                                #if 'Odot_BI' in dparams:
                                    #dw_B[k, :] += self.dw_dOdot[:, k, i, j] * \
                                        #dparams['Odot_BI'][i, j, :]

                #else:

                    #for k in range(3):
                        #for i in range(3):
                            #for j in range(3):

                                #if 'O_BI' in dparams:
                                    #dparams['O_BI'][i, j, :] += self.dw_dO[:, k, i, j] * \
                                        #dw_B[k, :]

                                #if 'Odot_BI' in dparams:
                                    #dparams['Odot_BI'][i, j, :] -= -self.dw_dOdot[:, k, i, j] * \
                                        #dw_B[k, :]

        #prob = Problem()
        #root = prob.root = Group()
        #prob.root.add('comp', Attitude_Angular(n=5), promotes=['*'])
        #prob.root.add('p1', IndepVarComp('O_BI', np.ones((3, 3, 5))), promotes=['*'])
        #prob.root.add('p2', IndepVarComp('Odot_BI', np.ones((3, 3, 5))), promotes=['*'])

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

    #def test_incompatible_connections(self):

        #class BadComp(Component):
            #def __init__(self):
                #super(BadComp, self).__init__()

                #self.add_param('x2', 100.0, units='m')
                #self.add_output('x3', 100.0)

        ## Explicit Connection
        #prob = Problem()
        #prob.root = Group()
        #prob.root.add('src', SrcComp())
        #prob.root.add('dest', BadComp())
        #prob.root.connect('src.x2', 'dest.x2')
        #with self.assertRaises(Exception) as cm:
            #prob.setup(check=False)

        #expected_msg = "Unit 'degC' in source 'src.x2' is incompatible with unit 'm' in target 'dest.x2'."

        #self.assertTrue(expected_msg in str(cm.exception))

        ## Implicit Connection
        #prob = Problem()
        #prob.root = Group()
        #prob.root.add('src', SrcComp(), promotes=['x2'])
        #prob.root.add('dest', BadComp(),promotes=['x2'])
        #with self.assertRaises(Exception) as cm:
            #prob.setup(check=False)

        #expected_msg = "Unit 'degC' in source 'src.x2' (x2) is incompatible with unit 'm' in target 'dest.x2' (x2)."

        #self.assertTrue(expected_msg in str(cm.exception))

    #def test_nested_relevancy_base(self):

        ## This one actually has nothing to do with units, but it tests the
        ## "rest" of the problem that the others are testing, namely that
        ## outscope vars could sometimes cause a problem even absent any units.

        #prob = Problem()
        #root = prob.root = Group()
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

        #root.nl_solver = Newton()
        #root.ln_solver = ScipyGMRES()

        #sub.nl_solver = Newton()
        #sub.ln_solver = DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('c1.y2')

        #prob.setup(check=False)

        #prob.run()

        ## Pollute the dpvec
        #sub.dpmat[None]['cc1.x1'] = 1e10

        ## Make sure we can calculate a good derivative in the presence of pollution

        #sub._jacobian_changed = True
        #sub.ln_solver.rel_inputs = ['sub.cc2.x', 'sub.cc1.x2']
        #rhs_buf = {None : np.array([3.5, 1.7])}
        #sol_buf = sub.ln_solver.solve(rhs_buf, sub, mode='fwd')[None]
        #assert_rel_error(self, sol_buf[0], -3.52052052, 1e-3)
        #assert_rel_error(self, sol_buf[1], -2.05205205, 1e-3)

    #def test_nested_relevancy(self):

        ## This test is just to make sure that values in the dp vector from
        ## higher scopes aren't sitting there converting themselves during sub
        ## iterations.

        #prob = Problem()
        #root = prob.root = Group()
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

        #root.nl_solver = Newton()
        #root.nl_solver.options['maxiter'] = 1
        #root.ln_solver = ScipyGMRES()
        #root.ln_solver.options['maxiter'] = 1

        #sub.nl_solver = Newton()
        #sub.ln_solver = DirectSolver()

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
        #root = prob.root = Group()
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

        #root.nl_solver = Newton()
        #root.nl_solver.options['maxiter'] = 1
        #root.ln_solver = ScipyGMRES()
        #root.ln_solver.options['maxiter'] = 1
        #root.ln_solver.options['mode'] = 'rev'

        #sub.nl_solver = Newton()
        #sub.ln_solver = DirectSolver()

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

            #def solve_nonlinear(self, params, unknowns, resids):
                #""" Doesn't do much. """
                #x1 = params['x1']
                #x2 = params['x2']
                #unknowns['y'] = 1.01*(x1 + x2)

            #def apply_linear(self, params, unknowns, dparams, dunknowns, dresids,
                             #mode):
                #"""Returns the product of the incoming vector with the Jacobian."""

                #if mode == 'fwd':
                    #if 'x1' in dparams:
                        #dresids['y'] += 1.01*dparams['x1']
                        #self.dx1count += 1
                    #if 'x2' in dparams:
                        #dresids['y'] += 1.01*dparams['x2']
                        #self.dx2count += 1

                #elif mode == 'rev':
                    #if 'x1' in dparams:
                        #dparams['x1'] = 1.01*dresids['y']
                        #self.dx1count += 1
                    #if 'x2' in dparams:
                        #dparams['x2'] = 1.01*dresids['y']
                        #self.dx2count += 1

        #prob = Problem()
        #root = prob.root = Group()
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

        #root.nl_solver = Newton()
        #root.nl_solver.options['maxiter'] = 1
        #root.ln_solver = ScipyGMRES()
        #root.ln_solver.options['maxiter'] = 1
        #root.ln_solver.options['mode'] = 'rev'

        #sub.nl_solver = Newton()
        #sub.ln_solver = DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup(check=False)
        #prob.run()

        ## x1 deriv code should be called less if the dparams vec only
        ## considers sub relevancy
        #self.assertTrue(sub.cc1.dx1count < sub.cc1.dx2count)

    #def test_nested_relevancy_gmres(self):

        ## This test is just to make sure that values in the dp vector from
        ## higher scopes aren't sitting there converting themselves during sub
        ## iterations.

        #prob = Problem()
        #root = prob.root = Group()
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

        #root.nl_solver = Newton()
        #root.nl_solver.options['maxiter'] = 1
        #root.ln_solver = ScipyGMRES()
        #root.ln_solver.options['maxiter'] = 1

        #sub.nl_solver = Newton()
        #sub.ln_solver = ScipyGMRES()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup(check=False)

        #prob.run()

        ## GMRES doesn't cause a successive build-up in the value of an out-of
        ## scope param, but the linear solver doesn't converge. We can test to
        ## make sure it does.
        #iter_count = sub.ln_solver.iter_count
        #self.assertTrue(iter_count < 20)
        #self.assertTrue(not np.isnan(prob['sub.cc2.y']))

    #def test_nested_relevancy_gmres_precon(self):

        ## Make sure preconditioners also work

        #prob = Problem()
        #root = prob.root = Group()
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

        #root.nl_solver = Newton()
        #root.nl_solver.options['maxiter'] = 1
        #root.ln_solver = ScipyGMRES()
        #root.ln_solver.options['maxiter'] = 1

        #sub.nl_solver = Newton()
        #sub.ln_solver = ScipyGMRES()
        #sub.ln_solver.precon = DirectSolver()

        #prob.driver.add_desvar('p1.xx')
        #prob.driver.add_objective('sub.cc2.y')

        #prob.setup(check=False)

        #prob.run()

        ## GMRES doesn't cause a successive build-up in the value of an out-of
        ## scope param, but the linear solver doesn't converge. We can test to
        ## make sure it does.
        #iter_count = sub.ln_solver.iter_count
        #self.assertTrue(iter_count < 20)
        #self.assertTrue(not np.isnan(prob['sub.cc2.y']))

#class PBOSrcComp(Component):

    #def __init__(self):
        #super(PBOSrcComp, self).__init__()

        #self.add_param('x1', 100.0)
        #self.add_output('x2', 100.0, units='degC', pass_by_obj=True)
        #self.deriv_options['type'] = 'fd'

    #def solve_nonlinear(self, params, unknowns, resids):
        #""" No action."""
        #unknowns['x2'] = params['x1']


#class PBOTgtCompF(Component):

    #def __init__(self):
        #super(PBOTgtCompF, self).__init__()

        #self.add_param('x2', 100.0, units='degF', pass_by_obj=True)
        #self.add_output('x3', 100.0)
        #self.deriv_options['type'] = 'fd'

    #def solve_nonlinear(self, params, unknowns, resids):
        #""" No action."""
        #unknowns['x3'] = params['x2']


#class TestUnitConversionPBO(unittest.TestCase):
    #""" Tests support for unit conversions on pass_by_obj connections."""

    #def test_basic(self):

        #prob = Problem()
        #prob.root = Group()
        #prob.root.add('src', PBOSrcComp())
        #prob.root.add('tgtF', PBOTgtCompF())
        #prob.root.add('px1', IndepVarComp('x1', 100.0), promotes=['x1'])
        #prob.root.connect('x1', 'src.x1')
        #prob.root.connect('src.x2', 'tgtF.x2')

        #prob.root.deriv_options['type'] = 'fd'

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
        #conv = prob.root.list_unit_conv(stream=stream)
        #self.assertTrue((('src.x2', 'tgtF.x2'), ('degC', 'degF')) in conv)


    #def test_radian_bug(self):

        #class Src(Component):

            #def __init__(self):
                #super(Src, self).__init__()

                #self.add_output('x1', 180.0, units='deg')
                #self.add_output('x2', np.pi, units='rad')
                #self.add_output('x3', 2.0, units='m')
                #self.deriv_options['type'] = 'fd'

            #def solve_nonlinear(self, params, unknowns, resids):
                #""" No action."""
                #pass


        #class Tgt(Component):

            #def __init__(self):
                #super(Tgt, self).__init__()

                #self.add_param('x1', 0.0, units='rad')
                #self.add_param('x2', 0.0, units='deg')
                #self.add_param('x3', 0.0, units='ft')
                #self.deriv_options['type'] = 'fd'

            #def solve_nonlinear(self, params, unknowns, resids):
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
