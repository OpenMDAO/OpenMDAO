from __future__ import print_function, division, absolute_import

import sys
import unittest
import time
import random
from distutils.version import LooseVersion
from collections import OrderedDict
from six import iteritems
from six.moves import cStringIO as StringIO

import numpy as np

import openmdao.api as om
from openmdao.utils.mpi import MPI
from openmdao.utils.assert_utils import assert_rel_error


if MPI:
    try:
        from openmdao.vectors.petsc_vector import PETScVector
    except ImportError:
        PETScVector = None


class InputParameterComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(name='input_parameter_options',
                             desc='Dictionary of options for the input parameters')

    def setup(self):
        name_prefix = 'input_parameters'

        for param_name, options in iteritems(self.options['input_parameter_options']):

            input_name = '{0}:{1}'.format(name_prefix, param_name)
            output_name = '{0}:{1}_out'.format(name_prefix, param_name)

            shape = (1, 1)
            size = np.prod(shape)

            val = np.atleast_1d(np.asarray(np.zeros(1)))
            default_val = val * np.ones(shape)

            self.add_input(input_name, val=default_val,
                           shape=shape)

            self.add_output(output_name, val=default_val,
                            shape=shape)

    def compute(self, inputs, outputs):
        pass


class ForwardFlightPhase(om.Group):

    def __init__(self, from_phase=None, **kwargs):
        self.user_input_parameter_options = {}
        super(ForwardFlightPhase, self).__init__(**kwargs)

    def add_input_parameter(self, name, targets=None):

        self.user_input_parameter_options[name] = {'name': name}
        self.user_input_parameter_options[name]['targets'] = (targets,)

    def setup(self):
        self.input_parameter_options = {}

        for ip in list(self.user_input_parameter_options.keys()):
            self.input_parameter_options[ip] = {}
            self.input_parameter_options[ip].update(self.user_input_parameter_options[ip])

        passthru = InputParameterComp(input_parameter_options=self.input_parameter_options)

        self.add_subsystem('input_params', subsys=passthru, promotes_inputs=['*'])


class StaticAnalysis(om.Group):

    def setup(self):
        self.add_subsystem('drymass',
                           om.ExecComp('dry_mass = m_empty', m_empty=6),
                           promotes=['*'])

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 1
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 1


def _get_problem():
    print_level = 0

    p = om.Problem(model=om.Group())

    desvars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
    desvars.add_output('P_tms', 150)

    p.model.add_design_var('P_tms', lower=1, upper=350)

    p.model.add_subsystem('static_analysis',
                        StaticAnalysis(),
                        promotes_outputs=['*'])

    hop0 = p.model.add_subsystem('hop0', om.ParallelGroup())

    takeoff = hop0.add_subsystem('takeoff', ForwardFlightPhase())
    takeoff.add_input_parameter('dry_mass', targets='dry_mass')
    p.model.connect('dry_mass', 'hop0.takeoff.input_parameters:dry_mass')

    forward_flight = hop0.add_subsystem('forward_flight', ForwardFlightPhase())
    forward_flight.add_input_parameter('dry_mass', targets='dry_mass')
    p.model.connect('dry_mass', 'hop0.forward_flight.input_parameters:dry_mass')


    EQ_constraint = om.ExecComp('y1=x1')
    p.model.add_subsystem('EQ_constraint', EQ_constraint)
    p.model.connect('P_tms', 'EQ_constraint.x1')

    p.setup(mode='auto', check=['unconnected_inputs'], force_alloc_complex=True)

    p.set_solver_print(level=print_level)

    p.run_model()

    return p


class SerialTestCase(unittest.TestCase):

    def test_serial_model(self):
        p = _get_problem()
        J = p.compute_totals(of=['EQ_constraint.y1'], wrt=['P_tms'], return_format='array', debug_print=False)
        np.testing.assert_allclose(J, np.array([[1.0]]), rtol=1e-7, atol=0, equal_nan=True,
                                   err_msg='', verbose=True)

@unittest.skipUnless(MPI and PETScVector, "only run with MPI and PETSc.")
class ParallelTestCase(unittest.TestCase):

    N_PROCS = 2

    def test_mpi_bug(self):
        p = _get_problem()
        J = p.compute_totals(of=['EQ_constraint.y1'], wrt=['P_tms'], return_format='array', debug_print=False)
        np.testing.assert_allclose(J, np.array([[1.0]]), rtol=1e-7, atol=0, equal_nan=True,
                                   err_msg='', verbose=True)

if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
