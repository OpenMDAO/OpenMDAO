"""
Test reproduces two bugs found with a complicated DYMOS model.
"""
from __future__ import print_function, division

from six import iteritems
import unittest

import numpy as np
from scipy.linalg import block_diag

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, IndepVarComp, ExplicitComponent
from openmdao.utils.options_dictionary import OptionsDictionary


def lagrange_matrices(x_disc, x_interp):
    nd = len(x_disc)
    ni = len(x_interp)
    Li = np.ones((ni, nd))
    Di = np.ones((ni, nd))
    return Li, Di


class MyOptions(OptionsDictionary):

    def __init__(self, read_only=False):
        super(MyOptions, self).__init__(read_only)

        self.declare(name='rate_param', allow_none=True, default=None)
        self.declare(name='shape', default=(1,))
        self.declare(name='rate_source')
        self.declare(name='lower', default=None, allow_none=True)
        self.declare(name='upper', default=None, allow_none=True)


def gauss_lobatto_subsets(n, first_seg=False):

    subsets = {
        'disc': np.arange(0, n, 2, dtype=int),
        'state_disc': np.arange(0, n, 2, dtype=int),
        'state_input': np.arange(0, n, 2, dtype=int) if first_seg
        else np.arange(2, n, 2, dtype=int),
        'control_disc': np.arange(n, dtype=int),
        'control_input': np.arange(n, dtype=int) if first_seg
        else np.arange(1, n, dtype=int),
        'segment_ends': np.array([0, n-1], dtype=int),
        'col': np.arange(1, n, 2, dtype=int),
        'all': np.arange(n, dtype=int),
    }

    return subsets


class GaussLobattoPathConstraintComp(ExplicitComponent):

    def initialize(self):
        self._path_constraints = []
        self._vars = []
        self.options.declare('grid_data', types=GridData)

    def _add_path_constraint(self, name, shape=(1,), lower=None, upper=None, ):
        kwargs = {'shape': shape, 'lower': lower, 'upper': upper}
        self._path_constraints.append((name, kwargs))

    def setup(self):
        grid_data = self.options['grid_data']

        num_nodes = grid_data.num_nodes
        for (name, kwargs) in self._path_constraints:
            shape = kwargs['shape']

            all_input_name = ''
            disc_input_name = 'disc_values:{0}'.format(name)
            col_input_name = 'col_values:{0}'.format(name)

            self.add_input(disc_input_name, shape=(3, ))
            self.add_input(col_input_name, shape=(2, ))

            output_name = 'path:{0}'.format(name)
            output_kwargs = {}
            output_kwargs['shape'] = (num_nodes, )
            self.add_output(output_name, **output_kwargs)

            self._vars.append((disc_input_name, col_input_name, all_input_name, output_name))

            constraint_kwargs = {k: kwargs.get(k, None)
                                 for k in ('lower', 'upper')}
            self.add_constraint(output_name, **constraint_kwargs)

            disc_shape = (3, )

            var_size = np.prod(shape)
            disc_size = np.prod(disc_shape)
            disc_rows = []
            for i in [0, 2, 4]:
                disc_rows.extend(range(i, i + var_size))
            disc_rows = np.asarray(disc_rows, dtype=int)

            self.declare_partials(
                of=output_name,
                wrt=disc_input_name,
                rows=disc_rows,
                cols=np.arange(disc_size),
                val=1.0)


class StateInterpComp(ExplicitComponent):

    def initialize(self):

        self.options.declare('grid_data', types=GridData,)
        self.options.declare('state_options', types=dict)

    def setup(self):

        state_options = self.options['state_options']

        self.add_input(name='dt_dstau', shape=(2, ))

        self.xd_str = {}
        self.fd_str = {}
        self.xdotc_str = {}

        for state_name, options in state_options.items():
            shape = options['shape']

            self.add_input(
                name='state_disc:{0}'.format(state_name),
                shape=(3,) + shape)

            self.add_input(
                name='staterate_disc:{0}'.format(state_name),
                shape=(3,) + shape)

            self.add_output(
                name='staterate_col:{0}'.format(state_name),
                shape=(2, ) + shape)

            self.xd_str[state_name] = 'state_disc:{0}'.format(state_name)
            self.fd_str[state_name] = 'staterate_disc:{0}'.format(state_name)
            self.xdotc_str[state_name] = 'staterate_col:{0}'.format(state_name)

        Ad, Bd = self.options['grid_data'].phase_hermite_matrices('state_disc', 'col')

        self.matrices = {'Ad': Ad, 'Bd': Bd}

        # Setup partials

        self.jacs = {'Ad': {}, 'Bd': {}}
        self.Ad_rows = {}
        self.Ad_cols = {}
        self.sizes = {}
        for name, options in iteritems(state_options):
            shape = options['shape']

            size = np.prod(shape)

            for key in self.jacs:
                jac = np.zeros((2, size, 3, size))
                for i in range(size):
                    jac[:, i, :, i] = self.matrices[key]
                jac = jac.reshape((2 * size, 3 * size), order='C')
                self.jacs[key][name] = jac

            self.sizes[name] = size

            Bd_rows, Bd_cols = np.where(self.jacs['Bd'][name] != 0)
            self.declare_partials(of=self.xdotc_str[name], wrt=self.fd_str[name],
                                  rows=Bd_rows, cols=Bd_cols,
                                  val=self.jacs['Bd'][name][Bd_rows, Bd_cols])

            self.Ad_rows[name], self.Ad_cols[name] = np.where(self.jacs['Ad'][name] != 0)
            self.declare_partials(of=self.xdotc_str[name], wrt=self.xd_str[name],
                                  rows=self.Ad_rows[name], cols=self.Ad_cols[name], val=self.Ad_cols[name])


class CollocationComp(ExplicitComponent):

    def initialize(self):

        self.options.declare('grid_data', types=GridData)
        self.options.declare('state_options', types=dict)

    def setup(self):
        state_options = self.options['state_options']

        self.add_input('dt_dstau', shape=(2, ))

        self.var_names = var_names = {}
        for state_name in state_options:
            var_names[state_name] = {
                'f_approx': 'f_approx:{0}'.format(state_name),
                'defect': 'defects:{0}'.format(state_name),
            }

        for state_name, options in iteritems(state_options):
            shape = options['shape']

            var_names = self.var_names[state_name]

            self.add_input(
                name=var_names['f_approx'],
                shape=(2, ) + shape)

            self.add_output(
                name=var_names['defect'],
                shape=(2, ) + shape)

            self.add_constraint(name=var_names['defect'],
                                equals=0.0)

        # Setup partials
        for state_name, options in state_options.items():

            r = np.arange(2)

            var_names = self.var_names[state_name]

            self.declare_partials(of=var_names['defect'],
                                  wrt=var_names['f_approx'],
                                  rows=r, cols=r, val=np.ones((2, )))


class ControlInterpComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('control_options', types=dict)
        self.options.declare('grid_data', types=GridData)

        # Save the names of the dynamic controls/parameters
        self._input_names = {}
        self._output_val_names = {}
        self._output_rate_names = {}

    def setup(self):
        num_nodes = self.options['grid_data'].num_nodes
        gd = self.options['grid_data']

        self.add_input('dt_dstau', shape=num_nodes)

        self.val_jacs = {}
        self.rate_jacs = {}

        L_da, D_da = gd.phase_lagrange_matrices('control_disc', 'all')
        self.L = np.dot(L_da, L_da)
        self.D = np.dot(D_da, D_da)

        control_options = self.options['control_options']
        num_control_input_nodes = 5

        for name, options in iteritems(control_options):
            self._input_names[name] = 'controls:{0}'.format(name)
            self._output_val_names[name] = 'control_values:{0}'.format(name)
            self._output_rate_names[name] = 'control_rates:{0}_rate'.format(name)
            shape = options['shape']
            input_shape = (num_control_input_nodes, ) + shape
            output_shape = (num_nodes, ) + shape

            self.add_input(self._input_names[name], val=np.ones(input_shape))

            self.add_output(self._output_val_names[name], shape=output_shape)
            self.add_output(self._output_rate_names[name], shape=output_shape)

            size = np.prod(shape)
            self.val_jacs[name] = np.zeros((num_nodes, size, num_control_input_nodes, size))
            self.rate_jacs[name] = np.zeros((num_nodes, size, num_control_input_nodes, size))
            for i in range(size):
                self.val_jacs[name][:, i, :, i] = self.L
                self.rate_jacs[name][:, i, :, i] = self.D
            self.val_jacs[name] = self.val_jacs[name].reshape((num_nodes * size,
                                                              num_control_input_nodes * size),
                                                              order='C')
            self.rate_jacs[name] = self.rate_jacs[name].reshape((num_nodes * size,
                                                                num_control_input_nodes * size),
                                                                order='C')

            rate_jac_rows, rate_jac_cols = np.where(self.rate_jacs[name] != 0)

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt=self._input_names[name],
                                  rows=rate_jac_rows, cols=rate_jac_cols,
                                  val=rate_jac_rows)


class GridData(object):

    def __init__(self, num_segments, segment_ends=None):

        segment_ends = np.atleast_1d(segment_ends)

        v0 = segment_ends[0]
        v1 = segment_ends[-1]
        segment_ends = -1. + 2 * (segment_ends - v0) / (v1 - v0)

        # List of all GridData attributes

        self.num_segments = num_segments
        self.num_nodes = 0

        self.subset_node_indices = {}
        self.subset_num_nodes = {}

        get_subsets = gauss_lobatto_subsets
        get_points = np.array([-1.        , -0.65465367,  0.        ,  0.65465367,  1.        ])

        # Determine the list of subset_names
        subset_names = get_subsets(1).keys()

        # Initialize num_nodes and subset_num_nodes
        self.num_nodes = 0
        for name in subset_names:
            self.subset_num_nodes[name] = 0

        # Compute the number of nodes in the phase (total and by subset)
        for iseg in range(num_segments):
            segment_nodes = get_points
            segment_subsets = get_subsets(len(segment_nodes),
                                          first_seg=iseg == 0)

            self.num_nodes += len(segment_nodes)

            for name, val in iteritems(segment_subsets):
                self.subset_num_nodes[name] += len(val)

        # Now that we know the sizes, allocate arrays
        self.node_stau = np.empty(self.num_nodes)
        self.node_ptau = np.empty(self.num_nodes)
        for name in subset_names:
            self.subset_node_indices[name] = np.empty(self.subset_num_nodes[name], int)

        # Populate the arrays
        subset_ind0 = {name: 0 for name in subset_names}
        subset_ind1 = {name: 0 for name in subset_names}
        for iseg in range(num_segments):
            segment_nodes = get_points
            segment_subsets = get_subsets(len(segment_nodes),
                                          first_seg=iseg == 0)

            for name in subset_names:
                subset_ind1[name] += len(segment_subsets[name])

            for name in subset_names:
                self.subset_node_indices[name][subset_ind0[name]:subset_ind1[name]] = \
                    segment_subsets[name]

    def phase_lagrange_matrices(self, given_set_name, eval_set_name):
        L_blocks = []
        D_blocks = []

        for iseg in range(self.num_segments):
            indices = self.subset_node_indices[given_set_name][0:5]
            nodes_given = self.node_stau[indices]

            indices = self.subset_node_indices[eval_set_name][0:5]
            nodes_eval = self.node_stau[indices]

            L_block, D_block = lagrange_matrices(nodes_given, nodes_eval)

            L_blocks.append(L_block)
            D_blocks.append(D_block)

        L = block_diag(*L_blocks)
        D = block_diag(*D_blocks)

        return L, D

    def phase_hermite_matrices(self, given_set_name, eval_set_name):

        Ad_seg = np.array([[-1.66654297,  1.49635125,  0.17019172],
                           [-0.17019172, -1.49635125,  1.66654297]])
        Bd_seg = np.array([[-0.13859771, -0.65306122, -0.04507576],
                           [-0.04507576, -0.65306122, -0.13859771]])

        Ad = block_diag(Ad_seg)
        Bd = block_diag(Bd_seg)

        return Ad, Bd


class TimeComp(ExplicitComponent):

    def initialize(self):
        # Required
        self.options.declare('grid_data')

    def setup(self):
        node_ptau = self.options['grid_data'].node_ptau

        self.add_input('t_initial', val=0.)
        self.add_input('t_duration', val=1.)
        self.add_output('dt_dstau', shape=len(node_ptau))


class PhaseBase(Group):

    def __init__(self, **kwargs):
        super(PhaseBase, self).__init__(**kwargs)

        self.state_options = {}
        self.control_options = {}
        self._path_constraints = {}
        self.grid_data = None

        self.ode_options = self.options['ode_class'].ode_options

        # Copy default value for options from the ODEOptions
        for state_name, options in iteritems(self.ode_options._states):
            self.state_options[state_name] = MyOptions()
            self.state_options[state_name]['rate_source'] = options['rate_source']

    def initialize(self):
        # Required metadata
        self.options.declare('num_segments', types=int, desc='Number of segments')
        self.options.declare('ode_class')

        # Optional metadata
        self.options.declare('segment_ends', default=None, allow_none=True)

    def add_control(self, name, lower=None, upper=None, rate_param=None):
        self.control_options[name] = MyOptions()

        self.control_options[name]['rate_param'] = rate_param
        self.control_options[name]['lower'] = lower
        self.control_options[name]['upper'] = upper

    def add_path_constraint(self, name, constraint_name=None, lower=None, upper=None):
        constraint_name = name.split('.')[-1]

        self._path_constraints[name] = {}
        self._path_constraints[name]['constraint_name'] = constraint_name
        self._path_constraints[name]['lower'] = lower
        self._path_constraints[name]['upper'] = upper

    def _add_objective(self, obj_path, shape=(1,)):
        super(PhaseBase, self).add_objective(obj_path, index=-1)

    def setup(self):
        num_segments = self.options['num_segments']
        segment_ends = self.options['segment_ends']

        self.grid_data = GridData(num_segments=num_segments, segment_ends=segment_ends)

        self._time_extents = self._setup_time()

        ctrl_rate_comp = ControlInterpComp(control_options=self.control_options,
                                           grid_data=self.grid_data)
        self._setup_controls()

        self.add_subsystem('control_interp_comp',
                           subsys=ctrl_rate_comp,
                           promotes_inputs=['controls:*'],
                           promotes_outputs=['control_rates:*'])
        self.connect('time.dt_dstau', 'control_interp_comp.dt_dstau')

        self._setup_rhs()
        self._setup_defects()
        self._setup_states()

        self._setup_path_constraints()

    def _setup_time(self):
        grid_data = self.grid_data

        indeps = []
        comps = []

        indeps.append('t_initial')
        self.connect('t_initial', 'time.t_initial')

        indep = IndepVarComp()
        for var in indeps:
            indep.add_output(var, val=1.0)
        self.add_subsystem('time_extents', indep, promotes_outputs=['*'])
        comps += ['time_extents']

        time_comp = TimeComp(grid_data=grid_data)
        self.add_subsystem('time', time_comp)

        self.add_design_var('t_initial')

        return comps

    def _setup_controls(self):
        grid_data = self.grid_data

        indep = self.add_subsystem('indep_controls', subsys=IndepVarComp(),
                                   promotes_outputs=['*'])

        for name, options in iteritems(self.control_options):
            desvar_indices = list(range(5))
            desvar_indices.pop(0)
            desvar_indices.pop()

            self.add_design_var(name='controls:{0}'.format(name),
                                lower=options['lower'],
                                upper=options['upper'],
                                indices=desvar_indices)

            indep.add_output(name='controls:{0}'.format(name), shape=(5, np.prod(options['shape'])))

        return 1


class _ODEStateOptionsDictionary(OptionsDictionary):
    def __init__(self, read_only=False):
        super(_ODEStateOptionsDictionary, self).__init__(read_only)
        self.declare('name')
        self.declare('rate_source')
        self.declare('targets')


class ODEOptions(object):

    def __init__(self, **kwargs):
        self._states = {}
        self._parameters = {}

    def declare_state(self, name, rate_source):
        options = _ODEStateOptionsDictionary()

        options['name'] = name
        options['rate_source'] = rate_source

        self._states[name] = options

    def declare_parameter(self, name, targets):
        options = _ODEStateOptionsDictionary()

        options['name'] = name
        options['targets'] = targets

        self._parameters[name] = options


class OptimizerBasedPhaseBase(PhaseBase):

    def setup(self):
        super(OptimizerBasedPhaseBase, self).setup()

        num_opt_controls = len([name for (name, options) in iteritems(self.control_options)])
        num_controls = len(self.control_options)

        indep_controls = ['indep_controls'] if num_opt_controls > 0 else []
        control_interp_comp = ['control_interp_comp'] if num_controls > 0 else []

        order = self._time_extents + indep_controls + \
            ['indep_states', 'time'] + control_interp_comp

        order = order + ['rhs_disc', 'state_interp', 'rhs_col', 'collocation_constraint']
        order.append('path_constraints')
        self.set_order(order)

    def _setup_rhs(self):
        grid_data = self.grid_data

        self.add_subsystem('state_interp',
                           subsys=StateInterpComp(grid_data=grid_data,
                                                  state_options=self.state_options))

        self.connect('time.dt_dstau', 'state_interp.dt_dstau',
                     src_indices=grid_data.subset_node_indices['col'])

        for name, options in iteritems(self.state_options):
            self.connect('states:{0}'.format(name),
                         'state_interp.state_disc:{0}'.format(name))

    def _setup_states(self):
        grid_data = self.grid_data
        num_state_input_nodes = grid_data.subset_num_nodes['state_input']

        indep = IndepVarComp()
        for name, options in iteritems(self.state_options):
            indep.add_output(name='states:{0}'.format(name),
                             shape=(num_state_input_nodes, np.prod(options['shape'])))
        self.add_subsystem('indep_states', indep, promotes_outputs=['*'])

        for name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])
            desvar_indices = list(range(size * num_state_input_nodes))
            del desvar_indices[:size]

            lb = np.zeros_like(desvar_indices, dtype=float)
            ub = np.zeros_like(desvar_indices, dtype=float)

            self.add_design_var(name='states:{0}'.format(name),
                                lower=lb,
                                upper=ub,
                                indices=[1, 2])

    def _setup_defects(self):
        grid_data = self.grid_data

        self.add_subsystem('collocation_constraint',
                           CollocationComp(grid_data=grid_data,
                                           state_options=self.state_options))

        self.connect('time.dt_dstau', ('collocation_constraint.dt_dstau'),
                     src_indices=grid_data.subset_node_indices['col'])


class Phase(OptimizerBasedPhaseBase):

    def _setup_controls(self):
        num_dynamic = super(Phase, self)._setup_controls()
        grid_data = self.grid_data

        for name, options in iteritems(self.control_options):

            if options['rate_param']:
                state_disc_idxs = grid_data.subset_node_indices['state_disc']
                targets = self.ode_options._parameters[options['rate_param']]['targets']

                self.connect('control_rates:{0}_rate'.format(name),
                             ['rhs_disc.{0}'.format(t) for t in targets],
                             src_indices=state_disc_idxs)

        return num_dynamic

    def _setup_path_constraints(self):
        gd = self.grid_data

        path_comp = GaussLobattoPathConstraintComp(grid_data=gd)
        self.add_subsystem('path_constraints', subsys=path_comp)

        for var, options in iteritems(self._path_constraints):
            con_name = options['constraint_name']

            self.connect(src_name='rhs_disc.{0}'.format(var),
                         tgt_name='path_constraints.disc_values:{0}'.format(con_name))

            kwargs = options.copy()
            kwargs.pop('constraint_name', None)
            path_comp._add_path_constraint(con_name, **kwargs)

    def _setup_rhs(self):
        super(Phase, self)._setup_rhs()

        grid_data = self.grid_data
        ODEClass = self.options['ode_class']

        rhs_disc = ODEClass(num_nodes=grid_data.subset_num_nodes['state_disc'])
        rhs_col = ODEClass(num_nodes=grid_data.subset_num_nodes['col'])

        self.add_subsystem('rhs_disc', rhs_disc)
        self.add_subsystem('rhs_col', rhs_col)

        for name, options in iteritems(self.state_options):

            self.connect('rhs_disc.{0}'.format(options['rate_source']),
                         'state_interp.staterate_disc:{0}'.format(name))

    def _setup_defects(self):
        super(Phase, self)._setup_defects()

        for name, options in iteritems(self.state_options):
            self.connect('state_interp.staterate_col:%s' % name,
                         'collocation_constraint.f_approx:%s' % name)

    def add_objective(self, name, shape=(1, )):
        super(Phase, self)._add_objective('states:{0}'.format(name), shape=shape)


class declare_state(object):

    def __init__(self, name, rate_source):
        self.name = name
        self.rate_source = rate_source

    def __call__(self, system_class):
        system_class.ode_options.declare_state(name=self.name, rate_source=self.rate_source)
        return system_class


class declare_parameter(object):
    def __init__(self, name, targets=None):
        self.name = name
        self.targets = targets

    def __call__(self, system_class):
        if not hasattr(system_class, 'ode_options'):
            setattr(system_class, 'ode_options', ODEOptions())

        system_class.ode_options.declare_parameter(name=self.name, targets=self.targets)
        return system_class


class SteadyFlightPathAngleComp(ExplicitComponent):
    """ Compute the flight path angle (gamma) based on true airspeed and climb rate. """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = 3
        self.add_input('climb_rate', val=np.zeros(nn))
        self.add_output('gam', val=np.zeros(nn))

        # Setup partials
        ar = np.arange(3)

        self.declare_partials(of='gam', wrt='climb_rate', rows=ar, cols=ar, val=ar)


class ThrustEquilibriumComp(ExplicitComponent):
    """
    Compute the rates of TAS and flight path angle required to match a given
    flight condition.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = 3

        # Parameter inputs
        self.add_input(name='gam', shape=(n, ))


class SteadyFlightEquilibriumGroup(Group):

    def initialize(self):
        self.options.declare('num_nodes')

    def setup(self):
        nn = 3

        self.add_subsystem('thrust_eq_comp',
                           subsys=ThrustEquilibriumComp(num_nodes=nn),
                           promotes_inputs=['gam'])


class RangeRateComp(ExplicitComponent):
    """
    Calculates range rate based on true airspeed and flight path angle.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = 3

        self.add_input('gam', val=np.zeros(nn))

        self.add_output('dXdt:range', val=np.ones(nn))

        # Setup partials
        ar = np.arange(3)
        self.declare_partials(of='dXdt:range', wrt='gam', rows=ar, cols=ar, val=ar)


@declare_state('range', rate_source='range_rate_comp.dXdt:range')
@declare_parameter('alt')
@declare_parameter('climb_rate', targets=['gam_comp.climb_rate'])
class AircraftODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = 3

        self.add_subsystem(name='gam_comp', subsys=SteadyFlightPathAngleComp(num_nodes=nn))

        self.connect('gam_comp.gam', ('flight_equilibrium.gam', 'range_rate_comp.gam'))

        self.add_subsystem(name='flight_equilibrium', subsys=SteadyFlightEquilibriumGroup(num_nodes=nn))

        self.add_subsystem(name='range_rate_comp', subsys=RangeRateComp(num_nodes=nn))




class TestColoringBugs(unittest.TestCase):

    def test_color_bug_1(self):

        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['dynamic_simul_derivs'] = True

        seg_ends = np.array([-1., 1.])

        phase = Phase(ode_class=AircraftODE, num_segments=1, segment_ends=seg_ends)
        p.model.add_subsystem('phase0', phase)

        phase.add_control('alt', lower=0.0, upper=50.0, rate_param='climb_rate')

        phase.add_path_constraint('gam_comp.gam', lower=0.01, upper=1.0)
        phase.add_objective('range')

        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.setup(mode='fwd')

        # Bug causd a RuntimeError here.
        p.run_driver()

    def test_color_bug_2(self):

        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['dynamic_simul_derivs'] = True

        seg_ends = np.array([-1., 1.])

        phase = Phase(ode_class=AircraftODE, num_segments=1, segment_ends=seg_ends)
        p.model.add_subsystem('phase0', phase)

        phase.add_control('alt', lower=0.0, upper=50.0, rate_param='climb_rate')

        phase.add_path_constraint('gam_comp.gam', lower=0.01, upper=1.0)
        phase.add_objective('range')

        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.setup(mode='auto')
        p.run_driver()

        # Bug caused an IndexError here.
        totals = p.compute_totals(of=['phase0.states:range'], wrt=['phase0.controls:alt'])


if __name__ == '__main__':
    unittest.main()
