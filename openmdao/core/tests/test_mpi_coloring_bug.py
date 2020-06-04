import unittest

import numpy as np

import openmdao.api as om
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import set_pyoptsparse_opt


# check that pyoptsparse is installed
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class TrajDesignParameterOptionsDictionary(om.OptionsDictionary):

    def __init__(self, read_only=False):
        super(TrajDesignParameterOptionsDictionary, self).__init__(read_only)

        self.declare(name='name')
        self.declare(name='val', default=np.zeros(1))
        self.declare(name='targets', types=dict, default=None, allow_none=True)
        self.declare(name='shape', default=(1,))


class StateOptionsDictionary(om.OptionsDictionary):

    def __init__(self, read_only=False):
        super(StateOptionsDictionary, self).__init__(read_only)

        self.declare(name='name')
        self.declare(name='val', default=0.0)
        self.declare(name='shape', default=(1,))
        self.declare(name='rate_source')


class CollocationComp(om.ExplicitComponent):

    def initialize(self):

        self.options.declare('state_options', types=dict)

    def setup(self):
        num_col_nodes = 1
        state_options = self.options['state_options']

        self.var_names = var_names = {}
        for state_name in state_options:
            var_names[state_name] = {
                'f_approx': 'f_approx:{0}'.format(state_name),
                'f_computed': 'f_computed:{0}'.format(state_name),
                'defect': 'defects:{0}'.format(state_name),
            }

        for state_name, options in state_options.items():
            shape = options['shape']

            var_names = self.var_names[state_name]

            self.add_input(
                name=var_names['f_approx'],
                shape=(num_col_nodes,) + shape)

            self.add_input(
                name=var_names['f_computed'],
                shape=(num_col_nodes,) + shape)

            self.add_output(
                name=var_names['defect'],
                shape=(num_col_nodes,) + shape)

            self.add_constraint(name=var_names['defect'], equals=0.0)

        # Setup partials
        state_options = self.options['state_options']

        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape)

            r = np.arange(num_col_nodes * size)

            var_names = self.var_names[state_name]

            self.declare_partials(of=var_names['defect'],
                                  wrt=var_names['f_approx'],
                                  rows=r, cols=r)

            self.declare_partials(of=var_names['defect'],
                                  wrt=var_names['f_computed'],
                                  rows=r, cols=r)

    def compute(self, inputs, outputs):
        state_options = self.options['state_options']

        for state_name in state_options:
            var_names = self.var_names[state_name]

            f_approx = inputs[var_names['f_approx']]
            f_computed = inputs[var_names['f_computed']]

            outputs[var_names['defect']] = ((f_approx - f_computed).T).T

    def compute_partials(self, inputs, partials):
        for state_name, options in self.options['state_options'].items():
            size = np.prod(options['shape'])
            var_names = self.var_names[state_name]

            k = np.repeat(1, size)

            partials[var_names['defect'], var_names['f_approx']] = k
            partials[var_names['defect'], var_names['f_computed']] = -k


class StateInterpComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('state_options', types=dict)

    def setup(self):
        num_disc_nodes = 2
        num_col_nodes = 1

        state_options = self.options['state_options']

        self.xd_str = {}
        self.fd_str = {}
        self.xc_str = {}
        self.xdotc_str = {}

        for state_name, options in state_options.items():
            shape = options['shape']

            self.add_input(
                name='state_disc:{0}'.format(state_name),
                shape=(num_disc_nodes,) + shape)

            self.add_input(
                name='staterate_disc:{0}'.format(state_name),
                shape=(num_disc_nodes,) + shape)

            self.add_output(
                name='state_col:{0}'.format(state_name),
                shape=(num_col_nodes,) + shape)

            self.add_output(
                name='staterate_col:{0}'.format(state_name),
                shape=(num_col_nodes,) + shape)

            self.xd_str[state_name] = 'state_disc:{0}'.format(state_name)
            self.fd_str[state_name] = 'staterate_disc:{0}'.format(state_name)
            self.xc_str[state_name] = 'state_col:{0}'.format(state_name)
            self.xdotc_str[state_name] = 'staterate_col:{0}'.format(state_name)

        Ai = np.array([[0.5, 0.5]])
        Bi = np.array([[ 0.25, -0.25]])
        Ad = np.array([[-0.75,  0.75]])
        Bd = np.array([[-0.25, -0.25]])

        self.matrices = {'Ai': Ai, 'Bi': Bi, 'Ad': Ad, 'Bd': Bd}

        # Setup partials

        self.jacs = {'Ai': {}, 'Bi': {}, 'Ad': {}, 'Bd': {}}
        self.Bi_rows = {}
        self.Bi_cols = {}
        self.Ad_rows = {}
        self.Ad_cols = {}
        self.sizes = {}
        self.num_disc_nodes = num_disc_nodes
        for name, options in state_options.items():
            shape = options['shape']

            size = np.prod(shape)

            for key in self.jacs:
                jac = np.zeros((num_col_nodes, size, num_disc_nodes, size))
                for i in range(size):
                    jac[:, i, :, i] = self.matrices[key]
                jac = jac.reshape((num_col_nodes * size, num_disc_nodes * size), order='C')
                self.jacs[key][name] = jac

            self.sizes[name] = size

            rs_dtdstau = np.zeros(num_col_nodes * size, dtype=int)
            r_band = np.arange(0, num_col_nodes, dtype=int) * size

            r0 = 0
            for i in range(size):
                rs_dtdstau[r0:r0 + num_col_nodes] = r_band + i
                r0 += num_col_nodes

            Ai_rows, Ai_cols = np.where(self.jacs['Ai'][name] != 0)
            self.declare_partials(of=self.xc_str[name], wrt=self.xd_str[name],
                                  rows=Ai_rows, cols=Ai_cols,
                                  val=self.jacs['Ai'][name][Ai_rows, Ai_cols])

            self.Bi_rows[name], self.Bi_cols[name] = np.where(self.jacs['Bi'][name] != 0)
            self.declare_partials(of=self.xc_str[name], wrt=self.fd_str[name],
                                  rows=self.Bi_rows[name], cols=self.Bi_cols[name])

            Bd_rows, Bd_cols = np.where(self.jacs['Bd'][name] != 0)
            self.declare_partials(of=self.xdotc_str[name], wrt=self.fd_str[name],
                                  rows=Bd_rows, cols=Bd_cols,
                                  val=self.jacs['Bd'][name][Bd_rows, Bd_cols])

            self.Ad_rows[name], self.Ad_cols[name] = np.where(self.jacs['Ad'][name] != 0)
            self.declare_partials(of=self.xdotc_str[name], wrt=self.xd_str[name],
                                  rows=self.Ad_rows[name], cols=self.Ad_cols[name])

    def compute(self, inputs, outputs):
        state_options = self.options['state_options']

        for name in state_options:
            xc_str = self.xc_str[name]
            xdotc_str = self.xdotc_str[name]
            xd_str = self.xd_str[name]
            fd_str = self.fd_str[name]

            xd = np.atleast_2d(inputs[xd_str])

            a = np.tensordot(self.matrices['Bi'], inputs[fd_str], axes=(1, 0)).T
            outputs[xc_str] = a.T

            outputs[xc_str] += np.tensordot(
                self.matrices['Ai'], xd, axes=(1, 0))

            outputs[xdotc_str] = np.tensordot(self.matrices['Ad'], xd, axes=(1, 0))

            outputs[xdotc_str] += np.tensordot(
                self.matrices['Bd'], inputs[fd_str], axes=(1, 0))

    def compute_partials(self, inputs, partials):
        for name, options in self.options['state_options'].items():
            xdotc_name = self.xdotc_str[name]
            xd_name = self.xd_str[name]

            xc_name = self.xc_str[name]
            fd_name = self.fd_str[name]

            r_nz, c_nz = self.Bi_rows[name], self.Bi_cols[name]
            partials[xc_name, fd_name] = (self.jacs['Bi'][name])[r_nz, c_nz]

            r_nz, c_nz = self.Ad_rows[name], self.Ad_cols[name]
            partials[xdotc_name, xd_name] = (self.jacs['Ad'][name])[r_nz, c_nz]


class StateIndependentsComp(om.ImplicitComponent):

    def initialize(self):

        self.options.declare('state_options', types=dict)

    def setup(self):
        state_options = self.options['state_options']

        num_state_input_nodes = 2

        var_names = {}
        for state_name in state_options:
            var_names[state_name] = {
                'defect': 'defects:{0}'.format(state_name),
            }

        for state_name, options in state_options.items():

            shape = options['shape']
            var_names = var_names[state_name]

            self.add_output(name='states:{0}'.format(state_name),
                            shape=(num_state_input_nodes, ) + shape)

        # Setup partials
        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape)
            state_var_name = 'states:{0}'.format(state_name)

            row_col = np.arange(num_state_input_nodes*np.prod(shape))
            self.declare_partials(of=state_var_name, wrt=state_var_name,
                                  rows=row_col, cols=row_col, val=-1.0)

class Trajectory(om.Group):

    def __init__(self):
        super(Trajectory, self).__init__()

        self.design_parameter_options = {}
        self._phases = {}

    def add_phase(self, name, phase):
        self._phases[name] = phase
        return phase

    def add_design_parameter(self, name, val, targets):
        if name not in self.design_parameter_options:
            self.design_parameter_options[name] = TrajDesignParameterOptionsDictionary()

        self.design_parameter_options[name]['val'] = val
        self.design_parameter_options[name]['targets'] = targets


    def _setup_design_parameters(self):
        if self.design_parameter_options:
            indep = self.add_subsystem('design_params', subsys=om.IndepVarComp(),
                                       promotes_outputs=['*'])

            for name, options in self.design_parameter_options.items():
                indep.add_output(name='design_parameters:{0}'.format(name),
                                 val=options['val'],
                                 shape=(1, np.prod(options['shape'])))

    def setup(self):
        super(Trajectory, self).setup()
        self._setup_design_parameters()

        phases_group = self.add_subsystem('phases', subsys=om.ParallelGroup(), promotes_inputs=['*'],
                                          promotes_outputs=['*'])

        for name, phs in self._phases.items():
            g = phases_group.add_subsystem(name, phs)
            g.linear_solver = om.DirectSolver()


class Phase(om.Group):

    def __init__(self, **kwargs):

        _kwargs = kwargs.copy()

        self.state_options = {}
        self._objectives = {}

        super(Phase, self).__init__(**_kwargs)

    def initialize(self):
        self.options.declare('ode_class', default=None)
        self.options.declare('transcription')

    def add_state(self, name, rate_source):
        if name not in self.state_options:
            self.state_options[name] = StateOptionsDictionary()
            self.state_options[name]['name'] = name

        self.set_state_options(name=name, rate_source=rate_source)

    def set_state_options(self, name, rate_source):
        self.state_options[name]['rate_source'] = rate_source

    def add_objective(self, name, loc='final', index=None, shape=(1,)):
        obj_dict = {'loc': loc,
                    'index': index,
                    'shape': shape}
        self._objectives[name] = obj_dict

    def setup(self):
        transcription = self.options['transcription']

        transcription.setup_states(self)
        transcription.setup_ode(self)
        transcription.setup_defects(self)

        transcription.setup_objective(self)


class GaussLobatto(object):

    def setup_objective(self, phase):
        for name, options in phase._objectives.items():
            index = options['index']

            shape = phase.state_options[name]['shape']
            obj_path = 'states:{0}'.format(name)

            shape = options['shape'] if shape is None else shape

            size = int(np.prod(shape))
            idx = 0 if index is None else index
            obj_index = -size + idx

            super(Phase, phase).add_objective(obj_path, index=obj_index)

    def setup_states(self, phase):
        indep = StateIndependentsComp(state_options=phase.state_options)

        num_connected = 0
        prom_inputs = ['initial_states:*'] if num_connected > 0 else None
        phase.add_subsystem('indep_states', indep, promotes_inputs=prom_inputs,
                            promotes_outputs=['*'])

        for name, options in phase.state_options.items():
            size = np.prod(options['shape'])
            desvar_indices = list(range(size * 2))

            phase.add_design_var(name='states:{0}'.format(name),
                                 indices=desvar_indices)

    def setup_ode(self, phase):
        map_input_indices_to_disc = np.array([0, 1])

        phase.add_subsystem('state_interp',
                            subsys=StateInterpComp(state_options=phase.state_options))

        for name, options in phase.state_options.items():
            size = np.prod(options['shape'])

            src_idxs_mat = np.reshape(np.arange(size * 2, dtype=int),
                                      (2, size), order='C')

            src_idxs = src_idxs_mat[map_input_indices_to_disc, :]

            phase.connect('states:{0}'.format(name),
                          'state_interp.state_disc:{0}'.format(name),
                          src_indices=src_idxs, flat_src_indices=True)

    def setup_defects(self, phase):

        phase.add_subsystem('collocation_constraint',
                            CollocationComp(state_options=phase.state_options))

        for name, options in phase.state_options.items():
            phase.connect('state_interp.staterate_col:{0}'.format(name),
                          'collocation_constraint.f_approx:{0}'.format(name))


class FiniteBurnODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('accel', val=np.zeros(nn))
        self.add_input('c', val=np.zeros(nn))

        self.add_output('deltav_dot', val=np.zeros(nn))

        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='deltav_dot', wrt='accel', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        at = inputs['accel']
        outputs['deltav_dot'] = at


def make_traj():

    t = GaussLobatto()

    traj = Trajectory()

    traj.add_design_parameter('c', val=1.5, targets={'burn1': ['c'], 'burn2': ['c']})

    # First Phase (burn)
    burn1 = Phase(ode_class=FiniteBurnODE, transcription=t)
    burn1 = traj.add_phase('burn1', burn1)
    burn1.add_state('deltav', rate_source='deltav_dot')

    # Third Phase (burn)
    burn2 = Phase(ode_class=FiniteBurnODE, transcription=t)
    traj.add_phase('burn2', burn2)
    burn2.add_state('deltav', rate_source='deltav_dot')

    burn2.add_objective('deltav', loc='final')

    return traj


@unittest.skipUnless(OPTIMIZER, "This test requires pyOptSparseDriver.")
class TestMPIColoringBug(unittest.TestCase):
    N_PROCS = 2

    def test_bug(self):
        class ColoringOnly(om.pyOptSparseDriver):

            def run(self):
                """
                Color the model.
                """
                if coloring_mod._use_total_sparsity:
                    if self._coloring_info['coloring'] is None and self._coloring_info['dynamic']:
                        coloring_mod.dynamic_total_coloring(self, run_model=True,
                                                            fname=self._get_total_coloring_fname())
                        self._setup_tot_jac_sparsity()

        p = om.Problem()

        p.driver = ColoringOnly()
        p.driver.declare_coloring()

        p.model = make_traj()

        p.setup(mode='rev')

        # Set Initial Guesses
        p.set_val('design_parameters:c', value=1.5)

        of = ['phases.burn2.indep_states.states:deltav', 'phases.burn1.collocation_constraint.defects:deltav', 'phases.burn2.collocation_constraint.defects:deltav', ]
        wrt = ['phases.burn1.indep_states.states:deltav', 'phases.burn2.indep_states.states:deltav']

        p.run_model()
        p.run_driver()

        J = p.driver._compute_totals(of=of, wrt=wrt, return_format='dict')
        dd = J['phases.burn1.collocation_constraint.defects:deltav']['phases.burn1.indep_states.states:deltav']

        assert_near_equal(np.array([[-0.75, 0.75]]), dd, 1e-6)
