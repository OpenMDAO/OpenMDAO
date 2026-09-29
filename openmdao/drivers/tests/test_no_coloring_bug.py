import unittest

import numpy as np

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import openmdao.api as om


class ParameterComp(om.ExplicitComponent):
    def setup(self):
        self.add_input(name='t_duration', val=1, shape=(1,))
        self.add_output(name='t_duration_val', val=1, shape=(1,))

        self.declare_partials('*', '*', method='cs')

    def add_parameter(self, name, val=1.0, shape=None, output_name=None):

        _out_name = output_name if output_name is not None else f'parameter_vals:{name}'
        in_val = np.asarray(val)

        self.add_input(name=f'parameters:{name}', val=in_val, shape=(1, 1))
        self.add_output(name=_out_name, shape=(1, 1))

    def compute(self, inputs, outputs):
        outputs.set_val(inputs.asarray())


class TimeComp(om.ExplicitComponent):

    def setup(self):
        nn = 3
        self.add_input('t_duration', val=1)
        self.add_output('dt_dstau', val=np.ones(nn))
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        t_duration = inputs['t_duration']
        outputs['dt_dstau'][:] = 0.5 * t_duration


class Trajectory(om.Group):

    def __init__(self, **kwargs):
        super(Trajectory, self).__init__(**kwargs)

        self._phases = {}
        self.options['auto_order'] = True
        self.phases = om.Group()

    def initialize(self):
        self.options.declare('parameter_options', types=dict, default={})

    @property
    def parameter_options(self):
        return self.options['parameter_options']

    def add_phase(self, name, phase, **kwargs):
        self._phases[name] = self.add_subsystem(name, phase, **kwargs)
        return phase

    def add_parameter(self, name, val=None, desc=None, opt=False,
                      targets=None, shape=None):
        self.parameter_options[name] = {}
        self.parameter_options[name]['name'] = name

        if val is not None:
            self.parameter_options[name]['val'] = val

        if targets is not None:
            self.parameter_options[name]['targets'] = targets

        if shape is not None:
            self.parameter_options[name]['shape'] = shape

    def setup(self):
        param_comp = ParameterComp()
        self.add_subsystem('param_comp', subsys=param_comp, promotes_inputs=['*'], promotes_outputs=['*'])

        for name, options in self.parameter_options.items():
            for phase_name, phs in self._phases.items():
                kwargs = {}

                phs.add_parameter(name, **kwargs)

        parameter_options = self.parameter_options
        promoted_inputs = []

        for name, options in parameter_options.items():
            promoted_inputs.append(f'parameters:{name}')

            options['shape'] = (1, )
            param_comp = self._get_subsystem('param_comp')
            param_comp.add_parameter(name, shape=options['shape'])
            self.add_design_var(name=f'parameters:{name}')

            tgts = ['cruise.parameters:aircraft:wing:root_chord']
            self.connect(f'parameter_vals:{name}', tgts)

        self.promotes('phases', inputs=['*'], outputs=['*'])


class ControlInterpComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('control_options', types=dict)

        self._input_names = {}
        self._output_val_names = {}

    def _configure_controls(self):
        control_options = self.options['control_options']
        num_output_nodes = 3

        for name, options in control_options.items():
            shape = options['shape']
            output_shape = (num_output_nodes,) + shape

            self._input_names[name] = f'controls:{name}'

            self._output_val_names[name] = f'control_values:{name}'
            self.add_output(self._output_val_names[name], shape=output_shape)

            self.add_input(self._input_names[name])

        self.declare_partials('*', '*', method='cs')

    def _configure_desvars(self):
        control_options = self.options['control_options']
        for name, options in control_options.items():
            dvname = f'controls:{name}'
            self.add_design_var(name=dvname)

    def configure_io(self):
        output_num_nodes = 3

        self.add_input('dt_dstau', shape=output_num_nodes)
        self._configure_controls()
        self._configure_desvars()

    def compute(self, inputs, outputs):
        for name, options in self.options['control_options'].items():
            outputs[self._output_val_names[name]][0] = outputs[self._output_val_names[name]][0]
            outputs[self._output_val_names[name]][1] = outputs[self._output_val_names[name]][1]
            outputs[self._output_val_names[name]][2] = outputs[self._output_val_names[name]][1] * 2


class FlightConditions(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = 3
        arange = np.arange(nn)

        self.add_input(
            'mach',
            val=np.zeros(nn),
        )
        self.add_output(
            'velocity',
            val=np.zeros(nn),
        )

        self.declare_partials(
            'velocity',
            ['mach'],
            rows=arange,
            cols=arange,
        )

    def compute(self, inputs, outputs):
        mach = inputs['mach']
        outputs['velocity'] = 0.5 * mach

    def compute_partials(self, inputs, J):
        J['velocity', 'mach'] = 0.5


class TimeseriesOutputComp(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._vars = {}
        self._sources = {}
        self.input_num_nodes = 0
        self.output_num_nodes = 0

    def _add_output_configure(self, name, shape, desc='', src=None):
        output_num_nodes = self.output_num_nodes
        output_name = name
        self.add_output(output_name, shape=(output_num_nodes,) + shape, desc=desc)

    def _configure_io(self, timeseries_options):

        self.input_num_nodes = 3
        self.output_num_nodes = 3

        self.add_input('dt_dstau', shape=(self.input_num_nodes,))

        ts_inputs = []
        for _, ts_output in timeseries_options['outputs'].items():
            name = ts_output['name']
            shape = ts_output['shape']
            src = ts_output['src']

            self._add_output_configure(name, shape=shape, src=src)

        return ts_inputs


class Phase(om.Group):

    def __init__(self, **kwargs):
        _kwargs = kwargs.copy()

        self.timeseries_options = {}

        self._timeseries = {'timeseries': {'subset': 'all',
                                           'outputs': {}}}

        super(Phase, self).__init__(**_kwargs)

    def initialize(self):
        self.options.declare('ode_class', default=None,
                             desc='System defining the ODE',
                             recordable=False)
        self.options.declare('state_options', types=dict, default={},
                             desc='Options for each state in this phase.')
        self.options.declare('parameter_options', types=dict, default={},
                             desc='Options for each parameter in this phase.')
        self.options.declare('control_options', types=dict, default={},
                             desc='Options for each control in this phase.')

    @property
    def state_options(self):
        return self.options['state_options']

    @property
    def parameter_options(self):
        return self.options['parameter_options']

    @property
    def control_options(self):
        return self.options['control_options']

    def add_state(self, name, shape=None,
                  targets=None,
                  val=None, source=None, ):
        if name not in self.state_options:
            self.state_options[name] = {}
            self.state_options[name]['name'] = name

        if shape is not None:
            self.state_options[name]['shape'] = shape

        if val is not None:
            self.state_options[name]['val'] = val


    def add_control(self, name, desc=None, opt=None,
                    targets=None, val=None, shape=None):
        if name not in self.control_options:
            self.control_options[name] = {}
            self.control_options[name]['name'] = name

        if val is not None:
            self.control_options[name]['val'] = val

        if shape is not None:
            self.control_options[name]['shape'] = shape

    def add_parameter(self, name, val=None, opt=False,
                      shape=None):
        if name not in self.parameter_options:
            self.parameter_options[name] = {}
            self.parameter_options[name]['name'] = name

        self.parameter_options[name]['opt'] = opt

        if val is not None:
            self.parameter_options[name]['val'] = val

    def add_timeseries_output(self, name, output_name=None, shape=None,
                              timeseries='timeseries', **kwargs):
        ts_output = {}
        ts_output['name'] = name
        ts_output['output_name'] = output_name
        ts_output['shape'] = shape

        self._timeseries[timeseries]['outputs'][output_name] = ts_output

        return output_name

    def setup(self):
        t_name = 'time'

        self.add_subsystem('param_comp', subsys=ParameterComp(),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        for ts_name, ts_options in self._timeseries.items():
            self.add_timeseries_output(t_name, timeseries=ts_name)

        time_comp = TimeComp()
        self.add_subsystem('time', time_comp, promotes_inputs=['*'], promotes_outputs=['*'])

        control_comp = ControlInterpComp(control_options=self.control_options)
        self.add_subsystem('control_comp', subsys=control_comp)

        self.add_subsystem(
            name='flight_conditions',
            subsys=FlightConditions(),
            promotes=['*'],
        )

        nn = 3
        wing_mesh = generate_mesh()

        wing_surface = {
            'name': 'wing',
            'mesh': wing_mesh,
        }

        surfaces = [wing_surface]

        for surface in surfaces:
            mesh = surface["mesh"]
            ny = mesh.shape[1]
            mesh_shape = mesh.shape

            # 2. Scale X
            val = np.ones(ny)
            if "chord_cp" in surface:
                promotes = ["chord"]
            else:
                promotes = []

            self.add_subsystem(
                "scale_x",
                ScaleX(val=val, mesh_shape=mesh_shape),
                promotes_inputs=promotes,
            )

            # 9. Rotate
            val = np.zeros(ny)
            if "twist_cp" in surface:
                promotes = ["twist"]
            else:
                val = np.zeros(ny)
                promotes = []

            self.add_subsystem(
                "rotate",
                Rotate(val=val, mesh_shape=mesh_shape),
                promotes_inputs=promotes,
                promotes_outputs=["mesh"],
            )

            names = ["scale_x", "rotate"]

            for j in np.arange(len(names) - 1):
                self.connect(names[j] + ".mesh", names[j + 1] + ".in_mesh")

        for surface in surfaces:
            name = surface["name"]

            self.add_subsystem(name, VLMGeometry(surface=surface))
            self.connect(name + ".S_ref", "sum_areas." + name + "_S_ref")

        self.add_subsystem(
            "sum_areas", SumAreas(surfaces=surfaces), promotes_outputs=["S_ref_total"]
        )

        self.add_subsystem(
            "CL_CD",
            TotalLiftDrag(),
            promotes_inputs=["S_ref_total"],
            promotes_outputs=["CD"],
        )
        self.add_constraint('CD', equals=0.0)

        self.promotes('CL_CD', inputs=[('v', 'velocity')], src_indices=[0])

        for surface in surfaces:
            name = surface['name']
            self.connect(f'mesh', f'{name}.def_mesh')

        for name, options in self._timeseries.items():
            timeseries_comp = TimeseriesOutputComp()
            self.add_subsystem(name, subsys=timeseries_comp)

        for name, options in self.control_options.items():
            options['targets'] = ['mach']
            options['shape'] = (1,)

        for name, options in self.parameter_options.items():
            options['targets'] = ['aircraft:wing:root_chord']
            options['shape'] = (1, )

        for state_name, options in self.state_options.items():
            options['shape'] = (1,)

        self.control_comp.configure_io()
        self.promotes('control_comp', any=['*control_values:*'])

        for name, options in self.control_options.items():
            self.connect(f'control_values:{name}', [f'{t}' for t in options['targets']])

        param_comp = self._get_subsystem('param_comp')

        for name, options in self.parameter_options.items():
            param_comp.add_parameter(name, shape=options['shape'])
            self.connect(f'parameter_vals:{name}', 'aircraft:wing:root_chord')

        idx = np.array([0, 1])

        for ts_name, ts_opts in self._timeseries.items():

            for output_name, output_options in ts_opts['outputs'].items():
                output_options['src'] = 't'
                output_options['shape'] = (1, )

        for timeseries_name, timeseries_options in self._timeseries.items():
            timeseries_comp = self._get_subsystem(timeseries_name)
            timeseries_comp._configure_io(timeseries_options)

        self.options['auto_order'] = True

    def configure(self):
        self.promotes(
            'scale_x',
            inputs=[('chord', 'aircraft:wing:root_chord')],
        )


class ScaleX(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("val", desc="Initial value for chord lengths")
        self.options.declare("mesh_shape", desc="Tuple containing mesh shape (nx, ny).")

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]
        self.add_input("chord", val=val)
        self.add_input("in_mesh", shape=mesh_shape)

        self.add_output("mesh", shape=mesh_shape)
        self.declare_partials("mesh", "chord", method='cs')

    def compute(self, inputs, outputs):
        mesh = inputs["in_mesh"]
        chord_dist = inputs["chord"]
        outputs["mesh"] = np.sum(chord_dist) * mesh


class Rotate(om.ExplicitComponent):

    def initialize(self):
        self.options.declare("val", desc="Initial value for chord lengths")
        self.options.declare("mesh_shape", desc="Tuple containing mesh shape (nx, ny).")

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]
        self.add_input("twist", val=val)
        self.add_input("in_mesh", shape=mesh_shape)

        self.add_output("mesh", shape=mesh_shape)
        self.declare_partials("mesh", "twist", method='cs')

    def compute(self, inputs, outputs):
        mesh = inputs["in_mesh"]
        chord_dist = inputs["twist"]
        outputs["mesh"] = np.sum(chord_dist) * mesh


class SumAreas(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("surfaces", types=list)

    def setup(self):
        for surface in self.options["surfaces"]:
            name = surface["name"]
            self.add_input(name + "_S_ref", val=1.0)

        self.add_output("S_ref_total", val=0.0)
        self.declare_partials("*", "*", method='cs')

    def compute(self, inputs, outputs):
        outputs["S_ref_total"] = 0.0
        for surface in self.options["surfaces"]:
            name = surface["name"]
            S_ref = inputs[name + "_S_ref"]
            outputs["S_ref_total"] += S_ref


class TotalLiftDrag(om.ExplicitComponent):

    def setup(self):
        self.add_input("S_ref_total", val=1.0)
        self.add_input("v", val=1.0)
        self.add_output("CD", val=1.0)

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        z = self._problem_meta['relevance']
        outputs["CD"] = 1.0 / inputs["S_ref_total"]


class VLMGeometry(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        nx = 1
        ny = 1

        self.add_input("def_mesh", val=np.ones((nx, ny, 3)))
        self.add_output("S_ref", val=1.0)
        self.declare_partials("S_ref", "def_mesh", method='cs')

    def compute(self, inputs, outputs):
        mesh = inputs["def_mesh"]
        outputs["S_ref"] = np.sum(mesh)


def generate_mesh():
    num_x = 1
    num_y = 1
    mesh = np.ones((num_x, num_y, 3))
    return mesh


class AviaryProblem(om.Problem):

    def build_model(self):
        traj = self.model = Trajectory()

        phase = Phase()
        phase.add_state('mass', 'mass')
        phase.add_control('mach')

        traj.add_phase('cruise', phase)
        traj.add_parameter('aircraft:wing:root_chord')

    def add_driver(self, optimizer='IPOPT'):
        driver = self.driver = om.pyOptSparseDriver()
        driver.options['optimizer'] = optimizer

    def add_objective(self, objective_type=None, ref=None):
        self.model.add_objective(f'cruise.timeseries.time', index=-1)


@use_tempdirs
class TestDriver(unittest.TestCase):

    @require_pyoptsparse('IPOPT')
    def test_basic_get(self):
        # This tests a bug where a model with pre/opt/post and no coloring was filtering out
        # relevant components. Specifically, SumAreas needs to run before TotalLiftDrag to
        # prevent a division by zero.

        prob = AviaryProblem()
        prob.options['group_by_pre_opt_post'] = True

        prob.build_model()
        prob.add_driver('IPOPT')
        prob.add_objective(objective_type='time')
        prob.setup()
        prob.set_val('parameters:aircraft:wing:root_chord', 1.)
        prob.set_val('cruise.rotate.twist', 1.)

        # Should run without an error.
        prob.run_driver()


if __name__ == "__main__":
    unittest.main()
