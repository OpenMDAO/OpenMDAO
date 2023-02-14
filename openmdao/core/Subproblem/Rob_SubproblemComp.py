import openmdao.api as om


class Paraboloid(om.ExplicitComponent):
    """
    The Component to be run in our subproblem.
    """

    def setup(self):
        self.add_input('x')
        self.add_input('y')

        self.add_output('f_xy')

        self.declare_partials(of='f_xy', wrt='x')
        self.declare_partials(of='f_xy', wrt='y')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['f_xy'] = (x - 3)**2 + x*y + (y - 7)**2 - 3

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        partials['f_xy', 'x'] = 2 * (x - 3) + y
        partials['f_xy', 'y'] = 2 * (y - 7) + x


class SubproblemComp(om.ExplicitComponent):
    """
    Wrap a subproblem as an Explicit calculation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._subprob = None
        self._prev_complex_step = False

    def initialize(self):
        self.options.declare('system_class')
        self.options.declare('system_init_kwargs', types=dict, allow_none=True, default=None)
        self.options.declare('subprob_reports', types=bool, default=False,
                             desc='Flag that determines whether reports are'
                                  'generated for the subproblem.')

    def setup(self):
        self._setup_subprob()

    def _setup_subprob(self):
        p = self._subprob = om.Problem(reports=self.options['subprob_reports'])
        sys_init_kwargs = self.options['system_init_kwargs'] or {}
        p.model.add_subsystem('subsys',
                              self.options['system_class'](**sys_init_kwargs),
                              promotes=['*'])

        # Now setup our subproblem
        p.setup(force_alloc_complex=False)
        p.final_setup()

        # Lets interrogate the subsystem to figure out what its inputs and outputs are
        inputs = p.model.list_inputs(out_stream=None, prom_name=True,
                                     units=True, shape=True, desc=True)

        outputs = p.model.list_outputs(out_stream=None, prom_name=True,
                                       units=True, shape=True, desc=True)

        # Store the inputs and outputs with promoted path instead of absolute
        inputs = {meta['prom_name']: meta for _, meta in inputs}
        outputs = {meta['prom_name']: meta for _, meta in outputs}

        self._input_names = []
        self._output_names = []

        # Remove the `prom_name` from the metadata and then store it for each
        # input and output.
        for var, meta in inputs.items():
            meta.pop('prom_name')
            self.add_input(var, **meta)
            self._input_names.append(var)

        for var, meta in outputs.items():
            meta.pop('prom_name')
            self.add_output(var, **meta) # essentially just means we want a dictionary as kwargs
            self._output_names.append(var)

            for ip in self._input_names:
                self.declare_partials(of=var, wrt=ip)

    def compute(self, inputs, outputs):
        p = self._subprob

        # Switch subproblem to use complex IO if in complex step mode.
        if self.under_complex_step != self._prev_complex_step:
            if self.under_complex_step:
                p.setup(force_alloc_complex=True)
                p.final_setup()
                p.set_complex_step_mode(True)
            else:
                p.setup()
                p.final_setup()
                p.set_complex_step_mode(False)
            self._prev_complex_step = self.under_complex_step

        for inp in self._input_names:
            p.set_val(inp, inputs[inp])

        p.run_model()

        for op in self._output_names:
            outputs[op] = p.get_val(op)

    def compute_partials(self, inputs, partials):
        p = self._subprob
        for inp in self._input_names:
            p.set_val(inp, inputs[inp])

        tots = p.compute_totals(of=self._output_names, wrt=self._input_names,
                                use_abs_names=False)

        for of in self._output_names:
            for wrt in self._input_names:
                partials[of, wrt] = tots[of, wrt]


if __name__ == '__main__':
    p = om.Problem()
    spc = SubproblemComp()
    spc.options['system_class'] = Paraboloid
    p.model.add_subsystem('subprob_comp', spc)
    p.setup(force_alloc_complex=True)
    p.set_val('subprob_comp.x', 5)
    p.set_val('subprob_comp.y', 7)
    p.run_model()
    cpd = p.check_partials(method='cs')