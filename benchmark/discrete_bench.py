
from openmdao.core.problem import Problem
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.explicitcomponent import ExplicitComponent


class ManyVarComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('ndiscrete_ins', types=int)
        self.options.declare('ndiscrete_outs', types=int)
        self.options.declare('nins', types=int)
        self.options.declare('nouts', types=int)

    def setup(self):
        ndiscrete_ins = self.options['ndiscrete_ins']
        ndiscrete_outs = self.options['ndiscrete_outs']
        nins = self.options['nins']
        nouts = self.options['nouts']

        for i in range(ndiscrete_ins):
            self.add_discrete_input(f'dinp{i}', i)
        for i in range(nins):
            self.add_input(f'inp{i}')

        for i in range(ndiscrete_outs):
            self.add_discrete_output(f'dout{i}', i)
        for i in range(nouts):
            self.add_output(f'out{i}')

        nvars = min(nins, nouts)
        for i in range(nvars):
            self.declare_partials(of=f'out{i}', wrt=f'inp{i}', method='fd')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        for i in range(min(self.options['ndiscrete_outs'], self.options['ndiscrete_ins'])):
            discrete_outputs[f'dout{i}'] = discrete_inputs[f'dinp{i}'] + 1
        for i in range(min(self.options['nouts'], self.options['nins'])):
            outputs[f'out{i}'] = inputs[f'inp{i}'] + 1.0


def build_model(ncomps=1, ndiscrete_ins=10, ndiscrete_outs=10, nins=10, nouts=10):
    p = Problem()
    model = p.model
    ivc = model.add_subsystem('ivc', IndepVarComp())
    for i in range(ndiscrete_ins):
        ivc.add_discrete_output(f'dout{i}', i)
    for i in range(nins):
        ivc.add_output(f'out{i}')


    ndiscretes = min(ndiscrete_ins, ndiscrete_outs)
    nconts = min(nins, nouts)

    for icomp in range(ncomps):
        model.add_subsystem(f'comp_{icomp}', ManyVarComp(ndiscrete_ins=ndiscrete_ins,
                                                         ndiscrete_outs=ndiscrete_outs,
                                                         nins=nins,
                                                         nouts=nouts))
        if icomp > 0:
            for i in range(ndiscretes):
                model.connect(f'comp_{icomp-1}.dout{i}', f'comp_{icomp}.dinp{i}')
            for i in range(nconts):
                model.connect(f'comp_{icomp-1}.out{i}', f'comp_{icomp}.inp{i}')


    for i in range(ndiscretes):
        model.connect(f'ivc.dout{i}', f'comp_0.dinp{i}')
    for i in range(nconts):
        model.connect(f'ivc.out{i}', f'comp_0.inp{i}')

    model.add_objective(f'comp_{ncomps-1}.out0')
    for i in range(1, nconts):
        model.add_constraint(f'comp_{ncomps-1}.out{i}', lower=0.0)
    for i in range(nconts):
        model.add_design_var(f'ivc.out{i}', lower=0.0, upper=10.0)

    return p


if __name__ == '__main__':
    p = build_model(ncomps=3, ndiscrete_ins=20, ndiscrete_outs=33, nins=70, nouts=511)

    import time
    start = time.time()

    p.setup()

    end = time.time()
    print("setup time: ", end - start)
    start = end

    p.run_model()

    end = time.time()
    print("run_model time: ", end - start)
    start = end

    try:
        J = p.compute_totals()
    except Exception as e:
        print(e)

    print("compute_totals time: ", time.time() - start)