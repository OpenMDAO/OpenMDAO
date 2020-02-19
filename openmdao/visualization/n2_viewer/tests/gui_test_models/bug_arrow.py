
import numpy as np

from openmdao.api import Group, ExplicitComponent, IndepVarComp


class MapScalars(ExplicitComponent):

    def setup(self):

        self.add_input('PR', val=2.0,
                       desc='User input design pressure ratio')

        self.add_output('s_Nc', shape=1,
                        desc='Scalar for design corrected shaft speed')

    def compute(self, inputs, outputs):
        pass

    def compute_partials(self, inputs, J):
        pass

class DummyComp(ExplicitComponent):

    def setup(self):

        self.add_input('x', 1.0)
        self.add_output('y', 1.0)

class CompressorMap(Group):

    def setup(self):
        self.add_subsystem('d1', DummyComp())
        self.add_subsystem('scalars', MapScalars(),
                            promotes_inputs=['PR',])
        self.add_subsystem('d2', DummyComp())


class Compressor(Group):

    def setup(self):
        map_calcs = CompressorMap()
        self.add_subsystem('map', map_calcs,
                            promotes=['PR'])


class Propulsor(Group):

    def setup(self):
        self.add_subsystem('fan', Compressor())


if __name__ == "__main__":

    from openmdao.api import Problem

    prob = Problem()

    des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])
    des_vars.add_output('FPR', 1.2)

    design = prob.model.add_subsystem('design', Propulsor())

    prob.model.connect('FPR', 'design.fan.PR')

    prob.setup(check=False)

    prob.run_model()
