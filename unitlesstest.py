import openmdao.api as om

class Diameter(om.ExplicitComponent):

    def setup(self):

        self.add_output('diameter', units='m')

class DiameterRatioA(om.ExplicitComponent):

    def setup(self):

        self.add_output('diameter_ratio_a', units='none')

class DiameterRatioB(om.ExplicitComponent):

    def setup(self):

        self.add_input('diameter_ratio_b', units='none')


if __name__ == '__main__':

    prob = om.Problem()
    prob.model.add_subsystem('D', Diameter(), promotes=['*'])
    prob.model.add_subsystem('D_ratio_a', DiameterRatioA(), promotes=['*'])
    prob.model.add_subsystem('D_ratio_b', DiameterRatioB(), promotes=['*'])
    prob.model.connect('diameter', 'diameter_ratio_b')

    prob.setup()
