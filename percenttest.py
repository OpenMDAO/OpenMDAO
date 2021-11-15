import openmdao.api as om


class Dectopercent(om.ExplicitComponent):

    def setup(self):

        self.add_input('val', units='dec')


if __name__ == '__main__':

    prob = om.Problem()
    prob.model.add_subsystem('val_comp', Dectopercent(), promotes=['*'])

    prob.setup()
    prob.set_val('val', 10, 'percent')

    print('Dec: ', prob.get_val('val', 'dec'))
    print('%: ', prob.get_val('val', 'percent'))