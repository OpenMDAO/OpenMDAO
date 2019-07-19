"""
A contrived example for issuing connections during configure rather than setup.
"""
import itertools
import unittest
from six import iteritems
import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error


class Squarer(om.ExplicitComponent):

    def initialize(self):
        self._vars = {}

    def add_var(self, name, units='m'):
        """
        Add a variable to be squared by the component.
        """
        self._vars[name] = {'units': units}

    def setup(self):
        for var, options in iteritems(self._vars):

            self.add_input(var,
                           units=options['units'])

            self.add_output('{0}_squared'.format(var),
                            units='{0}**2'.format(options['units']))

            self.declare_partials(of='{0}_squared'.format(var),
                                  wrt=var,
                                  method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for var in self._vars:
            outputs['{0}_squared'.format(var)] = inputs[var] ** 2


class Cuber(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('vec_size', types=(int,))
        self._vars = {}

    def add_var(self, name, units='m'):
        """
        Add a variable to be squared by the component.
        """
        self._vars[name] = {'units': units}

    def setup(self):
        for var, options in iteritems(self._vars):
            self.add_input(var, units=options['units'])

            self.add_output('{0}_cubed'.format(var), units='{0}**3'.format(options['units']))

            self.declare_partials(of='{0}_cubed'.format(var),
                                  wrt=var,
                                  method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for var in self._vars:
            outputs['{0}_cubed'.format(var)] = inputs[var] ** 3


class HostConnectInSetup(om.Group):

    def initialize(self):
        self._operations = {}

    def add_operation(self, name, subsys, vars):
        self._operations[name] = {}
        self._operations[name]['subsys'] = subsys
        self._operations[name]['vars'] = vars

    def setup(self):
        ivc = self.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        all_vars = set(itertools.chain(*[self._operations[name]['vars'] for name in self._operations]))

        for var in all_vars:
            ivc.add_output(var, shape=(1,), units='m')

        for name, options in iteritems(self._operations):
            for var in options['vars']:
                options['subsys'].add_var(var)
                self.connect(var, '{0}.{1}'.format(name, var))
            self.add_subsystem(name, options['subsys'])


class HostConnectInConfigure(om.Group):

    def initialize(self):
        self._operations = {}

    def add_operation(self, name, subsys, vars):
        self._operations[name] = {}
        self._operations[name]['subsys'] = subsys
        self._operations[name]['vars'] = vars

    def setup(self):
        ivc = self.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        all_vars = set(itertools.chain(*[self._operations[name]['vars'] for name in self._operations]))

        for var in all_vars:
            ivc.add_output(var, shape=(1,), units='m')

        for name, options in iteritems(self._operations):
            for var in options['vars']:
                options['subsys'].add_var(var)
            self.add_subsystem(name, options['subsys'])

    def configure(self):
        for name, options in iteritems(self._operations):
            for var in options['vars']:
                print('connecting {0} to {1}.{0}'.format(var, name))
                self.connect(var, '{0}.{1}'.format(name, var))


class TestConnectionsInSetup(unittest.TestCase):

    def test_connect_in_setup(self):

        p = om.Problem(model=om.Group())

        h = HostConnectInSetup()

        h.add_operation('squarer', Squarer(), ['a', 'b', 'c'])
        h.add_operation('cuber', Cuber(), ['a', 'b', 'x', 'y'])

        p.model.add_subsystem('h', h)

        p.setup()

        p['h.a'] = 3
        p['h.b'] = 4
        p['h.c'] = 5
        p['h.x'] = 6
        p['h.y'] = 7

        p.run_model()

        assert_rel_error(self, p['h.squarer.a_squared'], p['h.a'] ** 2)
        assert_rel_error(self, p['h.squarer.b_squared'], p['h.b'] ** 2)
        assert_rel_error(self, p['h.squarer.c_squared'], p['h.c'] ** 2)

        assert_rel_error(self, p['h.cuber.a_cubed'], p['h.a'] ** 3)
        assert_rel_error(self, p['h.cuber.b_cubed'], p['h.b'] ** 3)
        assert_rel_error(self, p['h.cuber.x_cubed'], p['h.x'] ** 3)
        assert_rel_error(self, p['h.cuber.y_cubed'], p['h.y'] ** 3)

    def test_connect_in_configure(self):

        p = om.Problem(model=om.Group())

        h = HostConnectInConfigure()

        h.add_operation('squarer', Squarer(), ['a', 'b', 'c'])
        h.add_operation('cuber', Cuber(), ['a', 'b', 'x', 'y'])

        p.model.add_subsystem('h', h)

        p.setup()

        p['h.a'] = 3
        p['h.b'] = 4
        p['h.c'] = 5
        p['h.x'] = 6
        p['h.y'] = 7

        p.run_model()

        assert_rel_error(self, p['h.squarer.a_squared'], p['h.a'] ** 2)
        assert_rel_error(self, p['h.squarer.b_squared'], p['h.b'] ** 2)
        assert_rel_error(self, p['h.squarer.c_squared'], p['h.c'] ** 2)

        assert_rel_error(self, p['h.cuber.a_cubed'], p['h.a'] ** 3)
        assert_rel_error(self, p['h.cuber.b_cubed'], p['h.b'] ** 3)
        assert_rel_error(self, p['h.cuber.x_cubed'], p['h.x'] ** 3)
        assert_rel_error(self, p['h.cuber.y_cubed'], p['h.y'] ** 3)


if __name__ == '__main__':
    unittest.main()
