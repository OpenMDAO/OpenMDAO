"""
A contrived example for issuing connections during configure rather than setup.
"""
import itertools
import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class Squarer(om.ExplicitComponent):

    def initialize(self):
        self._vars = {}

    def add_var(self, name, units='m'):
        """
        Add a variable to be squared by the component.
        """
        self._vars[name] = {'units': units}

    def setup(self):
        for var, options in self._vars.items():

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

    def config_var(self, name, units='m'):
        """
        This is a form of add_var that doesn't queue the IO until setup time.

        This method is intended to be called during configure by parent subsystems.
        """
        self._vars[name] = {'units': units}
        self.add_input(name, units=units)
        self.add_output('{0}_cubed'.format(name), units='{0}**3'.format(units))
        self.declare_partials(of='{0}_cubed'.format(name),
                              wrt=name,
                              method='cs')

    def setup(self):
        for var, options in self._vars.items():
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

        for name, options in self._operations.items():
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

        for name, options in self._operations.items():
            for var in options['vars']:
                options['subsys'].add_var(var)
            self.add_subsystem(name, options['subsys'])

    def configure(self):
        for name, options in self._operations.items():
            for var in options['vars']:
                print('connecting {0} to {1}.{0}'.format(var, name))
                self.connect(var, '{0}.{1}'.format(name, var))


class GroupQueuesIOInConfigure(om.Group):
    def initialize(self):
        self._vars_to_cube = {}

    def add_var_to_cube(self, name, units=None):
        self._vars_to_cube[name] = {}
        self._vars_to_cube[name]['units'] = units

    def setup(self):
        self.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        self.add_subsystem('cuber', Cuber(), promotes_inputs=['*'], promotes_outputs=['*'])

    def configure(self):
        for name, options in self._vars_to_cube.items():
            self.ivc.add_output(name, units=options['units'])
            self.cuber.add_var(name, units=options['units'])


class GroupAddsIOInConfigure(om.Group):
    def initialize(self):
        self._vars_to_cube = {}

    def add_var_to_cube(self, name, units=None):
        self._vars_to_cube[name] = {}
        self._vars_to_cube[name]['units'] = units

    def setup(self):
        self.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        self.add_subsystem('cuber', Cuber(), promotes_inputs=['*'], promotes_outputs=['*'])

    def configure(self):
        for name, options in self._vars_to_cube.items():
            self.ivc.add_output(name, units=options['units'])
            self.cuber.config_var(name, units=options['units'])


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

        assert_near_equal(p['h.squarer.a_squared'], p['h.a'] ** 2)
        assert_near_equal(p['h.squarer.b_squared'], p['h.b'] ** 2)
        assert_near_equal(p['h.squarer.c_squared'], p['h.c'] ** 2)

        assert_near_equal(p['h.cuber.a_cubed'], p['h.a'] ** 3)
        assert_near_equal(p['h.cuber.b_cubed'], p['h.b'] ** 3)
        assert_near_equal(p['h.cuber.x_cubed'], p['h.x'] ** 3)
        assert_near_equal(p['h.cuber.y_cubed'], p['h.y'] ** 3)

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

        assert_near_equal(p['h.squarer.a_squared'], p['h.a'] ** 2)
        assert_near_equal(p['h.squarer.b_squared'], p['h.b'] ** 2)
        assert_near_equal(p['h.squarer.c_squared'], p['h.c'] ** 2)

        assert_near_equal(p['h.cuber.a_cubed'], p['h.a'] ** 3)
        assert_near_equal(p['h.cuber.b_cubed'], p['h.b'] ** 3)
        assert_near_equal(p['h.cuber.x_cubed'], p['h.x'] ** 3)
        assert_near_equal(p['h.cuber.y_cubed'], p['h.y'] ** 3)


class TestAddSubcomponentIOInConfigure(unittest.TestCase):

    def test_add_subcomponent_io_in_configure(self):
        """
        This test directly adds IO to a component in configure, and should behave as expected.
        """
        p = om.Problem(model=om.Group())

        g = GroupAddsIOInConfigure()

        g.add_var_to_cube('foo', units='m')

        p.model.add_subsystem('g', subsys=g)

        p.setup()

        p.set_val('g.foo', 5)

        p.run_model()

        assert_near_equal(p.get_val('g.foo_cubed'), p.get_val('g.foo')**3)


if __name__ == '__main__':
    unittest.main()
