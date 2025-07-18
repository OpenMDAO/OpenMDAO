import unittest
from openmdao.utils.name_maps import NameResolver, _get_flags


class TestNameResolver(unittest.TestCase):
    def setUp(self):
        self.resolver = NameResolver('test_system')

    def test_init(self):
        resolver = NameResolver('test_system')
        self.assertEqual(resolver._pathname, 'test_system')
        self.assertEqual(resolver._prefix, 'test_system.')
        self.assertEqual(resolver._pathlen, 12)  # len('test_system') + 1
        self.assertEqual(resolver._abs2prom, {'input': {}, 'output': {}})
        self.assertEqual(resolver.msginfo, 'test_system')
        self.assertEqual(resolver._prom2abs, {'input': {}, 'output': {}})

        # Test with msginfo
        resolver = NameResolver('test_system', msginfo='custom_msg')
        self.assertEqual(resolver.msginfo, 'custom_msg')

    def test_add_mapping(self):
        _, locflags = _get_flags(local=True, continuous=True, distributed=False)
        # Test adding input mapping
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.assertEqual(self.resolver._abs2prom['input']['test_system.x'], ('x', locflags))

        # Test adding output mapping
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)
        self.assertEqual(self.resolver._abs2prom['output']['test_system.y'], ('y', locflags))

        # Test adding distributed mapping
        self.resolver.add_mapping('test_system.z', 'z', 'output', local=True, distributed=True)
        self.assertEqual(self.resolver._abs2prom['output']['test_system.z'], ('z', _get_flags(local=True, continuous=True, distributed=True)[1]))

        # Test adding non-continuous mapping
        self.resolver.add_mapping('test_system.w', 'w', 'input', local=True, continuous=False)
        self.assertEqual(self.resolver._abs2prom['input']['test_system.w'], ('w', _get_flags(local=True, continuous=False)[1]))

    def test_populate_prom2abs(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)
        self.resolver.add_mapping('test_system.z', 'z', 'output', local=True)

        # Call _populate_prom2abs
        self.resolver._populate_prom2abs()

        # Check the populated dictionaries
        self.assertEqual(self.resolver._prom2abs['input'], {'x': ['test_system.x']})
        self.assertEqual(self.resolver._prom2abs['output'], {'y': ['test_system.y'], 'z': ['test_system.z']})

    def test_abs2prom_iter_multiple(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x1', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.x2', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test iterating over multiple absolute names
        abs2prom = list(self.resolver.abs2prom_iter('input'))
        self.assertEqual(abs2prom, [('test_system.x1', 'x'), ('test_system.x2', 'x')])

    def test_rel2abs_iter_multiple(self):
        # Test converting multiple relative names
        relnames = ['x', 'y', 'z']
        absnames = list(self.resolver.rel2abs_iter(relnames))
        self.assertEqual(absnames, ['test_system.x', 'test_system.y', 'test_system.z'])

    def test_check_dups(self):
        resolver = NameResolver('test_system')

        # Add mappings with duplicate promoted names
        resolver.add_mapping('test_system.x1', 'x', 'output', local=True)
        resolver.add_mapping('test_system.x2', 'x', 'output', local=True)
        resolver._populate_prom2abs()

        # This should raise ValueError due to duplicate output names
        with self.assertRaises(ValueError):
            resolver._check_dup_prom_outs()

    def test_source_with_connections(self):
        # Setup some mappings and connections
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test error for no connections
        with self.assertRaises(RuntimeError):
            self.resolver.source('test_system.x')

        self.resolver._conns = {'test_system.x': 'test_system.y'}

        # Test getting source for input
        self.assertEqual(self.resolver.source('test_system.x'), 'test_system.y')

        # Test getting source for output
        self.assertEqual(self.resolver.source('test_system.y'), 'test_system.y')

        # Test getting source for promoted name
        self.assertEqual(self.resolver.source('x'), 'test_system.y')

        # Test error for non-existent connection
        with self.assertRaises(Exception):
            self.resolver.source('test_system.z')

    def test_prom2abs(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.v', 'v', 'input', local=True)
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.xx', 'x', 'input', local=False)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)
        self.resolver.add_mapping('test_system.z', 'z', 'output', local=True)

        # Test basic conversion
        self.assertEqual(self.resolver.prom2abs('v', 'input'), 'test_system.v')
        self.assertEqual(self.resolver.prom2abs('y', 'output'), 'test_system.y')

        # Test with ambiguity
        with self.assertRaises(Exception) as ctx:
            self.assertEqual(self.resolver.prom2abs('x', 'input'), 'test_system.xx')

        self.assertEqual(ctx.exception.args[0],
                         "test_system: The promoted name x is invalid because it refers to multiple inputs: [test_system.x ,test_system.xx]. Access the value from the connected output variable instead.")

        # Test error for non-existent name
        with self.assertRaises(KeyError):
            self.resolver.prom2abs('nonexistent', 'input')

    def test_abs2prom(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test basic conversion
        self.assertEqual(self.resolver.abs2prom('test_system.x', 'input'), 'x')
        self.assertEqual(self.resolver.abs2prom('test_system.y', 'output'), 'y')

        # Test with local=True
        self.assertEqual(self.resolver.abs2prom('test_system.x', 'input'), 'x')

        # Test error for non-existent name
        with self.assertRaises(KeyError):
            self.resolver.abs2prom('nonexistent', 'input')

    def test_abs2rel(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        self.assertEqual(self.resolver.abs2rel('test_system.x'), 'x')
        self.assertEqual(self.resolver.abs2rel('test_system.y'), 'y')

    def test_rel2abs(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        self.assertEqual(self.resolver.rel2abs('x'), 'test_system.x')
        self.assertEqual(self.resolver.rel2abs('y'), 'test_system.y')

    def test_absnames(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test getting absolute names
        self.assertEqual(self.resolver.absnames('x', 'input'), ['test_system.x'])
        self.assertEqual(self.resolver.absnames('y', 'output'), ['test_system.y'])

        # Test error for non-existent name
        with self.assertRaises(KeyError):
            self.resolver.absnames('nonexistent', 'input')

    def test_any2abs(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test converting various name types
        self.assertEqual(self.resolver.any2abs('x'), 'test_system.x')
        self.assertEqual(self.resolver.any2abs('test_system.x'), 'test_system.x')
        self.assertEqual(self.resolver.any2abs('y'), 'test_system.y')

        self.assertIsNone(self.resolver.any2abs('nonexistent'))

        self.assertEqual(self.resolver.any2abs('nonexistent', default='test_system.x'),
                         'test_system.x')

    def test_any2prom(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test converting various name types
        self.assertEqual(self.resolver.any2prom('x'), 'x')
        self.assertEqual(self.resolver.any2prom('test_system.x'), 'x')
        self.assertEqual(self.resolver.any2prom('y'), 'y')

        self.assertIsNone(self.resolver.any2prom('nonexistent'))

        self.assertEqual(self.resolver.any2prom('nonexistent', default='x'), 'x')

    def test_prom2prom(self):
        # Create two resolvers
        resolver1 = NameResolver('system1')
        resolver2 = NameResolver('system2')

        # Add mappings to both resolvers
        resolver1.add_mapping('system1.x', 'x', 'input', local=True)
        resolver2.add_mapping('system1.x', 'x', 'input', local=True)

        # Test converting between resolvers
        self.assertEqual(resolver1.prom2prom('x', resolver2), 'x')

        # Test with non-matching names
        self.assertEqual(resolver1.prom2prom('nonexistent', resolver2), 'nonexistent')

    def test_source(self):
        # Setup some mappings and connections
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test error for no connections
        with self.assertRaises(Exception):
            self.resolver.source('test_system.x')

        self.resolver._conns = {'test_system.x': 'test_system.y'}

        # Test getting source for input
        self.assertEqual(self.resolver.source('test_system.x'), 'test_system.y')

        # Test getting source for output
        self.assertEqual(self.resolver.source('test_system.y'), 'test_system.y')

        # Test error for non-existent connection
        with self.assertRaises(Exception):
            self.resolver.source('test_system.z')

    def test_num_proms(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test counting promoted names
        self.assertEqual(self.resolver.num_proms(), 2)
        self.assertEqual(self.resolver.num_proms('input'), 1)
        self.assertEqual(self.resolver.num_proms('output'), 1)

    def test_num_abs(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test counting absolute names
        self.assertEqual(self.resolver.num_abs(), 2)
        self.assertEqual(self.resolver.num_abs('input'), 1)
        self.assertEqual(self.resolver.num_abs('output'), 1)
        self.assertEqual(self.resolver.num_abs('input', local=True), 1)

    def test_is_prom(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test checking promoted names
        self.assertTrue(self.resolver.is_prom('x', 'input'))
        self.assertTrue(self.resolver.is_prom('y', 'output'))
        self.assertFalse(self.resolver.is_prom('nonexistent', 'input'))

    def test_is_abs(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test checking absolute names
        self.assertTrue(self.resolver.is_abs('test_system.x', 'input'))
        self.assertTrue(self.resolver.is_abs('test_system.y', 'output'))
        self.assertFalse(self.resolver.is_abs('nonexistent', 'input'))
        self.assertTrue(self.resolver.is_local('test_system.x', 'input'))

    def test_get_abs_iotype(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test getting iotype for absolute names
        self.assertEqual(self.resolver.get_abs_iotype('test_system.x'), 'input')
        self.assertEqual(self.resolver.get_abs_iotype('test_system.y'), 'output')
        self.assertIsNone(self.resolver.get_abs_iotype('nonexistent'))

        # Test with report_error=True
        with self.assertRaises(KeyError):
            self.resolver.get_abs_iotype('nonexistent', report_error=True)

    def test_get_prom_iotype(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test getting iotype for promoted names
        self.assertEqual(self.resolver.get_prom_iotype('x'), 'input')
        self.assertEqual(self.resolver.get_prom_iotype('y'), 'output')
        self.assertIsNone(self.resolver.get_prom_iotype('nonexistent'))

        # Test with report_error=True
        with self.assertRaises(KeyError):
            self.resolver.get_prom_iotype('nonexistent', report_error=True)

    def test_iterators(self):
        # Setup some mappings
        self.resolver.add_mapping('test_system.x', 'x', 'input', local=True)
        self.resolver.add_mapping('test_system.y', 'y', 'output', local=True)

        # Test prom2abs_iter
        prom2abs = list(self.resolver.prom2abs_iter('input'))
        self.assertEqual(prom2abs, [('x', ['test_system.x'])])

        # Test abs2prom_iter
        abs2prom = list(self.resolver.abs2prom_iter('input'))
        self.assertEqual(abs2prom, [('test_system.x', 'x')])

        # Test prom_iter
        prom = list(self.resolver.prom_iter('input'))
        self.assertEqual(prom, ['x'])

        # Test abs_iter
        abs_names = list(self.resolver.abs_iter('input'))
        self.assertEqual(abs_names, ['test_system.x'])

        _, locflags = _get_flags(local=True, continuous=True, distributed=False)
        # Test info_iter
        flags = list(self.resolver.flags_iter('input'))
        self.assertEqual(flags, [('test_system.x', locflags)])


if __name__ == '__main__':
    unittest.main()