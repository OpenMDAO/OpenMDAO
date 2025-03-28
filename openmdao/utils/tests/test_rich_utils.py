import unittest
from io import StringIO


try:
    import rich
except ImportError:
    rich = None

from openmdao.utils.rich_utils import rich_wrap, strip_formatting, strip_tags


class TestRichUtils(unittest.TestCase):
    def test_noop(self):
        s = 'foo bar baz'
        self.assertEqual(rich_wrap(s), s)

    def test_tag_set(self):
        s = 'foo bar baz'
        # Rich requires closing tag have ordering opposite of opeing tag.
        expected = f'[bold bright_red]{s}[/bright_red bold]'
        self.assertEqual(rich_wrap(s, {'bold', 'bright_red'}), expected)

    def test_strip_tags(self):
        s = 'foo bar baz'
        # Rich requires closing tag have ordering opposite of opeing tag.
        expected = f'[bold bright_red]{s}[/bright_red bold]'
        self.assertEqual(rich_wrap(s, {'bold', 'bright_red'}), expected)
        self.assertEqual(strip_tags(expected), s)

    @unittest.skipIf(rich is None, 'requires rich python package')
    def test_strip_formatting(self):
        s = 'foo bar baz'
        # Rich requires closing tag have ordering opposite of opeing tag.
        s = rich_wrap(s, {'bold', 'bright_red'})

        buf = StringIO()
        c = rich.console.Console(force_terminal=True, file=buf)
        c.print(s)
        test_val = repr(buf.getvalue())
        self.assertEqual(test_val.count(r'\x1b'), 2)

        buf = StringIO()
        c = rich.console.Console(force_terminal=True, file=buf)
        test_val = strip_formatting(buf.getvalue())
        self.assertEqual(test_val.count(r'\x1b'), 0)


if __name__ == '__main__':
    unittest.main()
