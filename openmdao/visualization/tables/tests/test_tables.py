import unittest
import numpy as np
import math
from html.parser import HTMLParser
from html.entities import name2codepoint

from openmdao.visualization.tables.table_builder import generate_table
from openmdao.utils.testing_utils import use_tempdirs


# limit our random strings to upper, lower case letters, numbers, :, _, and space
_char_map = [chr(i) for i in range(48, 59)]  # numbers + ':'
_char_map.extend([chr(i) for i in range(65, 91)])  # CAPS
_char_map.extend([chr(i) for i in range(97, 123)])  # lower case
_char_map.append(chr(95))  # '_'


#  generators for random table cells

def _str_gen(nvals, nwords=20, maxsize=8, seed=None):
    randgen = np.random.default_rng(seed)
    ctop = len(_char_map)
    for i in range(nvals):
        words = []
        for _ in range(nwords):
            sz = randgen.integers(low=min(4, maxsize), high=maxsize + 1)
            words.append(''.join([_char_map[c] for c in randgen.integers(ctop, size=sz)]))
        yield ' '.join(words)


def _bool_gen(nvals, seed=None):
    randgen = np.random.default_rng(seed)
    for i in range(nvals):
        yield randgen.random() > .5


def _real_gen(nvals, low=-10000., high=10000., seed=None):
    randgen = np.random.default_rng(seed)
    mult = high - low
    for i in range(nvals):
        yield randgen.random() * mult + low


def _int_gen(nvals, low=-10000, high=10000, seed=None):
    randgen = np.random.default_rng(seed)
    for i in range(nvals):
        yield randgen.integers(low=low, high=high)


_cell_creators = {
    'real': _real_gen,
    'int': _int_gen,
    'bool': _bool_gen,
    'str': _str_gen,
}

def _create_random_table_data(coltypes, nrows, seed=None):
    colgens = []
    for t, kwargs in coltypes:
        colgens.append(_cell_creators[t](nrows, seed=seed, **kwargs))
        if seed is not None:
            seed += 1

    headers = [s for s in _str_gen(len(coltypes), nwords=2, maxsize=5, seed=seed)]
    rows = []
    for i in range(nrows):
        rows.append([next(cg) for cg in colgens])

    return headers, rows


def random_table(tablefmt='text', nrows=10, coltypes=None, seed=None, **options):
    if coltypes is None:
        coltypes = [
            ('str', {'maxsize': 40}),
            ('real', {'low': -1e10, 'high': 1e10}),
            ('str', {'maxsize': 50}),
            ('bool', {}),
            ('str', {'maxsize': 10}),
            ('int', {'low': -99, 'high': 2500}),
        ]

    headers, data = _create_random_table_data(coltypes, nrows, seed=seed)

    return generate_table(data, tablefmt=tablefmt, headers=headers, **options)


class HTMLTableParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.found_table = False
        self.table_finished = False
        self.rowcount = 0
        self.headers = []
        self.rows = []
        self.stack = []
        self.current_row = None

    def check_stack(self, tag):
        top = self.stack[-1][0] if self.stack else None
        if top != tag:
            raise RuntimeError(f"Expected '{tag}' at top of stack but found '{top}'.")
        return self.stack[-1]

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.found_table = True
            self.stack.append([tag, None])
        elif tag in ('td', 'th'):
            self.stack.append([tag, None])
        elif tag == 'tr':
            self.stack.append([tag, None, None])

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            top = self.check_stack(tag)
            data = top[1]
            self.stack.pop()
            top = self.check_stack('tr')
            if top[1] is None:
                top[1] = []
                top[2] = tag
            elif top[2] != tag:
                raise RuntimeError("Mixed <tr> and <td> on same row!")
            top[1].append(data)
        elif tag == 'tr':
            top = self.check_stack(tag)
            if top[2] == 'td':
                self.rows.append(top[1])
            elif top[2] == 'th':
                self.headers = top[1]
            else:
                raise RuntimeError("Found empty <tr>")
            self.stack.pop()
        elif tag == 'table':
            self.table_finished = True
            if not self.stack or self.stack[-1][0] != 'table':
                raise RuntimeError("</table> doesn't have matching <table>")
            self.stack.pop()

    def handle_data(self, data):
        if self.stack and self.stack[-1][0] in ('td', 'th'):
            self.stack[-1][1] = data


@use_tempdirs
class TableTestCase(unittest.TestCase):
    def check_html(self, rows, headers, expected_rows, expected_headers, **kwargs):
        table = generate_table(rows, tablefmt='html', headers=headers, **kwargs)
        parser = HTMLTableParser()
        parser.feed(str(table))
        self.assertEqual(parser.headers, expected_headers)
        self.assertEqual(parser.rows, expected_rows)

    def check_text(self, fmt, rows, headers, expected, **kwargs):
        table = generate_table(rows, tablefmt=fmt, headers=headers, **kwargs)
        tstr = str(table)
        self.assertEqual(tstr.strip(), expected.strip())

    def table_row_iter(self, c1typ, c2typ, c3typ):
        def yield_ints(n):
            for i in range(n):
                yield i
        def yield_floats(n):
            for i in range(n):
                yield math.pi ** i
        def yield_strs(n):
            base = ['foobar blah', 'asdfas dffff', 'hello world blah blah']
            mult = n // len(base)
            strs = (base * (mult + 1))[:n]
            for s in strs:
                yield s
        def yield_nones(n):
            for i in range(n):
                yield None

        tdict = {
            'int': yield_ints,
            'float': yield_floats,
            'str': yield_strs,
            None: yield_nones,
        }
        colgens = [tdict[c1typ](3), tdict[c2typ](3), tdict[c3typ](3)]
        for i in range(3):
            row = [next(cg) for cg in colgens]
            yield row


    def test_no_header_text(self):
        expected = """
| --------- |
| 0 | 1 | 2 |
| 3 | 4 | 5 |
| 6 | 7 | 8 |
| - | - | - |"""
        self.check_text('text', np.arange(9).reshape((3,3)), None, expected)
        self.check_text('text', [[0,1,2],[3,4,5],[6,7,8]], None, expected)
        self.check_text('text', iter(np.arange(9).reshape((3,3))), None, expected)

    def test_no_header_github(self):
        expected = """
| 0 | 1 | 2 |
| 3 | 4 | 5 |
| 6 | 7 | 8 |"""
        self.check_text('github', np.arange(9).reshape((3,3)), None, expected)
        self.check_text('github', [[0,1,2],[3,4,5],[6,7,8]], None, expected)

    def test_no_header_rst(self):
        expected = """
=  =  =
0  1  2
3  4  5
6  7  8
=  =  ="""
        self.check_text('rst', np.arange(9).reshape((3,3)), None, expected)
        self.check_text('rst', [[0,1,2],[3,4,5],[6,7,8]], None, expected)

    def test_w_header_text(self):
        headers = [f"Col{i}" for i in range(3)]
        expected = """
| ------------------ |
| Col0 | Col1 | Col2 |
| ---- | ---- | ---- |
|    0 |    1 |    2 |
|    3 |    4 |    5 |
|    6 |    7 |    8 |
| ---- | ---- | ---- |"""
        self.check_text('text', np.arange(9).reshape((3,3)), headers, expected)
        self.check_text('text', [[0,1,2],[3,4,5],[6,7,8]], headers, expected)

    def test_w_header_rst(self):
        headers = [f"Col{i}" for i in range(3)]
        expected = """
====  ====  ====
Col0  Col1  Col2
====  ====  ====
   0     1     2
   3     4     5
   6     7     8
====  ====  ====
"""
        self.check_text('rst', np.arange(9).reshape((3,3)), headers, expected)
        self.check_text('rst', [[0,1,2],[3,4,5],[6,7,8]], headers, expected)

    def test_w_header_github(self):
        headers = [f"Col{i}" for i in range(3)]
        expected = """
| Col0 | Col1 | Col2 |
| ---: | ---: | ---: |
|    0 |    1 |    2 |
|    3 |    4 |    5 |
|    6 |    7 |    8 |
"""
        self.check_text('github', np.arange(9).reshape((3,3)), headers, expected)
        self.check_text('github', [[0,1,2],[3,4,5],[6,7,8]], headers, expected)

    def test_no_header_html(self):
        self.check_html(np.arange(9).reshape((3,3)), None, [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']], [])
        self.check_html([[0,1,2],[3,4,5],[6,7,8]], None, [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']], [])

    def test_w_header_html(self):
        headers = [f"Col{i}" for i in range(3)]
        self.check_html(np.arange(9).reshape((3,3)), headers, [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']], headers)
        self.check_html([[0,1,2],[3,4,5],[6,7,8]], headers, [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']], headers)

    def test_missing_vals(self):
        headers = ['Strings', 'Floats', 'Something else']
        expected = """
| ----------------------------------------------- |
| Strings               | Floats | Something else |
| --------------------- | ------ | -------------- |
| foobar blah           |    1.0 | N/A            |
| asdfas dffff          |  3.142 | N/A            |
| hello world blah blah |   9.87 | N/A            |
| --------------------- | ------ | -------------- |
"""
        self.check_text('text', self.table_row_iter('str', 'float', None), headers, expected, missing_val='N/A')

    def test_word_wrap(self):
        headers = ['Strings', 'Floats', 'Something else']
        expected = """
| ---------------------------------- |
| Strings     | Floats | Something   |
|             |        | else        |
| ----------- | ------ | ----------- |
| foobar blah |    1.0 | N/A         |
| asdfas      |  3.142 | N/A         |
| dffff       |        |             |
| hello world |   9.87 | N/A         |
| blah blah   |        |             |
| ----------- | ------ | ----------- |
"""
        self.check_text('text', self.table_row_iter('str', 'float', None), headers, expected, missing_val='N/A', max_width=38)

    def test_align(self):
        headers = ['Strings', 'Floats', 'Something else']
        column_meta = [{'header_align': 'center'}, {}, {'align': 'center'}]
        expected = """
| ---------------------------------- |
|   Strings   | Floats | Something   |
|             |        | else        |
| ----------- | ------ | ----------- |
| foobar blah |    1.0 |     N/A     |
| asdfas      |  3.142 |     N/A     |
| dffff       |        |             |
| hello world |   9.87 |     N/A     |
| blah blah   |        |             |
| ----------- | ------ | ----------- |
"""
        self.check_text('text', self.table_row_iter('str', 'float', None), headers, expected,
                        column_meta=column_meta, missing_val='N/A', max_width=38)

if __name__ == '__main__':
    import sys

    try:
        formats = [sys.argv[1]]
    except Exception:
        formats = ['rst', 'github', 'text', 'html', 'tabulator']

    rows = np.arange(9).reshape((3,3))
    for fmt in formats:
        tab = generate_table(rows, tablefmt=fmt)
        tab.max_width = 120
        tab.display()
