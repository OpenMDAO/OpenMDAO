import unittest
import numpy as np
import math
from html.parser import HTMLParser

import openmdao.api as om
from openmdao.visualization.tables.table_builder import generate_table
from openmdao.utils.testing_utils import use_tempdirs


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
| --------- |"""
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
| ------------------ |"""
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

        # check with dict and headers='keys'
        tdict = {
            'Col0': [0,3,6],
            'Col1': [1,4,7],
            'Col2': [2,5,8],
        }
        self.check_html(tdict, 'keys', [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']], headers)

        # check list of dicts
        dlist = [
            {'Col0': 0, 'Col1': 1, 'Col2': 2},
            {'Col0': 3, 'Col1': 4, 'Col2': 5},
            {'Col0': 6, 'Col1': 7, 'Col2': 8},
            ]
        self.check_html(dlist, 'keys', [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']], headers)

    def test_missing_vals(self):
        headers = ['Strings', 'Floats', 'Something else']
        expected = """
| ----------------------------------------------- |
| Strings               | Floats | Something else |
| --------------------- | ------ | -------------- |
| foobar blah           |    1.0 | N/A            |
| asdfas dffff          |  3.142 | N/A            |
| hello world blah blah |   9.87 | N/A            |
| ----------------------------------------------- |
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
| ---------------------------------- |
"""
        self.check_text('text', self.table_row_iter('str', 'float', None), headers, expected, missing_val='N/A', max_width=38)

    def test_word_wrap_box_grid(self):
        headers = ['Strings', 'Floats', 'Something else']
        expected = """
╔═════════════╤════════╤═════════════╗
║ Strings     ┊ Floats ┊ Something   ║
║             ┊        ┊ else        ║
╠═════════════╪════════╪═════════════╣
║ foobar blah ┊    1.0 ┊ N/A         ║
╟┈┈┈┈┈┈┈┈┈┈┈┈┈┿┈┈┈┈┈┈┈┈┿┈┈┈┈┈┈┈┈┈┈┈┈┈╢
║ asdfas      ┊  3.142 ┊ N/A         ║
║ dffff       ┊        ┊             ║
╟┈┈┈┈┈┈┈┈┈┈┈┈┈┿┈┈┈┈┈┈┈┈┿┈┈┈┈┈┈┈┈┈┈┈┈┈╢
║ hello world ┊   9.87 ┊ N/A         ║
║ blah blah   ┊        ┊             ║
╚═════════════╧════════╧═════════════╝
"""
        self.check_text('box_grid', self.table_row_iter('str', 'float', None), headers, expected, missing_val='N/A', max_width=38)

    def test_align(self):
        headers = ['Strings', 'Floats', 'Something else']
        column_meta = [{'header_align': 'center'}, {}, {'align': 'center', 'header_align': 'center'}]
        expected = """
| ---------------------------------- |
|   Strings   | Floats |  Something  |
|             |        |    else     |
| ----------- | ------ | ----------- |
| foobar blah |    1.0 |     N/A     |
| asdfas      |  3.142 |     N/A     |
| dffff       |        |             |
| hello world |   9.87 |     N/A     |
| blah blah   |        |             |
| ---------------------------------- |
"""
        self.check_text('text', self.table_row_iter('str', 'float', None), headers, expected,
                        column_meta=column_meta, missing_val='N/A', max_width=38)

    def test_basic_tabulator(self):
        # for now, just check that it doesn't crash
        headers = ['Strings', 'Floats', 'Something else']
        table = generate_table(self.table_row_iter('str', 'float', None),
                               tablefmt='tabulator', headers=headers)
        tstr = str(table)

    def test_embedded_newline(self):
        cells = [
            ["#", "title A", "title B"],
            ["1", "lorem", "ipsum dolor sit amet"],
            ["2", "lorem", "ipsum\ndolor sit amet"],
            ["3", "lorem", "ipsum dolor sit amet"],
        ]
        expected = """
╔═══╤═════════╤══════════════════════╗
║ # ┊ title A ┊ title B              ║
╠═══╪═════════╪══════════════════════╣
║ 1 ┊ lorem   ┊ ipsum dolor sit amet ║
╟┈┈┈┿┈┈┈┈┈┈┈┈┈┿┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈╢
║ 2 ┊ lorem   ┊ ipsum                ║
║   ┊         ┊ dolor sit amet       ║
╟┈┈┈┿┈┈┈┈┈┈┈┈┈┿┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈╢
║ 3 ┊ lorem   ┊ ipsum dolor sit amet ║
╚═══╧═════════╧══════════════════════╝
"""
        self.check_text('box_grid', cells, 'firstrow', expected)

    def test_embedded_newline2(self):
        cells = [
            ["#", "title A", "title B"],
            ["1", "lorem", "ipsum dolor sit amet"],
            ["2", "lorem", "ipsum\ndolor sit\namet"],
            ["3", "lorem", "ipsum dolor sit amet"],
        ]
        expected = """
╔═══╤═════════╤══════════════════════╗
║ # ┊ title A ┊ title B              ║
╠═══╪═════════╪══════════════════════╣
║ 1 ┊ lorem   ┊ ipsum dolor sit amet ║
╟┈┈┈┿┈┈┈┈┈┈┈┈┈┿┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈╢
║ 2 ┊ lorem   ┊ ipsum                ║
║   ┊         ┊ dolor sit            ║
║   ┊         ┊ amet                 ║
╟┈┈┈┿┈┈┈┈┈┈┈┈┈┿┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈╢
║ 3 ┊ lorem   ┊ ipsum dolor sit amet ║
╚═══╧═════════╧══════════════════════╝
"""
        self.check_text('box_grid', cells, 'firstrow', expected)

    def test_non_string_header(self):
        cells = [
            ["#", "title A", "1.0"],
            ["1", "lorem", "ipsum dolor sit amet"],
            ["2", "lorem", "ipsum dolor sit amet"],
            ["3", "lorem", "ipsum dolor sit amet"],
        ]
        expected = """
╔═══╤═════════╤══════════════════════╗
║ # ┊ title A ┊ 1.0                  ║
╠═══╪═════════╪══════════════════════╣
║ 1 ┊ lorem   ┊ ipsum dolor sit amet ║
╟┈┈┈┿┈┈┈┈┈┈┈┈┈┿┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈╢
║ 2 ┊ lorem   ┊ ipsum dolor sit amet ║
╟┈┈┈┿┈┈┈┈┈┈┈┈┈┿┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈╢
║ 3 ┊ lorem   ┊ ipsum dolor sit amet ║
╚═══╧═════════╧══════════════════════╝
"""
        self.check_text('box_grid', cells, 'firstrow', expected)
