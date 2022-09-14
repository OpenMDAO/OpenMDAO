
import sys
import os
import json
import textwrap
from html import escape
from io import StringIO
from numbers import Number, Integral
from openmdao.utils.notebook_utils import notebook, display, HTML, IFrame, colab
from openmdao.utils.file_utils import text2html

_a2sym = {'center': '^', 'right': '>', 'left': '<'}
_default_align = {
    'int': 'right',
    'real': 'right',
    'bool': 'center',
    'other': 'left',
}


class TableBuilder(object):
    allowed_col_meta = {'header', 'align', 'header_align', 'width', 'format', 'max_width'}

    def __init__(self, rows, headers=None, column_meta=None, precision=4, missingval=None,
                 max_width=None):
        self._raw_rows = rows
        self._column_meta = {}
        self._widths = None
        self._rows = None
        self.missingval = missingval
        self.max_width = max_width

        # these attributes change in subclasses
        self.column_sep = ' | '
        self.top_border = '-'
        self.header_bottom_border = '-'
        self.bottom_border = '-'
        self.left_border = '| '
        self.right_border = ' |'

        # these are the default format strings for the first formatting stage,
        # before the column width is set
        self._default_formats = {
            'real': "{" + f":.{precision}" + "}",
            'int': "{}",
            'bool': "{}",
            'other': "{}",
        }

        # for convenience, allow a user to specify header strings without putting them
        # inside a metadata dict
        if headers is not None:
            hlen = len(list(headers))
            for i, h in enumerate(headers):
                self.update_column_meta(i, header=h)

        if column_meta is not None:
            clen = len(list(column_meta))
            for i, meta in enumerate(column_meta):
                self.update_column_meta(i, **meta)

        if headers is not None and column_meta is not None and hlen != clen:
            raise RuntimeError("Number of headers and number of column metadata dicts must match "
                               f"if both are provided, but {hlen} != {clen}.")

    def _get_srows(self):
        """
        Get table rows with cells converted to strings.

        Returns
        -------
        list
            The list of table rows where each cell is a string.
        """
        if self._rows is None:
            self._update_col_meta_from_rows()
            self._rows = []
            for row in self._raw_rows:
                if self.missingval is not None:
                    row = [self.missingval if v is None else v for v in row]
                self._add_srow(row)

        return self._rows

    def _get_widths(self):
        if self._widths is not None:
            return self._widths  # widths already computed

        rows = self._get_srows()
        ncols = len(self._column_meta)

        if len(self._rows[0]) != ncols:
            raise RuntimeError(f"Number of row entries ({len(self._rows[0])}) must match number of "
                               f"columns ({ncols}) in TableBuilder.")

        self._widths = [0] * ncols

        for row in rows:
            for i, cell in enumerate(row):
                wid = len(cell)
                if wid > self._widths[i]:
                    self._widths[i] = wid

        sorted_meta = sorted(self._column_meta.items(), key=lambda x: x[0])

        for i, meta in sorted_meta:
            wid = len(meta['header'])
            if wid > self._widths[i]:
                self._widths[i] = wid

        total_width = sum(self._widths) + len(self.column_sep) * (ncols - 1) + \
            len(self.left_border) + len(self.right_border)

        # check for case where total table width is specified and we have to set max_width on
        # column(s) as a result
        if self.max_width is not None and self.max_width < total_width:

            winfo = [[i, w] for (i, meta), w in zip(sorted_meta, self._widths)
                     if meta['col_type'] == 'other' and not meta.get('fixed_width')]
            min_width = 10

            fixed_width = total_width - sum([w for _, w in winfo])
            allowed_width = self.max_width - fixed_width

            # subtract 1 from the widest column until we meed the total max_width requirement,
            # or get as close as we can without violating a minimum allowed width.
            while sum([w for _, w in winfo]) > allowed_width:
                sortedw = sorted(winfo, key=lambda x: x[-1], reverse=True)
                if sortedw[0][-1] > min_width:
                    sortedw[0][-1] -= 1
                else:
                    break

            if sum([w for _, w in winfo]) <= allowed_width:
                for i, w in winfo:
                    meta = self._column_meta[i]
                    meta['max_width'] = w

        return self._widths

    def _update_col_meta_from_rows(self):
        types = None
        for row in self._raw_rows:
            if types is None:
                types = [set() for r in row]

            for i, cell in enumerate(row):
                if isinstance(cell, Number):
                    if isinstance(cell, bool):
                        types[i].add('bool')
                    elif isinstance(cell, Integral):
                        types[i].add('int')
                    else:
                        types[i].add('real')
                else:
                    types[i].add('other')

        for i, tset in enumerate(types):
            if len(tset) > 1:
                col_type = 'other'  # mixed type column, so just use "{}" format
            else:
                col_type = tset.pop()

            align = _default_align[col_type]

            meta = {
                'header': '',
                'format': self._default_formats[col_type],
                'align': align,
                'header_align': align,
                'max_width': None,
                'col_type': col_type
            }

            if i in self._column_meta:
                meta.update(self._column_meta[i])

            self._column_meta[i] = meta

    def _add_srow(self, row):
        cells = [self._column_meta[i]['format'].format(cell) for i, cell in enumerate(row)]

        if self._rows and len(cells) != len(self._rows[-1]):
            raise RuntimeError("Can't add rows of unequal length to TableBuilder.")

        self._rows.append(cells)

    def update_column_meta(self, col_idx, **options):
        if col_idx < 0:
            col_idx = len(self._raw_rows[0]) + col_idx
        if col_idx < 0 or col_idx >= len(self._raw_rows[0]):
            raise IndexError(f"Index '{col_idx}' is not a valid table column index for a table with"
                             f" {len(self._raw_rows)} columns.  The leftmost column has column "
                             "index of 0.")
        if col_idx not in self._column_meta:
            self._column_meta[col_idx] = {}
        meta = self._column_meta[col_idx]
        for name, val in options.items():
            if name not in self.allowed_col_meta:
                raise KeyError(f"'{name}' is not a valid column metadata key.")
            meta[name] = val

    def _get_fixed_width_cell(self, col_meta, cell, width, align_name):
        align = col_meta.get(align_name, 'left')
        try:
            sym = _a2sym[align]
        except KeyError:
            raise KeyError(f"Expected one of ['left', 'right', 'center'] for '{align_name}' "
                           f"metadata, but got '{align}'.")
        return f"{cell:{sym}{width}}"

    def needs_wrap(self):
        needs_wrap = self.max_width is not None
        if not needs_wrap:
            for meta in self._column_meta.values():
                if meta['max_width'] is not None:
                    return True
        return needs_wrap

    def _stringified_header_iter(self):
        header_cells = [None] * len(self._column_meta)
        widths = self._get_widths()

        sorted_cols = sorted(self._column_meta.items(), key=lambda x: x[0])

        for i, meta in sorted_cols:
            header_cells[i] = self._get_fixed_width_cell(meta, meta['header'], widths[i],
                                                         'header_align')

        if self.needs_wrap():
            cell_lists = []
            for i, meta in sorted_cols:
                cell = header_cells[i]
                maxwid = meta['max_width']
                if maxwid is not None and maxwid < len(cell):
                    lines = textwrap.wrap(cell, maxwid)
                    # ensure all cells have same width in this column
                    cell_lists.append([f"{line:<{maxwid}}" for line in lines])
                else:
                    cell_lists.append([cell])

            # now find longest column
            maxlen = max([len(lst) for lst in cell_lists])
            for r in range(maxlen):
                cells = []
                for clist in cell_lists:
                    if len(clist) > r:
                        cells.append(clist[r])
                    else:
                        cells.append(' ' * len(clist[0]))
                yield cells
        else:
            yield header_cells

    def _stringified_row_iter(self):
        widths = self._get_widths()
        row_cells = [None] * len(self._column_meta)

        needs_wrap = self.needs_wrap()
        sorted_cols = sorted(self._column_meta.items(), key=lambda x: x[0])

        for row in self._get_srows():
            for i, meta in sorted_cols:
                row_cells[i] = self._get_fixed_width_cell(meta, row[i], widths[i], 'align')

            if needs_wrap:
                cell_lists = []
                for i, meta in sorted_cols:
                    cell = row_cells[i]
                    maxwid = meta['max_width']
                    if maxwid is not None and maxwid < len(cell):
                        lines = textwrap.wrap(cell, maxwid)
                        # ensure all cells have same width in this column
                        cell_lists.append([f"{line:<{maxwid}}" for line in lines])
                    else:
                        cell_lists.append([cell])

                # now find longest column
                maxlen = max([len(lst) for lst in cell_lists])
                for r in range(maxlen):
                    cells = []
                    for clist in cell_lists:
                        if len(clist) > r:
                            cells.append(clist[r])
                        else:
                            w = len(clist[0]) if clist else 0
                            cells.append(' ' * w)
                    yield cells
            else:
                yield row_cells

    def add_side_borders(self, line):
        if self.left_border or self.right_border:
            parts = [p for p in (self.left_border, line, self.right_border) if p]
            line = ''.join(parts)
        return line

    def get_top_border(self, header_cells):
        width = sum(len(h) for h in header_cells) + len(self.column_sep) * (len(header_cells) - 1)
        return self.add_side_borders((self.top_border * width)[:width])

    def get_header_bottom_border(self, header_cells):
        parts = [(self.header_bottom_border * len(h))[:len(h)] for h in header_cells]
        return self.add_side_borders(self.column_sep.join(parts))

    def get_bottom_border(self, header_cells):
        parts = [(self.bottom_border * len(h))[:len(h)] for h in header_cells]
        return self.add_side_borders(self.column_sep.join(parts))

    def write(self, stream=sys.stdout):
        for i, header_cells in enumerate(self._stringified_header_iter()):
            if i == 0 and self.top_border:
                print(self.get_top_border(header_cells), file=stream)

            print(self.add_side_borders(self.column_sep.join(header_cells)), file=stream)

        if self.header_bottom_border:
            print(self.get_header_bottom_border(header_cells), file=stream)

        for row_cells in self._stringified_row_iter():
            print(self.add_side_borders(self.column_sep.join(row_cells)), file=stream)

        if self.bottom_border:
            print(self.get_bottom_border(header_cells), file=stream)

    def __str__(self):
        """
        Return a string representation of the Table.
        """
        io = StringIO()
        self.write(stream=io)
        return io.getvalue()

    def display(self):
        print(self)


class RSTTableBuilder(TableBuilder):
    def __init__(self, rows, **kwargs):
        super().__init__(rows, **kwargs)
        self.column_sep = '  '
        self.top_border = '='
        self.header_bottom_border = '='
        self.bottom_border = '='
        self.left_border = ''
        self.right_border = ''

    def get_top_border(self, header_cells):
        parts = [(self.top_border * len(h))[:len(h)] for h in header_cells]
        return self.add_side_borders(self.column_sep.join(parts))


class GithubTableBuilder(TableBuilder):
    def __init__(self, rows, **kwargs):
        super().__init__(rows, **kwargs)
        self.column_sep = ' | '
        self.top_border = ''
        self.header_bottom_border = '-'
        self.bottom_border = ''
        self.left_border = '| '
        self.right_border = ' |'

    def get_header_bottom_border(self, header_cells):
        parts = []
        for cell, meta in zip(header_cells, self._column_meta.values()):
            align = meta['align']
            left = right = center = ''
            size = len(cell)
            if align == 'left':
                left = ':'
                size -= 1
            elif align == 'right':
                right = ':'
                size -= 1
            else:  # center
                left = right = ':'
                size -= 2
            parts.append(left + (self.header_bottom_border * size) + right)

        return self.add_side_borders(self.column_sep.join(parts))

    def needs_wrap(self):
        return False  # github tables seem to have no support for text wrapping in columns


_tabulator_typemeta = {
    'bool': {
        'align': 'center',
        'formatter': 'tickCross',
        'formatterParams': {'crossElement': False},
        'filter': 'tickCross',
        'headerFilterParams': {'tristate': True},
        'sorter': 'string',
    },
    'int': {
        'align': 'right',
        'formatter': 'plaintext',
        'filter': False,
        'sorter': 'number',
    },
    'real': {
        'align': 'right',
        'formatter': 'plaintext',
        'filter': False,
        'sorter': 'number',
    },
    'other': {
        'align': 'left',
        'formatter': 'textarea',
        'filter': 'input',
        'sorter': 'string',
    }
}


class HTMLTableBuilder(TableBuilder):
    def __init__(self, rows, html_id=None, **kwargs):
        super().__init__(rows, **kwargs)
        self._html_id = html_id

    def _stringified_row_iter(self):
        for row in self._get_srows():
            yield row

    def header_style(self, meta):
        return ''

    def write(self, stream=sys.stdout):
        sorted_cols = sorted(self._column_meta.items(), key=lambda x: x[0])
        ident = f' id="{self._html_id}"' if self._html_id else ''
        print(f"<table{ident}>", file=stream)

        print("   <tr>", file=stream, end='')
        for _, meta in sorted_cols:
            print(f"<th>{escape(meta['header'])}{self.header_style(meta)}</th>",
                  file=stream, end='')
        print("</tr>", file=stream)

        for row_cells in self._stringified_row_iter():
            print("   <tr>", file=stream, end='')
            for cell in row_cells:
                print(f"<td>{escape(cell)}</td>", file=stream, end='')
            print("   </tr>", file=stream)

        print("</table>", file=stream)

    def write_html(self, outfile=None):
        if outfile is None:
            outfile = 'table.html'

        title = ''
        style = """
            tr:nth-child(even) {
                background-color: #D6EEEE;
            }
        """
        table = \
"""
<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        .center {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 90%;
        }
        h2 {text-align: center;}
        tr:nth-child(even) {
            background-color: lightgray;
        }
        tr:hover {background-color: #D6EEEE;}
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
        }
        th, td {
            padding: 5px;
        }
        th {
            text-align: left;
        }
    </style>
</head>
<body>
<h2>""" + title + """</h2>
""" + str(self) + """
</body>
</html>
"""
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(table)

    def display(self, outfile=None):
        if outfile is None:
            outfile = 'table.html'

        self.write_html(outfile)

        if notebook:
            if not colab:
                display(IFrame(src=outfile, width="100%", height=700))
            else:
                display(HTML(outfile))
        else:
            # open it up in the browser
            from openmdao.utils.webview import webview
            webview(outfile)


class TabulatorJSBuilder(TableBuilder):
    allowed_col_meta = TableBuilder.allowed_col_meta.union({
        'filter',
        'header_align',
        'sorter',
        'formatter'
    })

    # allowed_table_meta = {
    #     'id',  # html id
    #     'layout',  # fitData, fitDataStretch, fitDataTable, fitColumns
    #     'height',  # number in pixels
    # }

    def __init__(self, rows, layout='fitColumns', height=None, html_id='tabul-table', title='',
                 display_in_notebook=True, show_browser=True, outfile='tabulator_table.html',
                 **kwargs):
        super().__init__(rows, **kwargs)
        self._table_meta = {
            'layout': layout,
            'height': height,
            'id': html_id if html_id.startswith('#') else '#' + html_id,
        }
        self._title = title
        self._display_in_notebook = display_in_notebook
        self._show_browser = show_browser
        self._outfile = outfile

    def _stringified_row_iter(self):
        for row in self._get_srows():
            yield row

    def _add_srow(self, row):
        cells = []
        for i, cell in enumerate(row):
            # If we convert booleans to strings for Tabulator the tri-state sorting won't work
            if isinstance(cell, bool):
                cells.append(cell)
            else:
                cells.append(self._column_meta[i]['format'].format(cell))

        if self._rows and len(cells) != len(self._rows[-1]):
            raise RuntimeError("Can't add rows of unequal length to TableBuilder.")

        self._rows.append(cells)

    def _update_col_meta_from_rows(self):
        types = [set() for r in self._raw_rows[0]]
        for row in self._raw_rows:
            for i, cell in enumerate(row):
                if isinstance(cell, Number):
                    if isinstance(cell, bool):
                        types[i].add('bool')
                    elif isinstance(cell, Integral):
                        types[i].add('int')
                    else:
                        types[i].add('real')
                else:
                    types[i].add('other')

        for i, tset in enumerate(types):
            if len(tset) > 1:
                col_type = 'other'  # mixed type column, so just use "{}" format
            else:
                col_type = tset.pop()

            meta = _tabulator_typemeta[col_type].copy()
            meta['format'] = self._default_formats[col_type]
            meta['header_align'] = meta['align']
            meta['max_width'] = None
            meta['col_type'] = col_type

            if i in self._column_meta:
                meta.update(self._column_meta[i])

            self._column_meta[i] = meta

    def get_table_data(self):
        rows = []
        idx = 1  # unique ID for use by Tabulator

        for row_cells in self._stringified_row_iter():
            dct = {'id': idx}
            for i, cell in enumerate(row_cells):
                dct[f'c{i}'] = cell
            rows.append(dct)
            idx += 1

        cols = []
        for i, meta in  sorted(self._column_meta.items(), key=lambda x: x[0]):
            cmeta = {
                'field': f'c{i}',
                'title': meta['header'],
                'maxWidth': meta['max_width'],
                'width': meta.get('width'),
                'hozAlign': meta['align'],
                'headerHozAlign': meta['header_align'],
                'headerFilter': meta['filter'],  # input, textarea, number, range, tickCross
                'sorter': meta['sorter'],  # string, number, alphanum, boolean, exists
                'formatter': meta['formatter'],  # plaintext, textarea, html, money, image, link,
                                                 # tickCross, traffic, star, progress, color,
                                                 # buttonTick, buttonCross,
                'formatterParams': meta.get('formatterParams', None),
                'editor': meta.get('editor', None),
                'editorParams': meta.get('editorParams', None),
                'headerFilterParams': meta.get('headerFilterParams', None),
            }

            cols.append(cmeta)

        # for big tables, use virtual DOM for speed (setting height activates it)
        if len(self._raw_rows) > 75 and self._table_meta['height'] is None:
            self._table_meta['height'] = 600

        return {
            'title': self._title,
            'rows': rows,
            'cols': cols,
            'meta': self._table_meta,
        }

    def write_html(self, outfile=None):
        import openmdao.visualization

        if outfile is None:
            outfile = self._outfile

        outfile = os.path.relpath(outfile)

        code_dir = os.path.dirname(openmdao.visualization.__file__)
        libs_dir = os.path.join(code_dir, 'common', 'libs')
        style_dir = os.path.join(code_dir, 'common', 'style')

        with open(os.path.join(code_dir, 'generic_table.html'), "r", encoding='utf-8') as f:
            template = f.read()

        with open(os.path.join(libs_dir, 'tabulator.min.js'), "r", encoding='utf-8') as f:
            tabulator_src = f.read()
        s = template.replace("<tabulator_src>", tabulator_src)
        del tabulator_src

        with open(os.path.join(style_dir, 'tabulator.min.css'), "r", encoding='utf-8') as f:
            tabulator_style = f.read()
        s = s.replace("<tabulator_style>", tabulator_style)

        jsontxt = json.dumps(self.get_table_data())
        s = s.replace("<table_data>", jsontxt)
        del jsontxt

        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(s)

    def __str__(self):
        """
        Return a string representation of the Table.
        """
        return ''

    def display(self, outfile=None):
        if outfile is None:
            outfile = self._outfile

        self.write_html(outfile)
        if notebook:
            if not colab:
                display(IFrame(src=self._outfile, width="100%", height=700))
            else:
                display(HTML(self._outfile))
        else:
            # open it up in the browser
            from openmdao.utils.webview import webview
            webview(outfile)


_table_types = {
    'text': TableBuilder,
    'github': GithubTableBuilder,
    'rst': RSTTableBuilder,
    'tabulator': TabulatorJSBuilder,
    'html': HTMLTableBuilder,
}


def to_table(rows, tablefmt='text', **options):
    try:
        table_class = _table_types[tablefmt]
    except Exception:
        raise KeyError(f"'{tablefmt}' is not a valid type choice for to_table.  Valid choices are: "
                       f"{sorted(_table_types)}.")

    return table_class(rows, **options)


if __name__ == '__main__':
    import numpy as np
    import sys

    coltypes = [
        ('str', {'maxsize': 20}),
        ('real', {'low': -1e10, 'high': 1e10}),
        ('bool', {}),
        ('int', {'low': -99, 'high': 2500}),
        ('str', {'maxsize': 50}),
        ('str', {'maxsize': 50}),
    ]

    try:
        tablefmt = sys.argv[1]
    except Exception:
        tablefmt = 'text'

    from openmdao.utils.tests.test_tables import random_table
    tab = random_table(tablefmt=tablefmt, coltypes=coltypes, nrows=30, html_id='foobar')

    tab.max_width = 100
    tab.display()
