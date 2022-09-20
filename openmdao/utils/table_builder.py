
import sys
import os
import json
import textwrap
from html import escape
from io import StringIO
from math import floor
from numbers import Number, Integral
from openmdao.utils.notebook_utils import notebook, display, HTML, IFrame, colab
from openmdao.utils.file_utils import text2html
from openmdao.utils.om_warnings import issue_warning


_a2sym = {'center': '^', 'right': '>', 'left': '<'}
_default_align = {
    'int': 'right',
    'real': 'right',
    'bool': 'center',
    'other': 'left',
}


class TableBuilder(object):
    allowed_col_meta = {'header', 'align', 'header_align', 'width', 'format',
                        'max_width', 'min_width', 'fixed_width'}

    def __init__(self, rows, headers=None, column_meta=None, precision=4, missingval=None,
                 max_width=None):
        self._raw_rows = rows # original rows passed in
        self._rows = None  # rows after initial formatting (total width not set)
        self._column_meta = {}
        self._data_widths = None  # width of data in each cell before a uniform column width is set
        self._header_widths = None  # width of headers before a uniform column width is set
        self.missingval = missingval
        self.max_width = max_width

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

    def sorted_meta(self):
        return [m for _, m in sorted(self._column_meta.items(), key=lambda x: x[0])]

    def _get_formatted_rows(self):
        """
        Get table rows with cells after initial formatting, before final width setting.

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

    def _cell_width(self, cell):
        return len(cell)

    def _header_cell_width(self, content_width):
        return content_width

    def _get_cell_types(self):
        types = [set() for r in self._raw_rows[0]] if self._raw_rows else []
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

        for tset in types:
            if len(tset) > 1:
                yield 'other'  # mixed type column, so just use "{}" format
            else:
                yield tset.pop()

    def update_column_meta(self, col_idx, **options):
        if col_idx < 0:  # allow negative indices
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
                raise KeyError(f"'{name}' is not a valid column metadata key. Allowed keys are "
                               f"{sorted(self.allowed_col_meta)}.")
            meta[name] = val

    def _set_widths(self):
        if self._data_widths is not None:
            return  # widths already computed

        rows = self._get_formatted_rows()
        ncols = len(self._column_meta)

        if len(self._rows[0]) != ncols:
            raise RuntimeError(f"Number of row entries ({len(self._rows[0])}) must match number of "
                               f"columns ({ncols}) in TableBuilder.")

        self._data_widths = [0] * ncols
        self._header_widths = [0] * ncols

        for row in rows:
            for i, cell in enumerate(row):
                wid = self._cell_width(cell)
                if wid > self._data_widths[i]:
                    self._data_widths[i] = wid

        # set widths and min widths
        for i, meta in enumerate(self.sorted_meta()):
            self._header_widths[i] = len(meta['header'])

        self._set_min_widths()
        self._set_max_column_widths()

    def _format_cell(self, meta, cell):
        return meta['format'].format(cell)

    def _add_srow(self, row):
        if self._rows and len(row) != len(self._rows[-1]):
            raise RuntimeError("Can't add rows of unequal length to TableBuilder.")

        self._rows.append([self._format_cell(self._column_meta[i], cell)
                           for i, cell in enumerate(row)])

    def _update_col_meta_from_rows(self):
        """
        Fill in missing column metadata based on the data types of column contents.
        """
        for i, col_type in enumerate(self._get_cell_types()):
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

    def needs_wrap(self):
        """
        Return True if the width of the table or any column exceeds the its specified max_width.
        """
        needs_wrap = self.max_width is not None and self.max_width < self._get_total_width()
        if not needs_wrap:
            for meta in self._column_meta.values():
                if meta['max_width'] is not None:
                    return True
        return needs_wrap

    def _set_min_widths(self):
        for meta, wcell, wheader in zip(self.sorted_meta(), self._data_widths, self._header_widths):
            header = meta['header']

            # check if header is splittable into words, and if so, allow for min_width using a
            # split header if min_width will be > min data width.
            longest_part = max(len(word) for word in header.strip().split())
            wheader = self._header_cell_width(longest_part)

            if meta['col_type'] == 'other':  # strings
                meta['min_width'] = max(10, wheader)
            else:
                meta['min_width'] = max(wcell, wheader)

    def get_max_total_col_width(self, winfo):
        fixed_width = self._get_total_width() - sum([w for _, w, _ in winfo])
        return self.max_width - fixed_width

    def _set_max_column_widths(self):
        # check for case where total table width is specified and we have to set max_width on
        # column(s) as a result
        if self.max_width is not None and self.max_width < self._get_total_width():
            winfo = [[i, w, meta['min_width']]
                      for (i, meta), w in zip(enumerate(self.sorted_meta()),
                                              self._get_column_widths())
                      if not meta.get('fixed_width')]

            allowed_width = self.get_max_total_col_width(winfo)

            # subtract 1 from the widest column until we meet the total max_width requirement,
            # or get as close as we can without violating a minimum allowed width.
            while sum([w for i, w, _ in winfo]) > allowed_width:
                for info in sorted(winfo, key=lambda x: x[1], reverse=True):
                    _, width, min_width = info
                    if width - 1 >= min_width:
                        info[1] -= 1
                        break
                else:
                    break

            if sum([w for _, w, _ in winfo]) <= allowed_width:
                for i, w, _ in winfo:
                    self._column_meta[i]['max_width'] = w

            return winfo

    def _get_column_widths(self):
        return [max(wd, wh) for wd, wh in zip(self._data_widths, self._header_widths)]

    def __str__(self):
        """
        Return a string representation of the Table.
        """
        io = StringIO()
        self.write(stream=io)
        return io.getvalue()


class TextTableBuilder(TableBuilder):

    def __init__(self, rows, column_sep=' | ', top_border='-', header_bottom_border='-',
                 bottom_border='-', left_border='| ', right_border=' |', **kwargs):
        super().__init__(rows, **kwargs)

        self.column_sep = column_sep
        self.top_border = top_border
        self.header_bottom_border = header_bottom_border
        self.bottom_border = bottom_border
        self.left_border = left_border
        self.right_border = right_border

    def _get_total_width(self):
        return sum(self._get_column_widths()) + len(self.column_sep) * (len(self._column_meta) - 1)\
            + len(self.left_border) + len(self.right_border)

    def _get_fixed_width_cell(self, col_meta, cell, width, align_name):
        """
        Return a string of the specified width and alignment for the given cell.

        Parameters
        ----------
        col_meta : dict
            Metadata for the current column.
        cell : object
            The cell data.
        width : int
            The desired width of the returned cell string.
        align_name : str
            The name of the alignment metadata, either 'align' or 'header_align', used to retrieve
            the corresponding alignment symbol needed for string formatting.

        Returns
        -------
        str
            A string of the specified width containing the cell data, aligned as specified.
        """
        align = col_meta.get(align_name, 'left')
        try:
            sym = _a2sym[align]
        except KeyError:
            raise KeyError(f"Expected one of ['left', 'right', 'center'] for '{align_name}' "
                           f"metadata, but got '{align}'.")
        return f"{cell:{sym}{width}}"

    def get_lengthened_columns(self, sorted_cols, header_cells):
        """
        Yield as many rows of cells as needed to allow for multi-line cells due to work wrapping.
        """
        cell_lists = []
        for meta, cell in zip(sorted_cols, header_cells):
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

    def _stringified_header_iter(self):
        """
        Yields one or more rows of header cells.
        """
        header_cells = [None] * len(self._column_meta)
        self._set_widths()

        widths = self._get_column_widths()

        sorted_cols = self.sorted_meta()

        for i, meta in enumerate(sorted_cols):
            header_cells[i] = self._get_fixed_width_cell(meta, meta['header'], widths[i],
                                                         'header_align')

        if self.needs_wrap():
            yield from self.get_lengthened_columns(sorted_cols, header_cells)
        else:
            yield header_cells

    def _stringified_row_iter(self):
        """
        Yields rows of data cells, allowing for multi-line rows due to word wrapping.
        """
        self._set_widths()
        widths = self._get_column_widths()
        row_cells = [None] * len(self._column_meta)

        needs_wrap = self.needs_wrap()
        sorted_cols = self.sorted_meta()

        for row in self._get_formatted_rows():
            for i, meta in enumerate(sorted_cols):
                row_cells[i] = self._get_fixed_width_cell(meta, row[i], widths[i], 'align')

            if needs_wrap:
                cell_lists = []
                for meta, cell in zip(sorted_cols, row_cells):
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

    def display(self):
        print(self)


class RSTTableBuilder(TextTableBuilder):
    def __init__(self, rows, **kwargs):
        super().__init__(rows, column_sep='  ', top_border='=', header_bottom_border='=',
                         bottom_border='=', left_border='', right_border='', **kwargs)

    def get_top_border(self, header_cells):
        parts = [(self.top_border * len(h))[:len(h)] for h in header_cells]
        return self.add_side_borders(self.column_sep.join(parts))


class GithubTableBuilder(TextTableBuilder):
    def __init__(self, rows, **kwargs):
        super().__init__(rows, column_sep=' | ', top_border='', header_bottom_border='-',
                         bottom_border='', left_border='| ', right_border=' |', **kwargs)

    def get_header_bottom_border(self, header_cells):
        parts = []
        for i, cell in enumerate(header_cells):
            meta = self._column_meta[i]
            align = meta['align']
            left = right = ''
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


class HTMLTableBuilder(TableBuilder):
    allowed_col_meta = TableBuilder.allowed_col_meta.union({
        'header_style',
        'style',
    })

    def __init__(self, rows, html_id=None, title='', style=None, **kwargs):
        super().__init__(rows, **kwargs)
        tstyle = {
            'margin': 'auto',
        }
        if style is not None:
            tstyle.update(style)
        self._style = tstyle
        self._html_id = html_id
        self._title = title

    def _stringified_row_iter(self):
        for row in self._get_formatted_rows():
            yield row

    def _header_style(self, meta):
        if 'header_style' in meta and meta['header_style']:
            parts = ' '.join([f"{name}: {val};" for name, val in meta['header_style'].items()])
            if parts:
                return f' style="{parts}"'
        return ''

    def _get_type_style(self, typename):
        style = self._type_styles[typename]
        if style:
            parts = ' '.join([f"{name}: {val};" for name, val in style.items()])
            if parts:
                return f' style="{parts}"'
        return ''

    def _get_style(self):
        parts = ' '.join([f"{name}: {val};" for name, val in self._style.items()])
        return f' style="{parts}"' if parts else ''

    def _get_id(self):
        return f' id="{self._html_id}"' if self._html_id else ''

    def write(self, stream=sys.stdout):
        print(f'<table{self._get_id()}{self._get_style()}>', file=stream)

        print("   <tr>", file=stream, end='')
        for meta in self.sorted_meta():
            print(f"<th{self._header_style(meta)}>{escape(meta['header'])}</th>",
                  file=stream, end='')
        print("</tr>", file=stream)

        for row_cells in self._stringified_row_iter():
            print("   <tr>", file=stream, end='')
            for cell, meta in zip(row_cells, self.sorted_meta()):
                typename = meta['col_type']
                print(f'<td class="{typename}_col">{escape(cell)}</td>', file=stream, end='')
            print("   </tr>", file=stream)

        print("</table>", file=stream)

    def write_html(self, outfile=None):
        if outfile is None:
            outfile = 'table.html'

        table = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <style>
                    h2 {{
                        text-align: center;
                    }}
                    tr:nth-child(odd) {{
                        background-color: #EFEFEF;
                    }}
                    tr:hover {{
                        background-color: #D6EEEE;
                    }}
                    table, th, td {{
                        border: 1px solid #999;
                        border-collapse: collapse;
                    }}
                    th, td {{
                        padding: 5px;
                    }}
                    th {{
                        text-align: center;
                        background-color: #e6e6e6;
                    }}
                    .real_col {{
                        text-align: right;
                    }}
                    .int_col {{
                        text-align: right;
                    }}
                    .bool_col {{
                        text-align: center;
                    }}
                    .other_col {{
                        text-align: left;
                    }}
                </style>
            </head>
            <body>
                <h2>{self._title}</h2>
                {str(self)}
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


_tabulator_typemeta = {
    'bool': {
        'align': 'center',
        'formatter': 'tickCross',
        'formatterParams': {'crossElement': False},
        'titleFormatter': 'textarea',
        'filter': 'tickCross',
        'headerFilterParams': {'tristate': True},
        'sorter': 'string',
    },
    'int': {
        'align': 'right',
        'formatter': 'plaintext',
        'titleFormatter': 'textarea',
        'filter': False,
        'sorter': 'number',
    },
    'real': {
        'align': 'right',
        'formatter': 'plaintext',
        'titleFormatter': 'textarea',
        'filter': False,
        'sorter': 'number',
    },
    'other': {
        'align': 'left',
        'formatter': 'textarea',
        'titleFormatter': 'textarea',
        'filter': 'input',
        'sorter': 'string',
    }
}


class TabulatorJSBuilder(TableBuilder):
    allowed_col_meta = TableBuilder.allowed_col_meta.union({
        'filter',
        'header_align',
        'sorter',
        'formatter'
    })

    def __init__(self, rows, layout=None, height=None, html_id='tabul-table', title='',
                 display_in_notebook=True, show_browser=True, outfile='tabulator_table.html',
                 filter=True, sort=True, **kwargs):
        super().__init__(rows, **kwargs)
        self._html_id = html_id if html_id.startswith('#') else '#' + html_id
        self._filter = filter
        self._sort = sort

        self._table_meta = {
            'layout': layout,
            'height': height,
        }
        self._title = title
        self._display_in_notebook = display_in_notebook
        self._show_browser = show_browser
        self._outfile = outfile

        # HTML sizing (in pixels)
        resize_handle_w = 0 # 5
        padding = 0 # 4
        title_right_padding = 0 # 25

        self._cell_padding = 2 * (resize_handle_w + padding)
        self._header_padding = self._cell_padding + title_right_padding
        self.font_size = 14

    def get_max_total_col_width(self, winfo):
        return self.max_width

    def _cell_width(self, cell):
        # special handling for bool cells
        if isinstance(cell, bool):
            return 5
        return len(cell)

    def _format_cell(self, meta, cell):
        if isinstance(cell, bool):
            return cell
        return meta['format'].format(cell)

    def _get_total_width(self):
        return sum(self._get_column_widths())

    def _stringified_row_iter(self):
        self._set_widths()
        for row in self._get_formatted_rows():
            yield row

    def _update_col_meta_from_rows(self):
        for i, col_type in enumerate(self._get_cell_types()):
            meta = _tabulator_typemeta[col_type].copy()
            meta['format'] = self._default_formats[col_type]
            meta['header_align'] = meta['align']
            meta['max_width'] = None
            meta['col_type'] = col_type

            if notebook:
                meta['headerSort'] = False

            if i in self._column_meta:
                meta.update(self._column_meta[i])

            self._column_meta[i] = meta

    def _header_cell_width(self, content_width):
        # content_width assumed in pixels
        return content_width + self._header_padding

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
        for i, meta in enumerate(self.sorted_meta()):
            cmeta = {
                'field': f'c{i}',
                'title': meta['header'],
                'hozAlign': meta['align'],
                'headerHozAlign': meta['header_align'],
                'headerFilter': meta['filter'] if self._filter else False,
                'sorter': meta['sorter'] if self._sort else False,
                'headerSort': self._sort,
                'formatter': meta['formatter'],  # plaintext, textarea, html, money, image, link,
                                                 # tickCross, traffic, star, progress, color,
                                                 # buttonTick, buttonCross,
                'formatterParams': meta.get('formatterParams', None),
                'titleFormatter': meta.get('titleFormatter', None),
                'titleFormatterParams': meta.get('titleFormatterParams', None),
                'editor': meta.get('editor', None),
                'editorParams': meta.get('editorParams', None),
                'headerFilterParams': meta.get('headerFilterParams', None),
                'widthGrow': meta.get('widthGrow'),
                'widthShrink': meta.get('widthShrink'),
            }

            if meta['max_width'] is not None:
                cmeta['initialMaxWidth'] = meta['max_width'] * self.font_size

            cols.append(cmeta)

        # for big tables, use virtual DOM for speed (setting height activates it)
        if len(self._raw_rows) > 20 and self._table_meta['height'] is None:
            self._table_meta['height'] = 600

        self._table_meta['data'] = rows
        self._table_meta['columns'] = cols

        if self._table_meta['layout'] is None:
            if self.needs_wrap():
                self._table_meta['layout'] = 'fitColumns'
            else:
                self._table_meta['layout'] = 'fitDataTable'

        return {
            'id': self._html_id,
            'title': self._title,
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


def to_table(rows, tablefmt='text', **options):
    _table_types = {
        'text': TextTableBuilder,
        'github': GithubTableBuilder,
        'rst': RSTTableBuilder,
        'tabulator': TabulatorJSBuilder,
        'html': HTMLTableBuilder,
    }

    try:
        table_class = _table_types[tablefmt]
    except Exception:
        raise KeyError(f"'{tablefmt}' is not a valid type choice for to_table.  Valid choices are: "
                       f"{sorted(_table_types)}.")

    return table_class(rows, **options)


if __name__ == '__main__':
    import sys

    coltypes = [
        ('str', {'maxsize': 20}),
        ('real', {'low': -1e10, 'high': 1e10}),
        ('bool', {}),
        ('int', {'low': -99, 'high': 2500}),
        ('str', {'maxsize': 80}),
        ('str', {'maxsize': 150}),
    ]

    try:
        formats = [sys.argv[1]]
    except Exception:
        formats = ['rst', 'github', 'text', 'html', 'tabulator']

    from openmdao.utils.tests.test_tables import random_table
    for fmt in formats:
        tab = random_table(tablefmt=fmt, coltypes=coltypes, nrows=55)
        if fmt == 'html':
            tab.update_column_meta(3, header_style={'text-align': 'right'})
        tab.max_width = 200
        tab.display()
