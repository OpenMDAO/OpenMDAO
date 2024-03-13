"""
Table building classes.
"""

import sys
import os
import json
import textwrap
from collections import namedtuple
from itertools import zip_longest, chain
from html import escape
from dataclasses import dataclass
from numbers import Number, Integral
from openmdao.utils.notebook_utils import notebook, display, HTML, IFrame, colab
from openmdao.utils.om_warnings import issue_warning


_align2symbol = {
    'center': '^',
    'right': '>',
    'left': '<'
}

_default_align = {
    'int': 'right',
    'real': 'right',
    'bool': 'center',
    'other': 'left',
}

_big_table_height = 600
_row_height_px = 45  # use to size the IFrame in a notebook


def _num_cols(rows):
    ncols = None
    for row in rows:
        i = 0
        for _ in row:
            i += 1
        if ncols is None or ncols < i:
            ncols = i
    return 0 if ncols is None else ncols


class TableBuilder(object):
    """
    Base class for all table builders.

    Parameters
    ----------
    rows : iter of iters
        Data used to fill table cells.
    headers : iter of str, 'keys', or None
        If not None, header strings for all columns.  Size must match
        number of columns in each row of data in rows.  A value of 'keys' means that rows is
        either a dict or a list of dicts and the keys of that/those dict(s) should be used
        as headers.
    column_meta : iter of dict or None
        If not None, contains a dict for each table column and the dict contains values
        of metadata for that column.
    precision : int or str
        Precision applied to all columns of real numbers.  May be overridden by column
        metadata for a specific column. Defaults to 4.
    missing_val : str
        A value that will replace any cell data having a value of None.  Defaults to ''.
    max_width : int or None
        If not None, specifies the maximum width, in characters, allowed for the table.
        If the table cannot meet the max width requirement by resizing columns, the max_width
        is ignored.

    Attributes
    ----------
    missing_val : str
        String to replace any data values of None.
    max_width : int or None
        If not None, specifies the maximum allowable width, in characters, of the table.
    _ncols : int
        Number of columns.
    _raw_rows : iter of iters
        Table row data from the caller, possibly converted to list of lists if caller passed in
        dict or iter of dicts.
    _rows : list of lists
        Table row data after initial formatting.
    _column_meta : dict
        Metadata for each column, keyed by column index, starting at 0.
    _data_widths : list of int
        Width of widest data cell in each column.
    _header_widths : list of int
        Width of each column header.
    _default_formats : dict
        Dict mapping each column type to its default format string.
    """

    allowed_col_meta = {'header', 'align', 'header_align', 'width', 'format',
                        'max_width', 'min_width', 'fixed_width'}

    def __init__(self, rows, headers=None, column_meta=None, precision=4, missing_val='',
                 max_width=None):
        """
        Initialize all attributes.
        """
        if headers in ('keys', 'firstrow'):
            rows, headers = self._to_rows(rows, headers)
        elif isinstance(headers, str):
            raise RuntimeError(f"If 'headers' is a string, it must be one of ['keys', 'firstrow'].")

        self._raw_rows = []
        for row in rows:
            self._raw_rows.append(list(row))
        self._ncols = _num_cols(self._raw_rows)
        self._rows = None  # rows after initial formatting (total width not set)
        self._column_meta = {}
        self._data_widths = None  # width of data in each cell before a uniform column width is set
        self._header_widths = None  # width of headers before a uniform column width is set
        self.missing_val = missing_val
        self.max_width = max_width

        # these are the default format strings for the first formatting stage,
        # before the column width is set
        self._default_formats = {
            'real': f"{{:.{precision}}}",
            'int': "{}",
            'bool': "{}",
            'other': "{}",
        }

        # for convenience, allow a user to specify header strings without putting them
        # inside a metadata dict
        if headers is not None:
            headers = list(headers)
            hlen = len(headers)
            if hlen != self._ncols:
                raise RuntimeError("Number of headers and number of data columns must match, but "
                                   f"{hlen} != {self._ncols}.")

            for i, h in enumerate(headers):
                if not isinstance(h, str):
                    h = str(h)
                self.update_column_meta(i, header=h)

        if column_meta is not None:
            column_meta = list(column_meta)
            clen = len(column_meta)
            if clen != self._ncols:
                raise RuntimeError("Number of column metadata dicts and number of data columns "
                                   f"must match, but {clen} != {self._ncols}.")

            for i, meta in enumerate(column_meta):
                self.update_column_meta(i, **meta)

        if headers is not None and column_meta is not None and hlen != clen:
            raise RuntimeError("Number of headers and number of column metadata dicts must match "
                               f"if both are provided, but {hlen} != {clen}.")

    def display(self, outfile=None):
        """
        Display this table.

        Parameters
        ----------
        outfile : str or None
            If None, print this table to stdout, else write it to the named file.
        """
        raise NotImplementedError("The display method is not defined for class "
                                  f"'{type(self).__name__}'.")

    def _to_rows(self, rows, headers):
        """
        Convert dict or iter of dicts into expected row and header data format.

        Parameters
        ----------
        rows : dict or iter of dicts
            Table cell and header data.
        headers : str
            Either 'keys' or 'firstrow'.

        Returns
        -------
        list of lists
            Table data cells.
        list of str
            Table headers.
        """
        new_rows = []

        if headers == 'firstrow':

            for i, row in enumerate(rows):
                if i == 0:
                    headers = list(row)
                else:
                    new_rows.append(list(row))

            return new_rows, headers

        headers = []

        if isinstance(rows, dict):
            # each value is a column, so re-arrange to be row major, and allow columns of
            # unequal length (to be compatible with tabulate).
            headers = list(rows.keys())

            # First, get max column length
            maxcol = 0
            for col in rows.values():
                clen = len(list(col))
                if clen > maxcol:
                    maxcol = clen

            for col in rows.values():
                if not new_rows:
                    new_rows = [[] for i in range(maxcol)]

                for row, cell in zip_longest(new_rows, col, fillvalue=''):
                    row.append(cell)
        else:
            # handle case where rows is an iter of dicts
            try:
                for row in rows:
                    headers = list(row.keys())
                    break
            except AttributeError:
                raise AttributeError("Since headers == 'keys', table builder was expecting an iter "
                                     f"of dicts, but got an iter of {type(row).__name__}.")
            for row in rows:
                new_rows.append(list(row.values()))

        return new_rows, headers

    def sorted_meta(self):
        """
        Return the column metadata sorted in ascending column order.

        Returns
        -------
        list
            The sorted column metadata.
        """
        return [m for _, m in sorted(self._column_meta.items(), key=lambda x: x[0])]

    def _get_formatted_rows(self):
        """
        Get table rows with cells after initial formatting, before final width setting.

        Returns
        -------
        list
            The list of table rows where each cell has been formatted.
        """
        if self._rows is None:
            self._update_col_meta_from_rows()
            sorted_cols = self.sorted_meta()
            self._rows = []
            unequal = False
            maxcols = 0
            for row in self._raw_rows:
                if self.missing_val is not None:
                    row = [self.missing_val if v is None else v for v in row]

                if self._rows and len(row) != len(self._rows[-1]):
                    unequal = True

                if len(row) > maxcols:
                    maxcols = len(row)

                self._rows.append([self._format_cell(meta, cell)
                                   for meta, cell in zip(sorted_cols, row)])

            if unequal:
                # make all rows have same number of cells
                for row in self._rows:
                    if len(row) < maxcols:
                        row.extend([self.missing_val] * (maxcols - len(row)))

        return self._rows

    def _cell_width(self, cell):
        """
        Return the width of the cell.

        Returns
        -------
        int
            The width of the cell.
        """
        return len(cell)

    def _get_cell_types(self):
        """
        Yield the type of cells each column contains.

        If columns have mixed types, yield 'other'.

        Yields
        ------
        str
            The type of the current column.
        """
        types = []
        for row in self._raw_rows:
            if not types:
                types = [set() for r in row] if row is not None else []

            for i, cell in enumerate(row):
                if isinstance(cell, Number):
                    if isinstance(cell, bool):
                        types[i].add('bool')
                    elif isinstance(cell, Integral):
                        types[i].add('int')
                    else:
                        types[i].add('real')
                elif cell is None:
                    pass  # don't add to types if cell is None
                else:
                    types[i].add('other')

        for tset in types:
            if len(tset) > 1 or len(tset) == 0:
                yield 'other'  # mixed type column (or all Nones), so just use "{}" format
            else:
                yield tset.pop()

    def update_column_meta(self, col_idx, **options):
        r"""
        Update metadata for the column at the specified index (starting at index 0).

        Parameters
        ----------
        col_idx : int
            The index of the column, starting at 0.
        **options : dict
            The metadata dict will be updated with these options.
        """
        if col_idx < 0:  # allow negative indices
            col_idx = self._ncols + col_idx

        if col_idx < 0 or col_idx >= self._ncols:
            raise IndexError(f"Index '{col_idx}' is not a valid table column index for a table with"
                             f" {self._ncols} columns.")

        if col_idx not in self._column_meta:
            self._column_meta[col_idx] = {}

        meta = self._column_meta[col_idx]
        for name, val in options.items():
            meta[name] = val

    def _set_widths(self, force_set_max=False):
        """
        Set data and header widths to their final values.

        Parameters
        ----------
        force_set_max : bool
            If True, compute the max column widths even if the table max_width limit is violated.
        """
        if self._data_widths is not None:
            return  # widths already computed

        rows = self._get_formatted_rows()

        if self._rows and len(self._rows[0]) != self._ncols:
            raise RuntimeError(f"Number of row entries ({len(self._rows[0])}) must match number of "
                               f"columns ({self._ncols}) in TableBuilder.")

        self._data_widths = [0] * self._ncols
        self._header_widths = [0] * self._ncols

        for row in rows:
            for i, cell in enumerate(row):
                wid = self._cell_width(cell)
                if wid > self._data_widths[i]:
                    self._data_widths[i] = wid

        # set widths and min widths
        for i, meta in enumerate(self.sorted_meta()):
            self._header_widths[i] = 0 if meta['header'] is None else len(meta['header'])

        self._set_min_widths()
        self._set_max_column_widths(force_set_max)

    def _format_cell(self, meta, cell):
        """
        Apply the initial formatting (before final width setting) to a cell.

        Parameters
        ----------
        meta : dict
            Column metadata dict.
        cell : object
            Data contained in a table cell.
        """
        if cell is None:
            return ''
        try:
            return meta['format'].format(cell)
        except Exception:
            return f'{cell}'

    def _update_col_meta_from_rows(self):
        """
        Fill in missing column metadata based on the data types of column contents.
        """
        for i, col_type in enumerate(self._get_cell_types()):
            align = _default_align[col_type]

            meta = {
                'header': None,
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

        Returns
        -------
        bool
            Indicates whether or not this table requires word wrapping to fit the maximum width.
        """
        needs_wrap = self.max_width is not None and self.max_width < self._get_total_width()
        if not needs_wrap:
            for meta in self._column_meta.values():
                if meta['max_width'] is not None:
                    return True
        return needs_wrap

    def _set_min_widths(self):
        """
        Set minimum width for columns.
        """
        for meta, wcell in zip(self.sorted_meta(), self._data_widths):
            header = meta['header']

            # check if header is splittable into words, and if so, allow for min_width using a
            # split header if min_width will be > min data width.
            if header is not None and header.strip():
                longest_part = max(len(word) for word in header.strip().split())
            else:
                longest_part = 0

            if meta['col_type'] == 'other':  # strings
                meta['min_width'] = max(10, longest_part)
            else:
                meta['min_width'] = max(wcell, longest_part)

    def _set_max_column_widths(self, force_set_max=False):
        """
        Set the maximum allowable column widths based on the table max_width.

        Parameters
        ----------
        force_set_max : bool
            If True, compute the max column widths even if the table max_width limit is violated.
        """
        # check for case where total table width is specified and we have to set max_width on
        # column(s) as a result
        if self.max_width is not None and self.max_width < self._get_total_width():
            winfo = [[i, w, meta['min_width']]
                     for (i, meta), w in zip(enumerate(self.sorted_meta()),
                                             self._get_column_widths())
                     if not meta.get('fixed_width')]

            fixed_width = self._get_total_width() - sum([w for _, w, _ in winfo])
            allowed_width = self.max_width - fixed_width

            # subtract 1 from the widest column until we meet the total max_width requirement,
            # or get as close as we can without violating a minimum allowed width.
            while sum([w for _, w, _ in winfo]) > allowed_width:
                for info in sorted(winfo, key=lambda x: x[1], reverse=True):
                    _, width, min_width = info
                    if width - 1 >= min_width:
                        info[1] -= 1
                        break
                else:
                    break

            if force_set_max or sum([w for _, w, _ in winfo]) <= allowed_width:
                for i, w, _ in winfo:
                    self._column_meta[i]['max_width'] = w

    def _get_column_widths(self):
        """
        Return the width of each column.

        Returns
        -------
        list
            The column widths.
        """
        return [max(wd, wh) for wd, wh in zip(self._data_widths, self._header_widths)]

    def write(self, outfile=None):
        """
        Write this table to the given output file.

        Parameters
        ----------
        outfile : str or None
            The output file.  If None, assume table should be sent to stdout.

        Returns
        -------
        str or None
            The output file name or None if written to stdout.
        """
        if outfile is None:
            sys.stdout.write(str(self))
        else:
            with open(outfile, 'w') as f:
                f.write(str(self))
            return outfile


@dataclass(frozen=True)
class Line:
    """
    Information about a line in the table.

    Parameters
    ----------
    left : str
        Left border string.
    sep : str
        Column separator.
    right : str
        Right border string.
    hline : str
        Horizontal line string.

    Attributes
    ----------
    left : str
        Left border string.
    sep : str
        Column separator.
    right : str
        Right border string.
    hline : str
        Horizontal line string.
    """

    left: str = ''
    sep: str = ''
    right: str = ''
    hline: str = ''

    def get_border_line(self, widths):
        """
        Return a border line given the column widths.

        Parameters
        ----------
        widths : list of int
            Column widths.

        Returns
        -------
        str
            The border line.
        """
        line = self.sep.join([(self.hline * w)[:w] for w in widths])
        return ''.join((self.left, line, self.right))

    def get_data_line(self, cells):
        """
        Return a table line containing the given cells.

        Parameters
        ----------
        cells : list of str
            Contents of table columns for the current row.

        Returns
        -------
        str
            The table line containing the cells, with separators and left and right borders.
        """
        return ''.join((self.left, self.sep.join(cells), self.right))


class TextTableBuilder(TableBuilder):
    r"""
    Base class for all text-based table builders.

    Parameters
    ----------
    rows : iter of iters
        Data used to fill table cells.
    top_border : Line
        Top border info.
    header_bottom_border : Line
        Header bottom border info.
    bottom_border : Line
        Bottom border info.
    header_line : Line
        Header line info.
    data_row_line : Line
        Data row line info.
    row_separator : Line or None
        If not None, info for lines between data rows.
    **kwargs : dict
        Keyword args for the base class.

    Attributes
    ----------
    top_border : Line
        Top border info.
    header_bottom_border : Line
        Header bottom border info.
    bottom_border : Line
        Bottom border info.
    header_line : Line
        Header line info.
    data_row_line : Line
        Data row line info.
    row_separator : Line or None
        If not None, info for lines between data rows.
    """

    def __init__(self, rows,
                 top_border=Line('| ', '---', ' |', '-'),
                 header_bottom_border=Line('| ', ' | ', ' |', '-'),
                 bottom_border=Line('| ', '---', ' |', '-'),
                 header_line=Line('| ', ' | ', ' |'),
                 data_row_line=Line('| ', ' | ', ' |'),
                 row_separator=None,
                 **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(rows, **kwargs)

        self.top_border = top_border
        self.header_line = header_line
        self.header_bottom_border = header_bottom_border
        self.bottom_border = bottom_border
        self.data_row_line = data_row_line
        self.row_separator = row_separator

    def _get_total_width(self):
        """
        Return the total table width in characters.
        """
        tot = sum(self._get_column_widths())
        tot += (len(self._column_meta) - 1) * len(self.top_border.sep)
        tot += len(self.top_border.left) + len(self.top_border.right)
        return tot

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
            sym = _align2symbol[align]
        except KeyError:
            raise KeyError(f"Expected one of ['left', 'right', 'center'] for '{align_name}' "
                           f"metadata, but got '{align}'.")
        return f"{cell:{sym}{width}}"

    def get_lengthened_columns(self, sorted_cols, header_cells, widths):
        """
        Yield as many rows of cells as needed to allow for multi-line cells due to word wrapping.

        Parameters
        ----------
        sorted_cols : list of dict
            List of sorted column metadata.
        header_cells : list of str
            List of header strings.
        widths : list of int
            Column widths.

        Yields
        ------
        list
            Each row after expanding due to word wrapping.
        """
        cell_lists = []
        for meta, cell, wid in zip(sorted_cols, header_cells, widths):
            maxwid = meta['max_width']
            if maxwid is not None and maxwid < len(cell):
                lines = textwrap.wrap(cell, maxwid)
                wid = maxwid
            elif '\n' in cell:
                lines = cell.split('\n')
            else:
                cell_lists.append([cell])
                continue

            sym = _align2symbol[meta['header_align']]
            if sym == '^':  # center
                lines = [line.strip() for line in lines]
            # ensure all cells have same width in this column
            cell_lists.append([f"{line:{sym}{wid}}" for line in lines])

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
        Yield one or more rows of header cells.

        Yields
        ------
        list
            Each header row after expanding due to word wrapping.
        """
        header_cells = [None] * self._ncols
        self._set_widths()

        widths = self._get_column_widths()

        sorted_cols = self.sorted_meta()

        for i, meta in enumerate(sorted_cols):
            header = meta['header']
            if header is not None:
                header_cells[i] = self._get_fixed_width_cell(meta, header, widths[i],
                                                             'header_align')

        yield from self.get_lengthened_columns(sorted_cols, header_cells, widths)

    def _stringified_row_iter(self):
        """
        Yield rows of data cells, allowing for multi-line rows due to word wrapping.

        The cells are all strings with the same width as their column.

        Yields
        ------
        list
            List of cells for the current row.
        """
        self._set_widths()
        widths = self._get_column_widths()
        row_cells = [None] * len(self._column_meta)

        sorted_cols = self.sorted_meta()

        for row in self._get_formatted_rows():
            for i, meta in enumerate(sorted_cols):
                row_cells[i] = self._get_fixed_width_cell(meta, row[i], widths[i], 'align')

            cell_lists = []
            for meta, cell, wid in zip(sorted_cols, row_cells, widths):
                maxwid = meta['max_width']
                if maxwid is not None and maxwid < len(cell):
                    lines = textwrap.wrap(cell, maxwid)
                    if not lines:  # empty cell or whitespace
                        lines = ['']
                    wid = maxwid
                elif '\n' in cell:
                    lines = cell.split('\n')
                else:
                    cell_lists.append([cell])
                    continue

                sym = _align2symbol[meta['align']]
                if sym == '^':  # center
                    lines = [line.strip() for line in lines]
                # ensure all cells have same width in this column
                cell_lists.append([f"{line:{sym}{wid}}" for line in lines])

            # now find longest column
            maxlen = max([len(lst) for lst in cell_lists])
            cell_list_group = []
            for r in range(maxlen):
                cells = []
                for clist in cell_lists:
                    if len(clist) > r:
                        cells.append(clist[r])
                    else:
                        w = len(clist[0]) if clist else 0
                        cells.append(' ' * w)
                cell_list_group.append(cells)

            yield cell_list_group

    def get_top_border(self, widths):
        """
        Return the top border string for this table.

        Parameters
        ----------
        widths : list of int
            Column widths.

        Returns
        -------
        str
            The top border string.
        """
        return self.top_border.get_border_line(widths)

    def get_header_bottom_border(self, widths):
        """
        Return the header bottom border string for this table.

        Parameters
        ----------
        widths : list of int
            Column widths.

        Returns
        -------
        str
            The header bottom border string.
        """
        return self.header_bottom_border.get_border_line(widths)

    def get_bottom_border(self, widths):
        """
        Return the bottom border string for this table.

        Parameters
        ----------
        widths : list of int
            List of column widths.

        Returns
        -------
        str
            The bottom border string.
        """
        return self.bottom_border.get_border_line(widths)

    def __str__(self):
        """
        Return this table to a string.

        Returns
        -------
        str
            This table as a string.
        """
        header_lines = []
        data_lines = []
        sep = self.data_row_line.sep
        row_cells = None
        row_list = list(self._stringified_row_iter())
        if row_list:
            widths = [len(c) for c in row_list[0][0]]
        else:
            widths = []

        for i, cell_list_group in enumerate(row_list):
            if i > 0 and self.row_separator:
                data_lines.append(self.row_separator.get_border_line(widths))
            for row_cells in cell_list_group:
                data_lines.append(self.data_row_line.get_data_line(row_cells))

        if self.bottom_border:
            data_lines.append(self.get_bottom_border(widths))

        if row_cells is not None and self.top_border:
            header_lines.append(self.get_top_border(widths))

        if sum(self._header_widths) > 0:
            for header_cells in self._stringified_header_iter():
                header_lines.append(self.header_line.get_data_line(header_cells))

            if self.header_bottom_border:
                header_lines.append(self.get_header_bottom_border(widths))

        return '\n'.join(chain(header_lines, data_lines))

    def display(self, outfile=None):
        """
        Display this table.

        Parameters
        ----------
        outfile : str or None
            If None, print this table to stdout, else write it to the named file.

        Returns
        -------
        str or None
            The name of the file where the table was written, or None if it was written to stdout.
        """
        if outfile is None:
            sys.stdout.write(str(self))
            sys.stdout.write('\n')
            return

        with open(outfile, 'w') as f:
            f.write(str(self))
            f.write('\n')

        return outfile


class GithubTableBuilder(TextTableBuilder):
    r"""
    Class that generates a table in Github markdown format.

    Parameters
    ----------
    rows : iter of iters
        Data used to fill table cells.
    **kwargs : dict
        Keyword args for the base class.
    """

    def __init__(self, rows, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(rows,
                         top_border=None,
                         header_bottom_border=Line('| ', ' | ', ' |', '-'),
                         bottom_border=None,
                         header_line=Line('| ', ' | ', ' |'),
                         data_row_line=Line('| ', ' | ', ' |'), **kwargs)

    def _repr_markdown_(self):
        return str(self)

    def get_header_bottom_border(self, widths):
        """
        Return the header bottom border string.

        Parameters
        ----------
        widths : list of int
            Column widths.

        Returns
        -------
        str
            The header bottom border string.
        """
        parts = []
        for i, width in enumerate(widths):
            meta = self._column_meta[i]
            align = meta['align']
            left = right = ''
            size = width
            if align == 'left':
                left = ':'
                size -= 1
            elif align == 'right':
                right = ':'
                size -= 1
            else:  # center
                left = right = ':'
                size -= 2
            parts.append(left + (self.header_bottom_border.hline * size) + right)

        return ''.join((self.header_bottom_border.left, self.header_bottom_border.sep.join(parts),
                        self.header_bottom_border.right))

    def needs_wrap(self):
        """
        Return False.

        This table does not allow word wrapping.

        Returns
        -------
        False
            No wrapping will be used.
        """
        return False  # github tables seem to have no support for text wrapping in columns


def _to_inline_style(dct):
    """
    For the given dict, return an inline HTML style string.

    Parameters
    ----------
    dct : dict
        The dict containing style parameters and their values.

    Returns
    -------
    str
        The inline style string.
    """
    parts = ' '.join([f"{name}: {val};" for name, val in dct.items()])
    if parts:
        return f' style="{parts}"'
    return ''


_default_html_table = 'table.html'


class HTMLTableBuilder(TableBuilder):
    r"""
    Class that generates a table in plain HTML format.

    Parameters
    ----------
    rows : iter of iters
        Data used to fill table cells.
    html_id : str or None
        If not None, the HTML id for the <table> block.
    title : str or None
        If not None, the title appearing above the table on the web page.
    center : bool
        If True, center the table on the page.
    style : dict or None
        If not None, a dict mapping table style parameters to their values.
    safe : bool
        If True (the default), html escape text in the cells.
    **kwargs : dict
        Keyword args for the base class.

    Attributes
    ----------
    _html_id : str or None
        If not None, this is the html id of the table block.
    _title : str or None
        If not None, this is the title that will appear on the web page above the table.
    _safe : bool
        If True (the default), html escape text in the cells.
    _data_style : dict
        Contains style metadata for <td> blocks.
    _header_style : dict
        Contains style metadata for <th> blocks.
    _style : dict or None
        Contains style metadata for the table.
    """

    allowed_col_meta = TableBuilder.allowed_col_meta.union({
        'header_style',
        'style',
    })

    def __init__(self, rows, html_id=None, title='', center=False, style=None, safe=True, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(rows, **kwargs)
        self._data_style = {
            'border': '1px solid #999',
            'border-collapse': 'collapse',
            'padding': '5px',
        }
        self._header_style = {
            'border': '1px solid #999',
            'border-collapse': 'collapse',
            'padding': '5px',
            'background-color': '#E9E9E9',
        }
        tstyle = {
            'border': '1px solid #999',
            'border-collapse': 'collapse',
        }
        if center:
            tstyle['margin'] = 'auto'

        if style is not None:
            tstyle.update(style)

        self._style = tstyle
        self._html_id = html_id
        self._title = title
        self._safe = safe

    def _repr_html_(self):
        return str(self)

    def _stringified_row_iter(self):
        """
        Yield rows of string data cells whose width matches their column width.

        Yields
        ------
        list
            List of cells for the current row.
        """
        for row in self._get_formatted_rows():
            yield row

    def _assemble(self):
        rlines = []

        def _escape(s):
            if s is None:
                return ''
            elif self._safe:
                return escape(s)
            return s

        for irow, row_cells in enumerate(self._stringified_row_iter()):
            row_style = {'background-color': '#F3F3F3' if irow % 2 else 'ghostwhite'}
            parts = [f"   <tr{_to_inline_style(row_style)}>"]
            for cell, meta in zip(row_cells, self.sorted_meta()):
                style = self._data_style.copy()
                style['text-align'] = meta['align']
                parts.append(f'<td{_to_inline_style(style)}>{_escape(cell)}</td>')
            parts.append("</tr>")
            rlines.append(''.join(parts))

        lines = []
        html_id = f' id="{self._html_id}"' if self._html_id else ''
        lines.append(f'<table{html_id}{_to_inline_style(self._style)}>')

        # we do the header out-of-order with the rows here because we need to
        # query the rows first to determine column data type to fill in missing
        # parts of the column metadata.

        # First, make sure we actually have a header
        has_header = False
        for meta in self.sorted_meta():
            if meta['header'] is not None:
                has_header = True
                break

        if has_header:
            hparts = ["    <tr>"]
            for meta in self.sorted_meta():
                style = self._header_style.copy()
                style['text-align'] = meta['header_align']
                if 'header_style' in meta and meta['header_style']:
                    style.update(meta['header_style'])
                header_style = _to_inline_style(style)
                hparts.append(f"<th{header_style}>{_escape(meta['header'])}</th>")
            hparts.append("</tr>")
            lines.append(''.join(hparts))

        lines.extend(rlines)  # now add the rows back in

        lines.append("</table>")

        return '\n'.join(lines)

    def __str__(self):
        """
        Return a string representation of the Table.
        """
        return textwrap.dedent("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <style>
                    h2 {{
                        text-align: center;
                    }}
                </style>
            </head>
            <body>
                <h2>{}</h2>
                {}
            </body>
            </html>
            """).format(self._title, textwrap.indent(self._assemble(), '    '))

    def write(self, outfile=_default_html_table):
        """
        Write this table to the given output file.

        If outfile is not given, write this table to 'table.html' in the current
        directory.

        Parameters
        ----------
        outfile : str
            The output file where the HTML will be written.

        Returns
        -------
        str
            The name of the file where the table was written.
        """
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(str(self))
        return outfile

    def display(self, outfile=None):
        """
        Display this table, either in a notebook or a browser.

        This table will also be written to outfile.  If outfile is not supplied, the table
        will be written to 'table.html' in the current directory.

        Parameters
        ----------
        outfile : str or None
            Table will be written to this file.
            If None, write this table to '{_default_html_table}'.

        Returns
        -------
        str
            The name of the file where the table was written.
        """
        if outfile is None:
            outfile = _default_html_table

        self.write(outfile)

        if notebook:
            if not colab:
                height = min(_big_table_height, len(self._rows) * _row_height_px)
                display(IFrame(src=outfile, width="100%", height=height))
            else:
                display(HTML(outfile))
        else:
            # open it up in the browser
            from openmdao.utils.webview import webview
            webview(outfile)

        return outfile


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

_default_tabulator_file = 'tabulator_table.html'


class TabulatorJSBuilder(TableBuilder):
    r"""
    Class that generates an interactive table using Tabulator.js.

    Parameters
    ----------
    rows : iter of iters
        Data used to fill table cells.
    html_id : str or None
        If not None, the HTML id for the <table> block.
    title : str or None
        If not None, the title appearing above the table on the web page.
    filter : bool
        If True, include filter fields in the column headers where it makes sense.
    sort : bool
        If True, add sorting to column headers.
    center : bool
        If True, center the table on the page.
    table_meta : dict or None
        If not None, a dict of Tabulator.js metadata names mapped to their values.
    **kwargs : dict
        Keyword args for the base class.

    Attributes
    ----------
    _html_id : str or None
        If not None, this is the html id of the table block.
    _title : str or None
        If not None, this is the title that will appear on the web page above the table.
    _filter : bool
        If True, include filter fields in the column headers where it makes sense.
    _sort : bool
        If True, add sorting to column headers.
    _center : bool
        If True, center the table on the page.
    _table_meta : dict
        Metadata for the table.
    font_size : int
        The font size used by the table.
    """

    allowed_col_meta = TableBuilder.allowed_col_meta.union({
        'filter',
        'header_align',
        'sorter',
        'formatter',
        'formatterParams',
        'labelField',
        'target',
    })

    def __init__(self, rows, html_id='tabul-table', title='', filter=True, sort=True, center=False,
                 table_meta=None, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(rows, **kwargs)
        self._html_id = html_id if html_id.startswith('#') else '#' + html_id
        self._title = title
        self._filter = filter
        self._sort = sort
        self._center = center

        self._table_meta = {
            'autoResize': True,
        }

        if table_meta is not None:
            self._table_meta.update(table_meta)

        self.font_size = 14

    def _cell_width(self, cell):
        """
        Return given cell's width in characters.

        This table's bool cells are left as bools instead of converted to strings in order to
        make the Tabulator.js table filtering work properly for bool columns, so this method
        will return a width for bool cells.

        Parameters
        ----------
        cell : str or bool
            The contents of the current table cell.

        Returns
        -------
        int
            The width of the cell.
        """
        # special handling for bool cells
        if isinstance(cell, bool):
            return 5
        return len(cell)

    def _format_cell(self, meta, cell):
        """
        Return the string formatted form of the cell, but leave bool cells unmodified.

        Parameters
        ----------
        meta : dict
            The metadata for the current column.
        cell : str or bool
            The current table cell.

        Returns
        -------
        str or bool
            The formatted cell if it's a str, else the unmodified cell.
        """
        if isinstance(cell, bool):
            return cell
        return super()._format_cell(meta, cell)

    def _get_total_width(self):
        """
        Return the total table width in characters.

        Returns
        -------
        int
            The total table width.
        """
        return sum(self._get_column_widths())

    def _stringified_row_iter(self):
        """
        Yield rows of string data cells.

        Yields
        ------
        list
            List of cells for the current row.
        """
        self._set_widths(force_set_max=True)
        for row in self._get_formatted_rows():
            yield row

    def _update_col_meta_from_rows(self):
        """
        Fill in missing column metadata based on the data types of column contents.
        """
        for i, col_type in enumerate(self._get_cell_types()):
            meta = _tabulator_typemeta[col_type].copy()
            meta['format'] = self._default_formats[col_type]
            meta['header_align'] = meta['align']
            meta['max_width'] = None
            meta['col_type'] = col_type
            meta['header'] = None

            if i in self._column_meta:
                meta.update(self._column_meta[i])

            self._column_meta[i] = meta

    def _get_table_data(self):
        """
        Return table data in a format that can be converted to json for Tabulator.js to use.

        Returns
        -------
        dict
            A dict to be converted to json and provided to Tabulator.js.
        """
        rows = []
        idx = 1  # unique ID for use by Tabulator

        for row_cells in self._stringified_row_iter():
            dct = {'id': idx}
            for i, cell in enumerate(row_cells):
                dct[f'c{i}'] = cell
            rows.append(dct)
            idx += 1

        if 'layout' not in self._table_meta:
            if self.needs_wrap():
                self._table_meta['layout'] = 'fitColumns'
            else:
                self._table_meta['layout'] = 'fitDataTable'

        if self._table_meta['layout'] == 'fitColumns':
            self._setup_fit_columns_layout()

        cols = []
        for i, meta in enumerate(self.sorted_meta()):
            cmeta = {
                'field': f'c{i}',
                # can't eliminate headers from tabulator.js tables, so just set to ' ' if None.
                # Setting to '' results in the header containing '&nbsp' which doesn't look great.
                'title': ' ' if meta['header'] is None else meta['header'],
                'hozAlign': meta['align'],
                'headerHozAlign': meta['header_align'],
                'headerFilter': meta['filter'] if self._filter else False,
                'sorter': meta['sorter'] if self._sort else False,
                'headerSort': self._sort,
                'formatter': meta['formatter'],  # plaintext, textarea, html, money, image, link,
                                                 # tickCross, traffic, star, progress, color,
                                                 # buttonTick, buttonCross,
                'formatterParams': meta.get('formatterParams'),
                'titleFormatter': meta.get('titleFormatter'),
                'titleFormatterParams': meta.get('titleFormatterParams'),
                'editor': meta.get('editor'),
                'editorParams': meta.get('editorParams'),
                'headerFilterParams': meta.get('headerFilterParams'),
                'widthGrow': meta.get('widthGrow'),
                'widthShrink': meta.get('widthShrink'),
                'tooltip': meta.get('tooltip'),
                'visible': meta.get('visible', True),
            }

            width = meta.get('width')
            if width is not None:
                cmeta['width'] = width
            cols.append(cmeta)

        # for big tables, use virtual DOM for speed (setting height activates it)
        if idx - 1 > 60 and ('height' not in self._table_meta or
                             self._table_meta['height'] is None):
            self._table_meta['height'] = _big_table_height

        self._table_meta['data'] = rows
        self._table_meta['columns'] = cols

        return {
            'id': self._html_id,
            'title': self._title,
            'meta': self._table_meta,
        }

    def _setup_fit_columns_layout(self):
        smeta = self.sorted_meta()
        maxws = [m['max_width'] for m in smeta]
        if None in maxws:
            issue_warning("Can't use 'layout' of 'fitColumns' with columns that have no "
                          "max_width set. Switching to layout of 'fitDataTable'.")
            self._table_meta['layout'] = 'fitDataTable'
            return

        mx = sum(maxws)
        pcts = [int(w / mx * 100) for w in maxws]
        rempcts = 100 - sum(pcts)
        while rempcts > 0:
            for i in range(len(pcts)):
                if rempcts > 0:
                    pcts[i] += 1
                    rempcts -= 1
                else:
                    break

        for pct, meta in zip(pcts, smeta):
            meta['width'] = f"{pct}%"

    def write(self, outfile=_default_tabulator_file):
        """
        Write this table to the given output file.

        If outfile is not given, write this table to 'tabulator_table.html' in the current
        directory.

        Parameters
        ----------
        outfile : str or None
            If not None, the output file where the HTML will be written.

        Returns
        -------
        str
            The output file name.
        """
        outfile = os.path.relpath(outfile)

        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(str(self))

        return outfile

    def __str__(self):
        """
        Return a string representation of the Table.

        Returns
        -------
        str
            This table as a string.
        """
        import openmdao.visualization

        code_dir = os.path.dirname(openmdao.visualization.__file__)
        libs_dir = os.path.join(code_dir, 'common', 'libs')
        style_dir = os.path.join(code_dir, 'common', 'style')

        format_dct = {}
        if self._center:
            format_dct['table_div_style'] = 'class="center"'
        else:
            format_dct['table_div_style'] = ''

        with open(os.path.join(code_dir, 'tables', 'generic_table.template'), "r",
                  encoding='utf-8') as f:
            template = f.read()

        with open(os.path.join(libs_dir, 'tabulator.5.4.4.min.js'), "r", encoding='utf-8') as f:
            tabulator_src = f.read()

        format_dct['tabulator_src'] = tabulator_src

        with open(os.path.join(style_dir, 'tabulator.5.4.4.min.css'), "r", encoding='utf-8') as f:
            tabulator_style = f.read()

        format_dct['tabulator_style'] = tabulator_style

        jsontxt = json.dumps(self._get_table_data())
        format_dct['table_data'] = jsontxt

        return template.format(**format_dct)

    def display(self, outfile=None):
        """
        Display this table, either in a notebook or a browser.

        This table will also be written to outfile.  If outfile is not supplied, the table
        will be written to '{_default_tabulator_file}' in the current directory.

        Parameters
        ----------
        outfile : str or None
            If None, write this table to '{_default_tabulator_file}'.

        Returns
        -------
        str
            The name of the file where the table was written.
        """
        if outfile is None:
            outfile = _default_tabulator_file

        self.write(outfile)
        if notebook:
            if not colab:
                height = min(_big_table_height, len(self._rows) * _row_height_px)
                display(IFrame(src=outfile, width="100%", height=height))
            else:
                display(HTML(outfile))
        else:
            # open it up in the browser
            from openmdao.utils.webview import webview
            webview(outfile)

        return outfile


def generate_table(rows, tablefmt='text', **options):
    r"""
    Return the specified table builder class.

    Parameters
    ----------
    rows : iter of iters
        This specifies the cells in each table row.
    tablefmt : str
        This determines which table builder object will be returned.
    **options : dict
        Named arguments specific to a given table builder class.

    Returns
    -------
    TableBuilder
        A table builder object.
    """
    _text_formats = {
        'rst': {
            'top_border': Line('', '  ', '', '='),
            'header_line': Line('', '  ', ''),
            'header_bottom_border': Line('', '  ', '', '='),
            'bottom_border': Line('', '  ', '', '='),
            'data_row_line': Line('', '  ', '')
        },
        'grid': {
            'top_border': Line('+-', '-+-', '-+', '-'),
            'header_line': Line('| ', ' | ', ' |'),
            'header_bottom_border': Line('+=', '=+=', '=+', '='),
            'bottom_border': Line('+-', '-+-', '-+', '-'),
            'data_row_line': Line('| ', ' | ', ' |'),
            'row_separator': Line('+-', '-+-', '-+', '-')
        },
        'simple_grid': {
            'top_border': Line("┌─", "─┬─", "─┐", "─"),
            'header_line': Line("│ ", " │ ", " │"),
            'header_bottom_border': Line("├─", "─┼─", "─┤", "─"),
            'row_separator': Line("├─", "─┼─", "─┤", "─"),
            'bottom_border': Line("└─", "─┴─", "─┘", "─"),
            'data_row_line': Line("│ ", " │ ", " │"),
        },
        'heavy_grid': {
            'top_border': Line("┏━", "━┳━", "━┓", "━"),
            'header_line': Line("┃ ", " ┃ ", " ┃"),
            'header_bottom_border': Line("┣━", "━╋━", "━┫", "━"),
            'row_separator': Line("┣━", "━╋━", "━┫", "━"),
            'bottom_border': Line("┗━", "━┻━", "━┛", "━"),
            'data_row_line': Line("┃ ", " ┃ ", " ┃"),
        },
        'double_grid': {
            'top_border': Line("╔═", "═╦═", "═╗", "═"),
            'header_line': Line("║ ", " ║ ", " ║"),
            'header_bottom_border': Line("╠═", "═╬═", "═╣", "═"),
            'row_separator': Line("╠═", "═╬═", "═╣", "═"),
            'bottom_border': Line("╚═", "═╩═", "═╝", "═"),
            'data_row_line': Line("║ ", " ║ ", " ║"),
        },
        'box_grid': {
            'top_border': Line("╔═", "═╤═", "═╗", "═"),
            'header_line': Line("║ ", " ┊ ", " ║"),
            'header_bottom_border': Line("╠═", "═╪═", "═╣", "═"),
            'row_separator': Line("╟┈", "┈┿┈", "┈╢", "┈"),
            'bottom_border': Line("╚═", "═╧═", "═╝", "═"),
            'data_row_line': Line("║ ", " ┊ ", " ║"),
        },
    }

    for fmt in list(_text_formats):
        if 'grid' in fmt:
            dct = _text_formats[fmt].copy()
            dct['row_separator'] = None
            _text_formats[fmt.replace('grid', 'outline')] = dct

    _table_types = {
        'text': TextTableBuilder,
        'github': GithubTableBuilder,
        'tabulator': TabulatorJSBuilder,
        'html': HTMLTableBuilder,
    }

    if tablefmt in _text_formats:
        kwargs = _text_formats[tablefmt].copy()
        kwargs['rows'] = rows
        kwargs.update(**options)
        return TextTableBuilder(**kwargs)

    try:
        builder = _table_types[tablefmt]
    except Exception:
        raise KeyError(f"'{tablefmt}' is not a valid type choice for generate_table. Valid "
                       f"choices are: {sorted(_table_types)}.")

    return builder(rows, **options)
