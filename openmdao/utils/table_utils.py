
import sys
from io import StringIO

from numbers import Number, Integral


_a2sym = {'center': '^', 'right': '>', 'left': '<'}


class TableBuilder(object):
    allowed_col_meta = {'header', 'align', 'width', 'format'}

    def __init__(self, rows, headers=None, column_meta=None, precision=4):
        self._raw_rows = rows
        self._column_meta = {}
        self._widths = None
        self._rows = None

        # these attributes change in subclasses
        self.column_sep = ' | '
        self.top_border = ''
        self.header_bottom_border = '-'
        self.bottom_border = ''
        self.left_border = ''
        self.right_border = ''

        # these are the default format strings for the first formatting stage,
        # before the column width is set
        self._default_formats = {
            'real': "{" + f":.{precision}" + "}",
            'integral': "{}",
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
            self._rows = []
            for row in self._raw_rows:
                self._add_srow(row)

        return self._rows

    def _compute_widths(self):
        if self._widths is not None:
            return  # widths already computed

        rows = self._get_srows()

        if len(self._rows) != len(self._column_meta):
            raise RuntimeError("Number of row entries must match number of column infos in "
                               "TableBuilder.")

        self._widths = [0] * len(self._column_meta)

        for row in rows:
            for i, cell in enumerate(row):
                wid = len(cell)
                if wid > self._widths[i]:
                    self._widths[i] = wid

        for i, cinfo in sorted(self._column_meta.items(), key=lambda x: x[0]):
            wid = len(cinfo['header'])
            if wid > self._widths[i]:
                self._widths[i] = wid

    def _update_col_meta_from_row(self, row):
        for i, cell in enumerate(row):
            align = 'left'
            if isinstance(cell, Number):
                align = 'right'
                if isinstance(cell, Integral):
                    format = 'integral'
                else:
                    format = 'real'
            else:
                format = 'other'

            if i in self._column_meta:
                if 'format' not in self._column_meta[i]:
                    self._column_meta[i]['format'] = self._default_formats[format]
                if 'align' not in self._column_meta[i]:
                    self._column_meta[i]['align'] = align
            else:
                self._column_meta[i] = {'header': '', 'format': format, 'align': align}

    def _add_srow(self, row):
        if not self._rows:  # if this is the first row
            self._update_col_meta_from_row(row)

        cells = [self._column_meta[i]['format'].format(cell) for i, cell in enumerate(row)]

        if self._rows and len(cells) != len(self._rows[-1]):
            raise RuntimeError("Can't add rows of unequal length to TableBuilder.")

        self._rows.append(cells)

    def update_column_meta(self, col_idx, **options):
        if col_idx not in self._column_meta:
            self._column_meta[col_idx] = {'header': ''}
        meta = self._column_meta[col_idx]
        for name, val in options.items():
            if name not in self.allowed_col_meta:
                raise KeyError(f"'{name}' is not a valid column metadata key.")
            meta[name] = val

    def _get_fixed_width_cell(self, col_meta, cell, width):
        align = col_meta.get('align', 'left')
        try:
            sym = _a2sym[align]
        except KeyError:
            raise KeyError("Expected one of ['left', 'right', 'center'] for 'align' metadata, but "
                           f"got '{align}'.")
        return f"{cell:{sym}{width}}"

    def _get_stringified_headers(self, sorted_cols):
        header_cells = [None] * len(self._column_meta)
        for i, c in sorted_cols:
            header_cells[i] = self._get_fixed_width_cell(c, c['header'], self._widths[i])

        return header_cells

    def _stringified_row_iter(self, sorted_cols):
        row_cells = [None] * len(self._column_meta)
        for row in self._get_srows():
            for i, c in sorted_cols:
                row_cells[i] = self._get_fixed_width_cell(c, row[i], self._widths[i])
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

    def dump(self, stream=sys.stdout):
        self._compute_widths()
        sorted_cols = sorted(self._column_meta.items(), key=lambda x: x[0])

        header_cells = self._get_stringified_headers(sorted_cols)
        if self.top_border:
            print(self.get_top_border(header_cells), file=stream)

        print(self.add_side_borders(self.column_sep.join(header_cells)), file=stream)

        if self.header_bottom_border:
            print(self.get_header_bottom_border(header_cells), file=stream)

        for row_cells in self._stringified_row_iter(sorted_cols):
            print(self.add_side_borders(self.column_sep.join(row_cells)), file=stream)

    def __str__(self):
        """
        Return a string representation of the Table.
        """
        io = StringIO()
        self.dump(stream=io)
        return io.getvalue()


class TextTableBuilder(TableBuilder):
    def __init__(self, rows=None, headers=None, column_meta=None, precision=4):
        super().__init__(rows, headers, column_meta, precision)
        self.column_sep = ' | '
        self.top_border = '-'
        self.header_bottom_border = '-'
        self.bottom_border = '-'
        self.left_border = '| '
        self.right_border = ' |'


class GithubTableBuilder(TableBuilder):
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


class TabulatorJSBuilder(TableBuilder):
    allowed_col_meta = TableBuilder.allowed_col_meta.union({'sort', 'filter'})

    def __init__(self, rows=None, headers=None, column_meta=None, precision=4):
        super().__init__(rows, headers, column_meta, precision)



# table = []
# idx = 1  # unique ID for use by Tabulator
# for tgt, src in connections.items():
#     usrc = units[src]
#     utgt = units[tgt]
#     if usrc != utgt:
#         # prepend these with '!' so they'll be colored red
#         if usrc:
#             usrc = '!' + units[src]
#         if utgt:
#             utgt = '!' + units[tgt]

#     row = {'id': idx, 'src': src, 'sprom': sprom[src], 'sunits': usrc,
#             'val': _val2str(vals[tgt]), 'tunits': utgt,
#             'tprom': tprom[tgt], 'tgt': tgt}
#     table.append(row)
#     idx += 1

# if title is None:
#     title = ''

# data = {
#     'title': title,
#     'table': table,
# }

### JS template ###

# var table =
#     new Tabulator("#connection-table", {
#         // set height of table (in CSS or here), this enables the Virtual DOM and
#         // improves render speed dramatically (can be any valid css height value)
#         height: 650,
#         data:tabledata, //assign data to table
#         layout:"fitColumns", //"fitDataFill",
#         footerElement:"<p class='middle'>" +
#                             "<span class='toggle'><input type='checkbox' onclick='src_abs_toggle(this)'>Absolute Outputs</input></span>" +
#                             "<span class='toggle'><input type='checkbox' checked='true' onclick='src_prom_toggle(this)'>Promoted Outputs</input></span>" +
#                             "<span class='toggle'><input type='checkbox' checked='true' onclick='sunits_toggle(this)'>Output Units</input></span>" +
#                             valstr +
#                             "<span class='toggle'><input type='checkbox' checked='true' onclick='tunits_toggle(this)'>Input Units</input></span>" +
#                             "<span class='toggle'><input type='checkbox' onclick='tgt_abs_toggle(this)'>Absolute Inputs</input></span>" +
#                             "<span class='toggle'><input type='checkbox' checked='true' onclick='tgt_prom_toggle(this)'>Promoted Inputs</input></span></p>",
#         columns:[ //Define Table Columns
#                 {title: "Output (absolute)", field:"src", hozAlign:"left", headerFilter:true,
#                     visible:false, minWidth:300,
#                 tooltip:function(cell){
#                     return cell.getData().sprom;
#                 }},
#                 {title: "Output (promoted)", field:"sprom", hozAlign:"left", headerFilter:true, minWidth:300,
#                     formatter:function(cell, formmaterParams, onRendered) {
#                     var promname = cell.getData().sprom;
#                     if (promname == cell.getData().src) {
#                         return promname;
#                     }
#                     else {
#                         return "<span class='promoted'>" + promname + "</span>";
#                     }
#                 },
#                 tooltip:function(cell){
#                     return cell.getData().src;
#                 }},
#                 {title: "Units", field:"sunits", hozAlign:"center", headerFilter:true,
#                     formatter:function(cell, formatterParams){
#                     var value = cell.getValue();
#                     if(value.startsWith("!")){
#                         return "<span class='unitnomatch'>" + value.substring(1) + "</span>";
#                     }else{
#                         return value;
#                     }
#                     }},
#                 {title: "Value", visible:data.show_values, field:"val", hozAlign:"center", headerFilter:true},
#                 {title: "Units", field:"tunits", hozAlign:"center", headerFilter:true,
#                     formatter:function(cell, formatterParams){
#                         var value = cell.getValue();
#                         if(value.startsWith("!")){
#                             return "<span class='unitnomatch'>" + value.substring(1) + "</span>";
#                         }else{
#                             return value;
#                         }
#                     }
#                 },
#                 {title: "Input (absolute)", field:"tgt", hozAlign:"left", headerFilter:true,
#                     visible:false, minWidth:300,
#                     tooltip:function(cell){
#                     return cell.getData().tprom;
#                     }},
#                 {title: "Input (promoted)", field:"tprom", hozAlign:"left", headerFilter:true,
#                     minWidth:300,
#                     formatter:function(cell, formmaterParams, onRendered) {
#                     var promname = cell.getData().tprom;
#                     if (promname == cell.getData().tgt) {
#                         return promname;
#                     }
#                     else {
#                         return "<span class='promoted'>" + promname + "</span>";
#                     }
#                     },
#                     tooltip:function(cell){
#                     return cell.getData().tgt;
#                 }},
#         ],
# });

if __name__ == '__main__':
    import numpy as np
    import sys

    rows = np.arange(49, dtype=float).reshape((7,7))
    hdrs = [
        'foowergw',
        'bagwerwgwer',
        'fxxxoo',
        'xxx',
        'zzz',
        'abctyjrtyjtjd',
        'efgh',
    ]
    if 'github' in sys.argv:
        klass = GithubTableBuilder
    elif 'text' in sys.argv:
        klass = TextTableBuilder
    elif 'tabulator' in sys.argv:
        klass = TabulatorJSBuilder
    else:
        klass = TableBuilder
    tab = klass(rows, headers=hdrs, precision=4)
    tab.update_column_meta(5, align='center')
    print(tab)