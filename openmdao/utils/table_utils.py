
import sys

from numbers import Number, Integral


class TableBuilder(object):
    def __init__(self, rows=None, column_meta=None, precision=4):
        if rows is None:
            self._raw_rows = []
        else:
            self._raw_rows = rows
        self._column_meta = {}
        self._widths = None
        self._rows = None

        # these are the default format strings for the first formatting stage,
        # before the column width is set
        self._default_formats = {
            'real': "{" + f":.{precision}" + "}",
            'integral': "{}",
            'other': "{}",
        }

        if column_meta is not None:
            for i, colinf in enumerate(column_meta):
                self.add_column_info(i, **colinf)

    def _get_srows(self):
        if self._rows is None:
            self._rows = []
            for row in self._raw_rows:
                self._add_srow(row)
        return self._rows

    def _check(self):
        if len(self._rows) != len(self._column_meta):
            raise RuntimeError("Number of row entries must match number of column infos in "
                               "TableBuilder.")

    def _compute_widths(self):
        if self._widths is not None:
            return  # widths already computed

        rows = self._get_srows()

        self._check()

        self._widths = [0] * len(self._column_meta)

        for row in rows:
            for i in range(len(self._column_meta)):
                wid = len(row[i])
                if wid > self._widths[i]:
                    self._widths[i] = wid

        for i, cinfo in sorted(self._column_meta.items(), key=lambda x: x[0]):
            wid = len(cinfo['header'])
            if wid > self._widths[i]:
                self._widths[i] = wid

    def add_row(self, row):
        # raw_list could have originally been an iterator, and if so add_row wouldn't normally
        # be called, but if it is, we copy the iterator's contents so we can add to them.
        if not isinstance(self._raw_rows, list):  # not likely, but handle it anyway
            self._raw_rows = list(self._raw_rows)
        self._raw_rows.append(row)

    def _add_srow(self, row):
        if not self._rows:  # if this is the first row
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

        cells = [self._column_meta[i]['format'].format(cell) for i, cell in enumerate(row)]

        if self._rows and len(cells) != len(self._rows[-1]):
            raise RuntimeError("Can't add rows of unequal length to TableBuilder.")

        self._rows.append(cells)

    def add_column_info(self, col_idx, **options):
        if col_idx not in self._column_meta:
            self._column_meta[col_idx] = {'header': ''}
        self._column_meta[col_idx].update(options)

    def dump(self, stream=sys.stdout):
        header_cells = [None] * len(self._column_meta)
        self._compute_widths()
        a2sym = {'center': '^', 'right': '>', 'left': '<'}
        sorted_cols = sorted(self._column_meta.items(), key=lambda x: x[0])
        for i, c in sorted_cols:
            sym = a2sym[c.get('align', 'left')]
            header_cells[i] = f"{c['header']:{sym}{self._widths[i]}}"

        header_line = ' | '.join(header_cells)

        print('-' * len(header_line), file=stream)
        print(header_line, file=stream)
        print('-' * len(header_line), file=stream)

        row_cells = [None] * len(self._column_meta)
        for row in self._get_srows():
            for i, c in sorted_cols:
                sym = a2sym[c.get('align', 'left')]
                row_cells[i] = f"{row[i]:{sym}{self._widths[i]}}"

            print(' | '.join(row_cells), file=stream)


class TabulatorJSBuilder(TableBuilder):
    def __init__(self, rows=None, column_meta=None, precision=4, **options):
        super().__init__(rows, column_meta, precision)
        self._options = options

    def __str__(self):
        """
        Return a string representation of the Table.
        """
        pass


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
    rows = np.arange(100, dtype=float).reshape((10,10))
    cols = [
        {'header': 'foowergw'},
        {'header': 'bagwerwgwer'},
        {'header': 'fxxxoo'},
        {'header': 'xxx'},
        {'header': 'zzz'},
        {'header': 'abctyjrtyjtjd'},
        {'header': 'efgh'},
        {'header': 'sdfgsd'},
        {'header': 'uioiu'},
        {'header': 'vccbfc'},
    ]
    tab = TabulatorJSBuilder(rows, column_meta=cols, precision=4)
    tab.add_column_info(5, align='center', format="{:40.3e}")
    tab.dump()