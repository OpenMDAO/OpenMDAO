
import sys

from numbers import Number, Integral


class TableBuilder(object):
    def __init__(self, rows=None, column_infos=None, precision=4):
        self._rows = []
        self._column_infos = {}
        self._widths = None
        self._precision = precision

        if column_infos is not None:
            for i, colinf in enumerate(column_infos):
                self.add_column_info(i, **colinf)

        if rows is not None:
            for row in rows:
                self.add_row(row)

    def _check(self):
        if len(self._rows) != len(self._column_infos):
            raise RuntimeError("Number of row entries must match number of column infos in "
                               "TableBuilder.")

    def _compute_widths(self):
        if self._widths is not None:
            return  # widths already computed

        self._check()
        
        self._widths = [0] * len(self._column_infos)

        for row in self._rows:
            for i in range(len(self._column_infos)):
                wid = len(row[i])
                if wid > self._widths[i]:
                    self._widths[i] = wid

        for i, cinfo in sorted(self._column_infos.items(), key=lambda x: x[0]):
            wid = len(cinfo['title'])
            if wid > self._widths[i]:
                self._widths[i] = wid

    def add_row(self, row):
        cells = []
        for i, cell in enumerate(row):
            align = 'left'
            if isinstance(cell, Number):
                align = 'right'
                if isinstance(cell, Integral):
                    cells.append(str(cell))
                else:
                    cells.append(f'{cell:.{self._precision}}')
            elif isinstance(cell, str):
                cells.append(cell)
            else:
                cells.append(str(cell))

            if i not in self._column_infos:
                self._column_infos[i] = {'title': ''}
                
            if 'align' not in self._column_infos[i]:
                self._column_infos[i]['align'] = align

        if self._rows and len(cells) != len(self._rows[-1]):
            raise RuntimeError("Can't add rows of unequal length to TableBuilder.")

        self._rows.append(cells)

    def add_column_info(self, col_idx, **options):
        if col_idx not in self._column_infos:
            self._column_infos[col_idx] = {'title': ''}
        self._column_infos[col_idx].update(options)

    def dump(self, stream=sys.stdout):
        header_cells = [None] * len(self._column_infos)
        self._compute_widths()
        a2sym = {'center': '^', 'right': '>', 'left': '<'}
        sorted_cols = sorted(self._column_infos.items(), key=lambda x: x[0])
        for i, c in sorted_cols:
            sym = a2sym[c.get('align', 'left')]
            header_cells[i] = f"{c['title']:{sym}{self._widths[i]}}"

        header_line = ' | '.join(header_cells)
        
        print('-' * len(header_line), file=stream)
        print(header_line, file=stream)
        print('-' * len(header_line), file=stream)
        
        row_cells = [None] * len(self._column_infos)
        for row in rows:
            for i, c in sorted_cols:
                sym = a2sym[c.get('align', 'left')]
                row_cells[i] = f"{row[i]:{sym}{self._widths[i]}}"
                
            print(' | '.join(row_cells), file=stream)


class TabulatorJSBuilder(TableBuilder):
    def __init__(self, rows=None, column_infos=None, precision=4, **options):
        super().__init__(rows, column_infos, precision)
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
        {'title': 'foo'},
        {'title': 'bar'},
        {'title': 'fxxxoo'},
        {'title': 'xxx'},
        {'title': 'zzz'},
        {'title': 'abcd'},
        {'title': 'efgh'},
        {'title': 'sdfgsd'},
        {'title': 'uioiu'},
        {'title': 'vccbfc'},
    ]
    tab = TabulatorJSBuilder(rows, column_infos=cols)
    tab.dump()