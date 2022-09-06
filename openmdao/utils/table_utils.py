

table = []
idx = 1  # unique ID for use by Tabulator
for tgt, src in connections.items():
    usrc = units[src]
    utgt = units[tgt]
    if usrc != utgt:
        # prepend these with '!' so they'll be colored red
        if usrc:
            usrc = '!' + units[src]
        if utgt:
            utgt = '!' + units[tgt]

    row = {'id': idx, 'src': src, 'sprom': sprom[src], 'sunits': usrc,
            'val': _val2str(vals[tgt]), 'tunits': utgt,
            'tprom': tprom[tgt], 'tgt': tgt}
    table.append(row)
    idx += 1

# add rows for unconnected sources
for src, _ in all_vars['output']:
    if src not in src2tgts:
        if show_values:
            v = _val2str(system._abs_get_val(src))
        else:
            v = ''
        row = {'id': idx, 'src': src, 'sprom': sprom[src], 'sunits': units[src],
                'val': v, 'tunits': '', 'tprom': NOCONN, 'tgt': NOCONN}
        table.append(row)
        idx += 1

if title is None:
    title = ''

data = {
    'title': title,
    'table': table,
}



### JS template ###

var table =
    new Tabulator("#connection-table", {
        // set height of table (in CSS or here), this enables the Virtual DOM and
        // improves render speed dramatically (can be any valid css height value)
        height: 650,
        data:tabledata, //assign data to table
        layout:"fitColumns", //"fitDataFill",
        footerElement:"<p class='middle'>" +
                            "<span class='toggle'><input type='checkbox' onclick='src_abs_toggle(this)'>Absolute Outputs</input></span>" +
                            "<span class='toggle'><input type='checkbox' checked='true' onclick='src_prom_toggle(this)'>Promoted Outputs</input></span>" +
                            "<span class='toggle'><input type='checkbox' checked='true' onclick='sunits_toggle(this)'>Output Units</input></span>" +
                            valstr +
                            "<span class='toggle'><input type='checkbox' checked='true' onclick='tunits_toggle(this)'>Input Units</input></span>" +
                            "<span class='toggle'><input type='checkbox' onclick='tgt_abs_toggle(this)'>Absolute Inputs</input></span>" +
                            "<span class='toggle'><input type='checkbox' checked='true' onclick='tgt_prom_toggle(this)'>Promoted Inputs</input></span></p>",
        columns:[ //Define Table Columns
                {title: "Output (absolute)", field:"src", hozAlign:"left", headerFilter:true,
                    visible:false, minWidth:300,
                tooltip:function(cell){
                    return cell.getData().sprom;
                }},
                {title: "Output (promoted)", field:"sprom", hozAlign:"left", headerFilter:true, minWidth:300,
                    formatter:function(cell, formmaterParams, onRendered) {
                    var promname = cell.getData().sprom;
                    if (promname == cell.getData().src) {
                        return promname;
                    }
                    else {
                        return "<span class='promoted'>" + promname + "</span>";
                    }
                },
                tooltip:function(cell){
                    return cell.getData().src;
                }},
                {title: "Units", field:"sunits", hozAlign:"center", headerFilter:true,
                    formatter:function(cell, formatterParams){
                    var value = cell.getValue();
                    if(value.startsWith("!")){
                        return "<span class='unitnomatch'>" + value.substring(1) + "</span>";
                    }else{
                        return value;
                    }
                    }},
                {title: "Value", visible:data.show_values, field:"val", hozAlign:"center", headerFilter:true},
                {title: "Units", field:"tunits", hozAlign:"center", headerFilter:true,
                    formatter:function(cell, formatterParams){
                        var value = cell.getValue();
                        if(value.startsWith("!")){
                            return "<span class='unitnomatch'>" + value.substring(1) + "</span>";
                        }else{
                            return value;
                        }
                    }
                },
                {title: "Input (absolute)", field:"tgt", hozAlign:"left", headerFilter:true,
                    visible:false, minWidth:300,
                    tooltip:function(cell){
                    return cell.getData().tprom;
                    }},
                {title: "Input (promoted)", field:"tprom", hozAlign:"left", headerFilter:true,
                    minWidth:300,
                    formatter:function(cell, formmaterParams, onRendered) {
                    var promname = cell.getData().tprom;
                    if (promname == cell.getData().tgt) {
                        return promname;
                    }
                    else {
                        return "<span class='promoted'>" + promname + "</span>";
                    }
                    },
                    tooltip:function(cell){
                    return cell.getData().tgt;
                }},
        ],
});
