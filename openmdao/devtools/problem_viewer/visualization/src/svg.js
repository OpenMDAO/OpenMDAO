var CONNECTION_COLOR = "black",
    UNKNOWN_IMPLICIT_COLOR = "orange",
    UNKNOWN_EXPLICIT_COLOR = "#AAA",
    N2_COMPONENT_BOX_COLOR = "#555",
    N2_BACKGROUND_COLOR = "#eee",
    N2_GRIDLINE_COLOR = "white",
    PT_STROKE_COLOR = "#eee",
    UNKNOWN_GROUP_COLOR = "#888",
    PARAM_COLOR = "Plum",
    PARAM_GROUP_COLOR = "Orchid",
    GROUP_COLOR = "steelblue",
    COMPONENT_COLOR = "DeepSkyBlue",
    COLLAPSED_COLOR = "#555";

function SaveSvg(parentDiv) {
    //get svg element.
    var svgData = parentDiv.querySelector("#svgId").outerHTML;

    //add name spaces.
    if (!svgData.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) {
        svgData = svgData.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    }
    if (!svgData.match(/^<svg[^>]+"http\:\/\/www\.w3\.org\/1999\/xlink"/)) {
        svgData = svgData.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
    }

    //add xml declaration
    svgData = '<?xml version="1.0" standalone="no"?>\r\n' + svgData;

    svgData = vkbeautify.xml(svgData);
    var svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
    var svgUrl = URL.createObjectURL(svgBlob);
    var downloadLink = document.createElement("a");
    downloadLink.href = svgUrl;
    downloadLink.download = "partition_tree_n2.svg";
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}

function UpdateSvgCss(svgStyleElement, FONT_SIZE_PX){
        var myCssText =
        "rect { " +
        "    stroke: " + PT_STROKE_COLOR + "; " +
        "} " +
        "g.unknown > rect { " +
        "    fill: " + UNKNOWN_EXPLICIT_COLOR + "; " +
        "    fill-opacity: .8; " +
        "} " +
        "g.unknown_implicit > rect { " +
        "    fill: " + UNKNOWN_IMPLICIT_COLOR + "; " +
        "    fill-opacity: .8; " +
        "} " +
        "g.param > rect { " +
        "    fill: " + PARAM_COLOR + "; " +
        "    fill-opacity: .8; " +
        "} " +
        "g.unconnected_param > rect { " +
        "    fill: " + UNCONNECTED_PARAM_COLOR + "; " +
        "    fill-opacity: .8; " +
        "} " +
        "g.subsystem > rect { " +
        "    cursor: pointer; " +
        "    fill-opacity: .8; " +
        "    fill: " + GROUP_COLOR + "; " +
        "} " +
        "g.component > rect { " +
        "    cursor: pointer; " +
        "    fill-opacity: .8; " +
        "    fill: " + COMPONENT_COLOR + "; " +
        "} " +
        "g.param_group > rect { " +
        "    cursor: pointer; " +
        "    fill-opacity: .8; " +
        "    fill: " + PARAM_GROUP_COLOR + "; " +
        "} " +
        "g.unknown_group > rect { " +
        "    cursor: pointer; " +
        "    fill-opacity: .8; " +
        "    fill: " + UNKNOWN_GROUP_COLOR + "; " +
        "} " +
        "g.minimized > rect { " +
        "    cursor: pointer; " +
        "    fill-opacity: .8; " +
        "    fill: " + COLLAPSED_COLOR + "; " +
        "} " +
        "text { " +
        //"    dominant-baseline: middle; " +
        //"    dy: .35em; " +
        "} " +
        "#svgId"+" g.partition_group > text { " +
        "    text-anchor: end; " +
        "    pointer-events: none; " +
        "    font-family: helvetica, sans-serif; " +
        "    font-size: " + FONT_SIZE_PX +"px; " +
        "} " +
        "/* n2 diagram*/  " +
        "g.component_box > rect { " +
        "    stroke: " + N2_COMPONENT_BOX_COLOR + "; " +
        "    stroke-width: 2; " +
        "    fill: none; " +
        "} " +
        ".bordR1, .bordR2, .bordR3, .bordR4, .ssMid, .grpMid, .svMid, .vsMid, .vMid, .sgrpMid, .grpsMid { " +
        "    stroke: none; " +
        "    stroke-width: 0; " +
        "    fill-opacity: 1; " +
        "} " +
        "[class^=n2_hover_elements] { " +
        "    pointer-events: none; " +
        "} " +
        ".background { " +
        "    fill: " + N2_BACKGROUND_COLOR + "; " +
        "} " +
        ".horiz_line, .vert_line { /*n2 gridlines*/ " +
        "    stroke: " + N2_GRIDLINE_COLOR + "; " +
        "}";

        svgStyleElement.innerHTML = myCssText;
    }