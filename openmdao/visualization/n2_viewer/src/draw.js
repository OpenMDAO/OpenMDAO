var d3ContentDiv, svgDiv, svg;
var n2Group;

var WIDTH_N2_PX = HEIGHT_PX;
var PTREE_N2_GAP_PX = 10; //spacing between partition tree and n2 diagram
var n2Dx = 0, n2Dy = 0, n2Dx0 = 0, n2Dy0 = 0;
var d3NodesArray, d3RightTextNodesArrayZoomed = []
var d3RightTextNodesArrayZoomedBoxInfo
var drawableN2ComponentBoxes;
var matrix;

var gridLines;

var sharedTransition;
var FindRootOfChangeFunction = null;

var lastLeftClickedElement = document.getElementById("ptN2ContentDivId"),
    leftClickIsForward = true,
    lastClickWasLeft = true;

var enterIndex = 0,
    exitIndex = 0;

function DrawRect(x, y, width, height, fill) {
    n2Group.insert("rect")
        .attr("class", "n2_hover_elements")
        .attr("y", y)
        .attr("x", x)
        .attr("width", width)
        .attr("height", height)
        .attr("fill", fill)
        .attr("fill-opacity", "1");
}

function DrawArrowsParamView(startIndex, endIndex) {
    var lineWidth = Math.min(5, n2Dx * .5, n2Dy * .5);
    n2Diag.arrowMarker.attr("markerWidth", lineWidth * .4)
        .attr("markerHeight", lineWidth * .4);

    var boxStart = d3RightTextNodesArrayZoomedBoxInfo[startIndex];
    var boxEnd = d3RightTextNodesArrayZoomedBoxInfo[endIndex];

    //draw multiple horizontal lines but no more than one vertical line for box to box connections
    var startIndices = [], endIndices = [];
    for (var startsI = boxStart.startI; startsI <= boxStart.stopI; ++startsI) {
        for (var endsI = boxEnd.startI; endsI <= boxEnd.stopI; ++endsI) {
            if (matrix.node(startsI, endsI) !== undefined) {
                startIndices.push(startsI);
                endIndices.push(endsI);
            }
        }
    }

    for (var i = 0; i < startIndices.length; ++i) {
        var startI = startIndices[i];
        var endI = endIndices[i];
        new N2Arrow({
            start: { col: startI, row: startI },
            end: { col: endI, row: endI },
            color: (startIndex < endIndex) ? GREEN_ARROW_COLOR : RED_ARROW_COLOR,
            width: lineWidth
        }, n2Diag.n2Groups);
    }
}
