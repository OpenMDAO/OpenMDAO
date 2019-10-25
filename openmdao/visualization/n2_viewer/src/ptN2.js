// The modelData object is generated and populated by n2_viewer.py
let n2Diag = new N2Diagram(modelData);

// GLOBAL VARIABLES /////
// TODO: Get rid of all these while refactoring ///////////////
var mouseOverOffDiagN2 = MouseoverOffDiagN2;
var mouseOverOnDiagN2 = MouseoverOnDiagN2;
var mouseOutN2 = MouseoutN2;
var mouseClickN2 = MouseClickN2;

var lastLeftClickedEle;
var lastRightClickedEle;
var lastRightClickedObj;
var lastRightClickedElement = n2Diag.model.root;
///////////////////////////////////////////////////////////////

CreateDomLayout();
CreateToolbar();

var collapseDepthElement = parentDiv.querySelector("#idCollapseDepthDiv");
for (var i = 2; i <= n2Diag.model.maxDepth; ++i) {
    var option = document.createElement("span");
    option.className = "fakeLink";
    option.id = "idCollapseDepthOption" + i + "";
    option.innerHTML = "" + i + "";
    var f = function (idx) {
        return function () {
            CollapseToDepthSelectChange(idx);
        };
    }(i);
    option.onclick = f;
    collapseDepthElement.appendChild(option);
}

n2Diag.update(false);
SetupLegend(d3, n2Diag.dom.d3ContentDiv);

//right click => collapse
function RightClick(d, ele) {
    lastLeftClickedEle = d;
    lastRightClickedObj = d;
    lastRightClickedEle = ele;
    d3.event.preventDefault();
    collapse();
}

function collapse() {
    let d = lastLeftClickedEle;
    if (!d.hasChildren()) return;
    if (d.depth > n2Diag.zoomedElement.depth) { //dont allow minimizing on root node
        lastRightClickedElement = d;
        FindRootOfChangeFunction = FindRootOfChangeForRightClick;
        N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
        lastClickWasLeft = false;
        d.toggleMinimize();
        n2Diag.update();
    }
}

function SetupLeftClick(d) {
    lastLeftClickedElement = d;
    lastClickWasLeft = true;
    if (lastLeftClickedElement.depth > n2Diag.zoomedElement.depth) {
        leftClickIsForward = true; //forward
    }
    else if (lastLeftClickedElement.depth < n2Diag.zoomedElement.depth) {
        leftClickIsForward = false; //backwards
    }
    n2Diag.updateZoomedElement(d);
    N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
}

//left click => navigate
function LeftClick(d, ele) {
    if (!d.hasChildren()) return;
    if (d3.event.button != 0) return;
    n2Diag.backButtonHistory.push({ "el": n2Diag.zoomedElement });
    n2Diag.forwardButtonHistory = [];
    SetupLeftClick(d);
    n2Diag.update();
    d3.event.preventDefault();
    d3.event.stopPropagation();
}

function BackButtonPressed() {
    if (n2Diag.backButtonHistory.length == 0) return;
    var d = n2Diag.backButtonHistory.pop().el;
    parentDiv.querySelector("#backButtonId").disabled = (n2Diag.backButtonHistory.length == 0) ? "disabled" : false;
    for (var obj = d; obj != null; obj = obj.parent) { //make sure history item is not minimized
        if (obj.isMinimized) return;
    }
    n2Diag.forwardButtonHistory.push({ "el": n2Diag.zoomedElement });
    SetupLeftClick(d);
    n2Diag.update();
}

function ForwardButtonPressed() {
    if (n2Diag.forwardButtonHistory.length == 0) return;
    var d = n2Diag.forwardButtonHistory.pop().el;
    parentDiv.querySelector("#forwardButtonId").disabled = (n2Diag.forwardButtonHistory.length == 0) ? "disabled" : false;
    for (var obj = d; obj != null; obj = obj.parent) { //make sure history item is not minimized
        if (obj.isMinimized) return;
    }
    n2Diag.backButtonHistory.push({ "el": n2Diag.zoomedElement });
    SetupLeftClick(d);
    n2Diag.update();
}

function FindRootOfChangeForRightClick(d) {
    return lastRightClickedElement;
}

function FindRootOfChangeForCollapseDepth(d) {
    for (let obj = d; obj != null; obj = obj.parent) { //make sure history item is not minimized
        if (obj.depth == n2Diag.chosenCollapseDepth) return obj;
    }
    return d;
}

function FindRootOfChangeForCollapseUncollapseOutputs(d) {
    return (d.hasOwnProperty("parentComponent")) ? d.parentComponent : d;
}

function MouseoverOffDiagN2(d) {
    function GetObjectsInChildrenWithCycleArrows(d, arr) {
        if (d.cycleArrows) {
            arr.push(d);
        }
        if (d.hasChildren()) {
            for (var i = 0; i < d.children.length; ++i) {
                GetObjectsInChildrenWithCycleArrows(d.children[i], arr);
            }
        }
    }
    function GetObjectsWithCycleArrows(d, arr) {
        for (var obj = d.parent; obj != null; obj = obj.parent) { //start with parent.. the children will get the current object to avoid duplicates
            if (obj.cycleArrows) {
                arr.push(obj);
            }
        }
        GetObjectsInChildrenWithCycleArrows(d, arr);
    }

    function HasObjectInChildren(d, toMatchObj) {
        if (d === toMatchObj) {
            return true;
        }
        if (d.hasChildren()) {
            for (var i = 0; i < d.children.length; ++i) {
                if (HasObjectInChildren(d.children[i], toMatchObj)) {
                    return true;
                }
            }
        }
        return false;
    }

    function HasObject(d, toMatchObj) {
        // console.log("HasObject(", d, ", ", toMatchObj, ")");
        for (var obj = d; obj != null; obj = obj.parent) {
            if (obj === toMatchObj) {
                return true;
            }
        }
        return HasObjectInChildren(d, toMatchObj);
    }

    var lineWidth = Math.min(5, n2Diag.matrix.nodeSize.width * .5, n2Diag.matrix.nodeSize.height * .5);
    n2Diag.dom.arrowMarker.attr("markerWidth", lineWidth * .4)
        .attr("markerHeight", lineWidth * .4);
    var src = n2Diag.layout.visibleNodes[d.row];
    var tgt = n2Diag.layout.visibleNodes[d.col];
    var boxEnd = n2Diag.matrix.boxInfo[d.col];

    new N2Arrow({
        start: { col: d.row, row: d.row },
        end: { col: d.col, row: d.col },
        color: N2Style.color.redArrow,
        width: lineWidth
    }, n2Diag.dom.n2Groups, n2Diag.matrix.nodeSize);

    if (d.row > d.col) {
        var targetsWithCycleArrows = [];
        GetObjectsWithCycleArrows(tgt, targetsWithCycleArrows);
        for (var ti = 0; ti < targetsWithCycleArrows.length; ++ti) {
            var arrows = targetsWithCycleArrows[ti].cycleArrows;
            for (var ai = 0; ai < arrows.length; ++ai) {
                if (HasObject(src, arrows[ai].src)) {
                    var correspondingSrcArrows = arrows[ai].arrows;
                    for (var si = 0; si < correspondingSrcArrows.length; ++si) {
                        var beginObj = correspondingSrcArrows[si].begin;
                        var endObj = correspondingSrcArrows[si].end;
                        //alert(beginObj.name + "->" + endObj.name);
                        var firstBeginIndex = -1, firstEndIndex = -1;

                        //find first begin index
                        for (var mi = 0; mi < n2Diag.layout.visibleNodes.length; ++mi) {
                            var rtNode = n2Diag.layout.visibleNodes[mi];
                            if (HasObject(rtNode, beginObj)) {
                                firstBeginIndex = mi;
                                break;
                            }
                        }
                        if (firstBeginIndex == -1) {
                            alert("error: first begin index not found");
                            return;
                        }

                        //find first end index
                        for (var mi = 0; mi < n2Diag.layout.visibleNodes.length; ++mi) {
                            var rtNode = n2Diag.layout.visibleNodes[mi];
                            if (HasObject(rtNode, endObj)) {
                                firstEndIndex = mi;
                                break;
                            }
                        }
                        if (firstEndIndex == -1) {
                            alert("error: first end index not found");
                            return;
                        }

                        if (firstBeginIndex != firstEndIndex) {
                            DrawArrowsParamView(firstBeginIndex, firstEndIndex, n2Diag.matrix.nodeSize);
                        }
                    }
                }
            }
        }
    }

    var leftTextWidthR = n2Diag.layout.visibleNodes[d.row].nameWidthPx,
        leftTextWidthC = n2Diag.layout.visibleNodes[d.col].nameWidthPx;
    DrawRect(-leftTextWidthR - n2Diag.layout.size.partitionTreeGap, n2Diag.matrix.nodeSize.height * d.row, leftTextWidthR, n2Diag.matrix.nodeSize.height, N2Style.color.redArrow); //highlight var name
    DrawRect(-leftTextWidthC - n2Diag.layout.size.partitionTreeGap, n2Diag.matrix.nodeSize.height * d.col, leftTextWidthC, n2Diag.matrix.nodeSize.height, N2Style.color.greenArrow); //highlight var name
}

function MouseoverOnDiagN2(d) {
    //d=hovered element
    // console.log('MouseoverOnDiagN2:'); console.log(d);
    var hoveredIndexRC = d.col; //d.dims.x == d.dims.y == row == col
    var leftTextWidthHovered = n2Diag.layout.visibleNodes[hoveredIndexRC].nameWidthPx;

    // Loop over all elements in the matrix looking for other cells in the same column as
    var lineWidth = Math.min(5, n2Diag.matrix.nodeSize.width * .5, n2Diag.matrix.nodeSize.height * .5);
    n2Diag.dom.arrowMarker.attr("markerWidth", lineWidth * .4)
        .attr("markerHeight", lineWidth * .4);
    DrawRect(-leftTextWidthHovered - n2Diag.layout.size.partitionTreeGap, n2Diag.matrix.nodeSize.height * hoveredIndexRC, leftTextWidthHovered, n2Diag.matrix.nodeSize.height, N2Style.color.highlightHovered); //highlight hovered
    for (var i = 0; i < n2Diag.layout.visibleNodes.length; ++i) {
        var leftTextWidthDependency = n2Diag.layout.visibleNodes[i].nameWidthPx;
        var box = n2Diag.matrix.boxInfo[i];
        if (n2Diag.matrix.exists(hoveredIndexRC, i)) { //i is column here
            if (i != hoveredIndexRC) {
                new N2Arrow({
                    end: { col: i, row: i },
                    start: { col: hoveredIndexRC, row: hoveredIndexRC },
                    color: N2Style.color.greenArrow,
                    width: lineWidth
                }, n2Diag.dom.n2Groups, n2Diag.matrix.nodeSize);
                DrawRect(-leftTextWidthDependency - n2Diag.layout.size.partitionTreeGap, n2Diag.matrix.nodeSize.height * i, leftTextWidthDependency, n2Diag.matrix.nodeSize.height, N2Style.color.greenArrow); //highlight var name
            }
        }

        if (n2Diag.matrix.cell(i, hoveredIndexRC) !== undefined) { //i is row here
            if (i != hoveredIndexRC) {
                new N2Arrow({
                    start: { col: i, row: i },
                    end: { col: hoveredIndexRC, row: hoveredIndexRC },
                    color: N2Style.color.redArrow,
                    width: lineWidth
                }, n2Diag.dom.n2Groups, n2Diag.matrix.nodeSize);
                DrawRect(-leftTextWidthDependency - n2Diag.layout.size.partitionTreeGap, n2Diag.matrix.nodeSize.height * i, leftTextWidthDependency, n2Diag.matrix.nodeSize.height, N2Style.color.redArrow); //highlight var name
            }
        }
    }
}

function MouseoutN2() {
    n2Diag.dom.n2TopGroup.selectAll(".n2_hover_elements").remove();
}

function MouseClickN2(d) {
    var newClassName = "n2_hover_elements_" + d.row + "_" + d.col;
    var selection = n2Diag.dom.n2TopGroup.selectAll("." + newClassName);
    if (selection.size() > 0) {
        selection.remove();
    }
    else {
        n2Diag.dom.n2TopGroup.selectAll("path.n2_hover_elements, circle.n2_hover_elements")
            .attr("class", newClassName);
    }
}

function ReturnToRootButtonClick() {
    n2Diag.backButtonHistory.push({ "el": n2Diag.zoomedElement });
    n2Diag.forwardButtonHistory = [];
    SetupLeftClick(n2Diag.model.root);
    n2Diag.update();
}

function UpOneLevelButtonClick() {
    if (n2Diag.zoomedElement === n2Diag.model.root) return;
    n2Diag.backButtonHistory.push({ "el": n2Diag.zoomedElement });
    n2Diag.forwardButtonHistory = [];
    SetupLeftClick(n2Diag.zoomedElement.parent);
    n2Diag.update();
}

function CollapseOutputsButtonClick(startNode) {
    function CollapseOutputs(d) {
        if (d.subsystem_type && d.subsystem_type === "component") {
            d.isMinimized = true;
        }
        if (d.hasChildren()) {
            for (var i = 0; i < d.children.length; ++i) {
                CollapseOutputs(d.children[i]);
            }
        }
    }
    FindRootOfChangeFunction = FindRootOfChangeForCollapseUncollapseOutputs;
    N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
    lastClickWasLeft = false;
    CollapseOutputs(startNode);
    n2Diag.update();
}

function UncollapseButtonClick(startNode) {
    function Uncollapse(d) {
        if (! d.isParam()) {
            d.isMinimized = false;
        }
        if (d.hasChildren()) {
            for (var i = 0; i < d.children.length; ++i) {
                Uncollapse(d.children[i]);
            }
        }
    }
    FindRootOfChangeFunction = FindRootOfChangeForCollapseUncollapseOutputs;
    N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
    lastClickWasLeft = false;
    Uncollapse(startNode);
    n2Diag.update();
}

function CollapseToDepthSelectChange(newChosenCollapseDepth) {
    function CollapseToDepth(d, depth) {
        if (d.isParamOrUnknown()) {
            return;
        }

        d.isMinimized = (d.depth < depth)? false : true;

        if (d.hasChildren()) {
            for (let child of d.children) {
                CollapseToDepth(child, depth);
            }
        }
    }

    n2Diag.chosenCollapseDepth = newChosenCollapseDepth;
    if (n2Diag.chosenCollapseDepth > n2Diag.zoomedElement.depth) {
        CollapseToDepth(n2Diag.model.root, n2Diag.chosenCollapseDepth);
    }
    FindRootOfChangeFunction = FindRootOfChangeForCollapseDepth;
    N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
    lastClickWasLeft = false;
    n2Diag.update();
}

function ToggleSolverNamesCheckboxChange() {
    n2Diag.toggleSolverNameType();
    parentDiv.querySelector("#toggleSolverNamesButtonId").className = !n2Diag.showLinearSolverNames ? "myButton myButtonToggledOn" : "myButton";
    SetupLegend(d3, n2Diag.dom.d3ContentDiv);
    n2Diag.update();
};

function ShowPathCheckboxChange() {
    n2Diag.showPath = !n2Diag.showPath;
    parentDiv.querySelector("#currentPathId").style.display = n2Diag.showPath ? "block" : "none";
    parentDiv.querySelector("#showCurrentPathButtonId").className = n2Diag.showPath ? "myButton myButtonToggledOn" : "myButton";
}

function ToggleLegend() {
    showLegend = !showLegend;
    parentDiv.querySelector("#showLegendButtonId").className = showLegend ? "myButton myButtonToggledOn" : "myButton";
    SetupLegend(d3, n2Diag.dom.d3ContentDiv);
}

function CreateDomLayout() {
    document.getElementById("searchButtonId").onclick = SearchButtonClicked;
}

function CreateToolbar() {
    var div = document.getElementById("toolbarDiv")
    div.querySelector("#returnToRootButtonId").onclick = ReturnToRootButtonClick;
    div.querySelector("#backButtonId").onclick = BackButtonPressed;
    div.querySelector("#forwardButtonId").onclick = ForwardButtonPressed;
    div.querySelector("#upOneLevelButtonId").onclick = UpOneLevelButtonClick;
    div.querySelector("#uncollapseInViewButtonId").onclick = function () { UncollapseButtonClick(n2Diag.zoomedElement); };
    div.querySelector("#uncollapseAllButtonId").onclick = function () { UncollapseButtonClick(n2Diag.model.root); };
    div.querySelector("#collapseInViewButtonId").onclick = function () { CollapseOutputsButtonClick(n2Diag.zoomedElement); };
    div.querySelector("#collapseAllButtonId").onclick = function () { CollapseOutputsButtonClick(n2Diag.model.root); };
    div.querySelector("#clearArrowsAndConnectsButtonId").onclick = n2Diag.clearArrows.bind(n2Diag);
    div.querySelector("#showCurrentPathButtonId").onclick = ShowPathCheckboxChange;
    div.querySelector("#showLegendButtonId").onclick = ToggleLegend;

    div.querySelector("#toggleSolverNamesButtonId").onclick = ToggleSolverNamesCheckboxChange;

    for (var i = 8; i <= 14; ++i) {
        var f = function (idx) {
            return function () { n2Diag.fontSizeSelectChange(idx); };
        }(i);
        div.querySelector("#idFontSize" + i + "px").onclick = f;
    }

    for (var i = 600; i <= 1000; i += 50) {
        var f = function (idx) {
            return function () { n2Diag.verticalResize(idx); };
        }(i);
        div.querySelector("#idVerticalResize" + i + "px").onclick = f;
    }
    for (var i = 2000; i <= 4000; i += 1000) {
        var f = function (idx) {
            return function () { n2Diag.verticalResize(idx); };
        }(i);
        div.querySelector("#idVerticalResize" + i + "px").onclick = f;
    }

    div.querySelector("#saveSvgButtonId").onclick = n2Diag.saveSvg.bind(n2Diag);
    div.querySelector("#helpButtonId").onclick = DisplayModal;
}
