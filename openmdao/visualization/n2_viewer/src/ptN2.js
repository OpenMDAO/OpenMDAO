// The modelData object is generated and populated by n2_viewer.py
let n2Diag = new N2Diagram(modelData);
let n2UI = new N2UserInterface(n2Diag);
let n2MouseFuncs = n2Diag.getMouseFuncs();

// GLOBAL VARIABLES /////
// TODO: Get rid of all these while refactoring ///////////////
var lastLeftClickedEle;
var lastRightClickedEle;
var lastRightClickedObj;
var lastRightClickedElement = n2Diag.model.root;
///////////////////////////////////////////////////////////////

CreateDomLayout();
CreateToolbar();

n2Diag.update(false);
SetupLegend(d3, n2Diag.dom.d3ContentDiv);

//right click => collapse
function RightClick(d, ele) {
    lastLeftClickedEle = d;
    lastRightClickedObj = d;
    lastRightClickedEle = ele;
    d3.event.preventDefault();
    n2UI.collapse();
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
