function PtN2Diagram(parentDiv, modelJSON) {

    // TODO: Get rid of all these after refactoring ///////////////
    var model = n2Diag.model; ////

    svgDiv = n2Diag.svgDiv; ////
    svg = n2Diag.svg; ////

    var root = model.root;
    var abs2prom = model.abs2prom;

    var showPath = n2Diag.showPath; //default off ////
    var DEFAULT_TRANSITION_START_DELAY = N2Diagram.defaultTransitionStartDelay; ////


    var chosenCollapseDepth = n2Diag.chosenCollapseDepth; ////

    var tooltip = n2Diag.toolTip; ////
    ///////////////////////////////////////////////////////////////

    mouseOverOnDiagN2 = MouseoverOnDiagN2;
    mouseOverOffDiagN2 = MouseoverOffDiagN2;
    mouseClickN2 = MouseClickN2;
    mouseOutN2 = MouseoutN2;

    // TODO: Get rid of all these globals after refactoring ///////////////
    d3NodesArray = n2Diag.layout.zoomedNodes;
    d3RightTextNodesArrayZoomed = n2Diag.layout.visibleNodes;

    d3SolverNodesArray = n2Diag.layout.zoomedSolverNodes;
    d3SolverRightTextNodesArrayZoomed = n2Diag.layout.visibleSolverNodes;
    ///////////////////////////////////////////////////////////////

    CreateDomLayout();
    CreateToolbar();

    var lastRightClickedElement = model.root;

    var collapseDepthElement = parentDiv.querySelector("#idCollapseDepthDiv");
    for (var i = 2; i <= model.maxDepth; ++i) {
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

    Update(false);
    SetupLegend(d3, n2Diag.d3ContentDiv);

    function Update(computeNewTreeLayout = true) {
        n2Diag.update(computeNewTreeLayout);

        // TODO: Get rid of all these after refactoring ///////////////
        d3NodesArray = n2Diag.layout.zoomedNodes;
        d3RightTextNodesArrayZoomed = n2Diag.layout.visibleNodes;

        d3SolverNodesArray = n2Diag.layout.zoomedSolverNodes;
        d3SolverRightTextNodesArrayZoomed = n2Diag.layout.visibleSolverNodes;
        ///////////////////////////////////////////////////////////////

        var sel = n2Diag.pTreeGroup.selectAll(".partition_group")
            .data(d3NodesArray, function (d) {
                return d.id;
            });

        var nodeEnter = sel.enter().append("svg:g")
            .attr("class", function (d) {
                return "partition_group " + n2Diag.style.getNodeClass(d);
            })
            .attr("transform", function (d) {
                return "translate(" + n2Diag.scales.previous.model.x(d.x0) + "," + n2Diag.scales.previous.model.y(d.y0) + ")";
            })
            .on("click", function (d) { LeftClick(d, this); })
            .on("contextmenu", function (d) { RightClick(d, this); })
            .on("mouseover", function (d) {
                if (abs2prom != undefined) {
                    if (d.type == "param" || d.type == "unconnected_param") {
                        return tooltip.text(abs2prom.input[d.absPathName])
                            .style("visibility", "visible");
                    }
                    if (d.type == "unknown") {
                        return tooltip.text(abs2prom.output[d.absPathName])
                            .style("visibility", "visible");
                    }
                }
            })
            .on("mouseleave", function (d) {
                if (abs2prom != undefined) {
                    return tooltip.style("visibility", "hidden");
                }
            })
            .on("mousemove", function () {
                if (abs2prom != undefined) {
                    return tooltip.style("top", (d3.event.pageY - 30) + "px")
                        .style("left", (d3.event.pageX + 5) + "px");
                }
            });

        nodeEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.width0 * n2Diag.transitCoords.previous.model.x;
            })
            .attr("height", function (d) {
                return d.height0 * n2Diag.transitCoords.previous.model.y;
            });

        nodeEnter.append("svg:text")
            .attr("dy", ".35em")
            //.attr("text-anchor", "end")
            .attr("transform", function (d) {
                var anchorX = d.width0 * n2Diag.transitCoords.previous.model.x - N2Layout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.height0 * n2Diag.transitCoords.previous.model.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity0;
            })
            .text(n2Diag.layout.getText);

        var nodeUpdate = nodeEnter.merge(sel).transition(sharedTransition)
            .attr("class", function (d) {
                return "partition_group " + n2Diag.style.getNodeClass(d);
            })
            .attr("transform", function (d) {
                return "translate(" + n2Diag.scales.model.x(d.x) + "," + n2Diag.scales.model.y(d.y) + ")";
            });

        nodeUpdate.select("rect")
            .attr("width", function (d) {
                return d.width * n2Diag.transitCoords.model.x;
            })
            .attr("height", function (d) {
                return d.height * n2Diag.transitCoords.model.y;
            });

        nodeUpdate.select("text")
            .attr("transform", function (d) {
                var anchorX = d.width * n2Diag.transitCoords.model.x - N2Layout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.height * n2Diag.transitCoords.model.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(n2Diag.layout.getText);


        // Transition exiting nodes to the parent's new position.
        var nodeExit = sel.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + n2Diag.scales.model.x(d.x) + "," + n2Diag.scales.model.y(d.y) + ")";
            })
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                return d.width * n2Diag.transitCoords.model.x;//0;//
            })
            .attr("height", function (d) {
                return d.height * n2Diag.transitCoords.model.y;
            });

        nodeExit.select("text")
            .attr("transform", function (d) {
                var anchorX = d.width * n2Diag.transitCoords.model.x - N2Layout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.height * n2Diag.transitCoords.model.y / 2 + ")";
                //return "translate(8," + d.height * n2Diag.transitCoords.model.y / 2 + ")";
            })
            .style("opacity", 0);


        var selSolver = n2Diag.pSolverTreeGroup.selectAll(".solver_group")
            .data(d3SolverNodesArray, function (d) {
                return d.id;
            });
/*
        function getSolverClass(showLinearSolverNames, linear_solver_name, nonlinear_solver_name) {
            if (showLinearSolverNames) {
                if (linearSolverNames.indexOf(linear_solver_name) >= 0) {
                    solver_class = linearSolverClasses[linear_solver_name]
                } else {
                    solver_class = linearSolverClasses["other"]; // user must have defined their own solver that we do not know about
                }
            } else {
                if (nonLinearSolverNames.indexOf(nonlinear_solver_name) >= 0) {
                    solver_class = nonLinearSolverClasses[nonlinear_solver_name]
                } else {
                    solver_class = nonLinearSolverClasses["other"]; // user must have defined their own solver that we do not know about
                }
            }
            return solver_class;
        }
*/
        var nodeSolverEnter = selSolver.enter().append("svg:g")
            .attr("class", function (d) {
                solver_class = n2Diag.style.getSolverClass(N2Layout.showLinearSolverNames, { 'linear': d.linear_solver, 'nonLinear': d.nonlinear_solver})
                return solver_class + " " + "solver_group " + n2Diag.style.getNodeClass(d);
            })
            .attr("transform", function (d) {
                x = 1.0 - d.xSolver0 - d.widthSolver0; // The magic for reversing the blocks on the right side
                // The solver tree goes from the root on the right and expands to the left
                return "translate(" + n2Diag.scales.previous.solver.x(x) + "," + n2Diag.scales.previous.solver.y(d.ySolver0) + ")";
            })
            .on("click", function (d) { LeftClick(d, this); })
            .on("contextmenu", function (d) { RightClick(d, this); })
            .on("mouseover", function (d) {
                if (abs2prom != undefined) {
                    if (d.type == "param" || d.type == "unconnected_param") {
                        return tooltip.text(abs2prom.input[d.absPathName])
                            .style("visibility", "visible");
                    }
                    if (d.type == "unknown") {
                        return tooltip.text(abs2prom.output[d.absPathName])
                            .style("visibility", "visible");
                    }
                }
            })
            .on("mouseleave", function (d) {
                if (abs2prom != undefined) {
                    return tooltip.style("visibility", "hidden");
                }
            })
            .on("mousemove", function () {
                if (abs2prom != undefined) {
                    return tooltip.style("top", (d3.event.pageY - 30) + "px")
                        .style("left", (d3.event.pageX + 5) + "px");
                }
            });

        nodeSolverEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.widthSolver0 * n2Diag.transitCoords.previous.solver.x;//0;//
            })
            .attr("height", function (d) {
                return d.heightSolver0 * n2Diag.transitCoords.previous.solver.y;
            });

        nodeSolverEnter.append("svg:text")
            .attr("dy", ".35em")
            .attr("transform", function (d) {
                var anchorX = d.widthSolver0 * n2Diag.transitCoords.previous.solver.x - N2Layout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.heightSolver0 * n2Diag.transitCoords.previous.solver.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity0;
            })
            .text(n2Diag.layout.getSolverText);

        var nodeSolverUpdate = nodeSolverEnter.merge(selSolver).transition(sharedTransition)
            .attr("class", function (d) {
                solver_class = n2Diag.style.getSolverClass(N2Layout.showLinearSolverNames, { 'linear': d.linear_solver, 'nonLinear': d.nonlinear_solver});
                return solver_class + " " + "solver_group " + n2Diag.style.getNodeClass(d);
            })
            .attr("transform", function (d) {
                x = 1.0 - d.xSolver - d.widthSolver; // The magic for reversing the blocks on the right side
                return "translate(" + n2Diag.scales.solver.x(x) + "," + n2Diag.scales.solver.y(d.ySolver) + ")";
            });

        nodeSolverUpdate.select("rect")
            .attr("width", function (d) {
                return d.widthSolver * n2Diag.transitCoords.solver.x;
            })
            .attr("height", function (d) {
                return d.heightSolver * n2Diag.transitCoords.solver.y;
            });

        nodeSolverUpdate.select("text")
            .attr("transform", function (d) {
                var anchorX = d.widthSolver * n2Diag.transitCoords.solver.x - N2Layout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.heightSolver * n2Diag.transitCoords.solver.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(n2Diag.layout.getSolverText);


        // Transition exiting nodes to the parent's new position.
        var nodeSolverExit = selSolver.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + n2Diag.scales.solver.x(d.xSolver) + "," + n2Diag.scales.solver.y(d.ySolver) + ")";
            })
            .remove();

        nodeSolverExit.select("rect")
            .attr("width", function (d) {
                return d.widthSolver * n2Diag.transitCoords.solver.x;//0;//
            })
            .attr("height", function (d) {
                return d.heightSolver * n2Diag.transitCoords.solver.y;
            });

        nodeSolverExit.select("text")
            .attr("transform", function (d) {
                var anchorX = d.widthSolver * n2Diag.transitCoords.solver.x - N2Layout.rightTextMarginPx;
                return "translate(" + anchorX + "," + d.heightSolver * n2Diag.transitCoords.solver.y / 2 + ")";
            })
            .style("opacity", 0);


        ClearArrowsAndConnects()
        n2Diag.matrix.draw();
    }

    updateFunc = Update;

    function ClearArrows() {
        n2Diag.n2TopGroup.selectAll("[class^=n2_hover_elements]").remove();
    }

    function ClearArrowsAndConnects() {
        ClearArrows();
        // newConnsDict = {};
        // PrintConnects();
    }



    var lastLeftClickedEle;
    var lastRightClickedEle;
    var lastRightClickedObj;

    //right click => collapse
    function RightClick(d, ele) {
        var e = d3.event;
        lastLeftClickedEle = d;
        lastRightClickedObj = d;
        lastRightClickedEle = ele;
        e.preventDefault();
        collapse();
    }

    var menu = document.querySelector('#context-menu');
    var menuState = 0;
    var contextMenuActive = "context-menu--active";

    function collapse() {
        var d = lastLeftClickedEle;
        if (!d.children) return;
        if (d.depth > zoomedElement.depth) { //dont allow minimizing on root node
            lastRightClickedElement = d;
            FindRootOfChangeFunction = FindRootOfChangeForRightClick;
            TRANSITION_DURATION = TRANSITION_DURATION_FAST;
            lastClickWasLeft = false;
            Toggle(d);
            Update();
        }
    }

    function SetupLeftClick(d) {
        lastLeftClickedElement = d;
        lastClickWasLeft = true;
        if (lastLeftClickedElement.depth > zoomedElement.depth) {
            leftClickIsForward = true; //forward
        }
        else if (lastLeftClickedElement.depth < zoomedElement.depth) {
            leftClickIsForward = false; //backwards
        }
        n2Diag.updateZoomedElement(d);
        TRANSITION_DURATION = TRANSITION_DURATION_FAST;
    }

    //left click => navigate
    function LeftClick(d, ele) {
        if (!d.children) return;
        if (d3.event.button != 0) return;
        n2Diag.backButtonHistory.push({ "el": zoomedElement });
        n2Diag.forwardButtonHistory = [];
        SetupLeftClick(d);
        Update();
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
        n2Diag.forwardButtonHistory.push({ "el": zoomedElement });
        SetupLeftClick(d);
        Update();
    }

    function ForwardButtonPressed() {
        if (n2Diag.forwardButtonHistory.length == 0) return;
        var d = n2Diag.forwardButtonHistory.pop().el;
        parentDiv.querySelector("#forwardButtonId").disabled = (n2Diag.forwardButtonHistory.length == 0) ? "disabled" : false;
        for (var obj = d; obj != null; obj = obj.parent) { //make sure history item is not minimized
            if (obj.isMinimized) return;
        }
        n2Diag.backButtonHistory.push({ "el": zoomedElement });
        SetupLeftClick(d);
        Update();
    }

    function Toggle(d) {

        if (d.isMinimized)
            d.isMinimized = false;
        else
            d.isMinimized = true;
    }

    function FindRootOfChangeForRightClick(d) {
        return lastRightClickedElement;
    }

    function FindRootOfChangeForCollapseDepth(d) {
        for (var obj = d; obj != null; obj = obj.parent) { //make sure history item is not minimized
            if (obj.depth == chosenCollapseDepth) return obj;
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
            if (d.children) {
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
            if (d.children) {
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

        var lineWidth = Math.min(5, n2Dx * .5, n2Dy * .5);
        n2Diag.arrowMarker.attr("markerWidth", lineWidth * .4)
            .attr("markerHeight", lineWidth * .4);
        var src = d3RightTextNodesArrayZoomed[d.row];
        var tgt = d3RightTextNodesArrayZoomed[d.col];
        var boxEnd = d3RightTextNodesArrayZoomedBoxInfo[d.col];

        new N2Arrow({
            start: { col: d.row, row: d.row },
            end: { col: d.col, row: d.col },
            color: N2Style.color.redArrow,
            width: lineWidth
        }, n2Diag.n2Groups);

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
                            for (var mi = 0; mi < d3RightTextNodesArrayZoomed.length; ++mi) {
                                var rtNode = d3RightTextNodesArrayZoomed[mi];
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
                            for (var mi = 0; mi < d3RightTextNodesArrayZoomed.length; ++mi) {
                                var rtNode = d3RightTextNodesArrayZoomed[mi];
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
                                DrawArrowsParamView(firstBeginIndex, firstEndIndex);
                            }
                        }
                    }
                }
            }
        }

        var leftTextWidthR = d3RightTextNodesArrayZoomed[d.row].nameWidthPx,
            leftTextWidthC = d3RightTextNodesArrayZoomed[d.col].nameWidthPx;
        DrawRect(-leftTextWidthR - PTREE_N2_GAP_PX, n2Dy * d.row, leftTextWidthR, n2Dy, N2Style.color.redArrow); //highlight var name
        DrawRect(-leftTextWidthC - PTREE_N2_GAP_PX, n2Dy * d.col, leftTextWidthC, n2Dy, N2Style.color.greenArrow); //highlight var name
    }

    function MouseoverOnDiagN2(d) {
        //d=hovered element
        // console.log('MouseoverOnDiagN2:'); console.log(d);
        var hoveredIndexRC = d.col; //d.x == d.y == row == col
        var leftTextWidthHovered = d3RightTextNodesArrayZoomed[hoveredIndexRC].nameWidthPx;

        // Loop over all elements in the matrix looking for other cells in the same column as
        var lineWidth = Math.min(5, n2Dx * .5, n2Dy * .5);
        n2Diag.arrowMarker.attr("markerWidth", lineWidth * .4)
            .attr("markerHeight", lineWidth * .4);
        DrawRect(-leftTextWidthHovered - PTREE_N2_GAP_PX, n2Dy * hoveredIndexRC, leftTextWidthHovered, n2Dy, N2Style.color.highlightHovered); //highlight hovered
        for (var i = 0; i < d3RightTextNodesArrayZoomed.length; ++i) {
            var leftTextWidthDependency = d3RightTextNodesArrayZoomed[i].nameWidthPx;
            var box = d3RightTextNodesArrayZoomedBoxInfo[i];
            if (n2Diag.matrix.node(hoveredIndexRC, i) !== undefined) { //i is column here
                if (i != hoveredIndexRC) {
                    new N2Arrow({
                        end: { col: i, row: i },
                        start: { col: hoveredIndexRC, row: hoveredIndexRC },
                        color: N2Style.color.greenArrow,
                        width: lineWidth
                    }, n2Diag.n2Groups);
                    DrawRect(-leftTextWidthDependency - PTREE_N2_GAP_PX, n2Dy * i, leftTextWidthDependency, n2Dy, N2Style.color.greenArrow); //highlight var name
                }
            }

            if (n2Diag.matrix.node(i, hoveredIndexRC) !== undefined) { //i is row here
                if (i != hoveredIndexRC) {
                    new N2Arrow({
                        start: { col: i, row: i },
                        end: { col: hoveredIndexRC, row: hoveredIndexRC },
                        color: N2Style.color.redArrow,
                        width: lineWidth
                    }, n2Diag.n2Groups);
                    DrawRect(-leftTextWidthDependency - PTREE_N2_GAP_PX, n2Dy * i, leftTextWidthDependency, n2Dy, N2Style.color.redArrow); //highlight var name
                }
            }
        }
    }

    function MouseoutN2() {
        n2Diag.n2TopGroup.selectAll(".n2_hover_elements").remove();
    }

    function MouseClickN2(d) {
        var newClassName = "n2_hover_elements_" + d.row + "_" + d.col;
        var selection = n2Diag.n2TopGroup.selectAll("." + newClassName);
        if (selection.size() > 0) {
            selection.remove();
        }
        else {
            n2Diag.n2TopGroup.selectAll("path.n2_hover_elements, circle.n2_hover_elements")
                .attr("class", newClassName);
        }
    }

    function ReturnToRootButtonClick() {
        n2Diag.backButtonHistory.push({ "el": zoomedElement });
        n2Diag.forwardButtonHistory = [];
        SetupLeftClick(root);
        Update();
    }

    function UpOneLevelButtonClick() {
        if (zoomedElement === root) return;
        n2Diag.backButtonHistory.push({ "el": zoomedElement });
        n2Diag.forwardButtonHistory = [];
        SetupLeftClick(zoomedElement.parent);
        Update();
    }

    function CollapseOutputsButtonClick(startNode) {
        function CollapseOutputs(d) {
            if (d.subsystem_type && d.subsystem_type === "component") {
                d.isMinimized = true;
            }
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    CollapseOutputs(d.children[i]);
                }
            }
        }
        FindRootOfChangeFunction = FindRootOfChangeForCollapseUncollapseOutputs;
        TRANSITION_DURATION = TRANSITION_DURATION_SLOW;
        lastClickWasLeft = false;
        CollapseOutputs(startNode);
        Update();
    }

    function UncollapseButtonClick(startNode) {
        function Uncollapse(d) {
            if (d.type !== "param" && d.type !== "unconnected_param") {
                d.isMinimized = false;
            }
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    Uncollapse(d.children[i]);
                }
            }
        }
        FindRootOfChangeFunction = FindRootOfChangeForCollapseUncollapseOutputs;
        TRANSITION_DURATION = TRANSITION_DURATION_SLOW;
        lastClickWasLeft = false;
        Uncollapse(startNode);
        Update();
    }

    function CollapseToDepthSelectChange(newChosenCollapseDepth) {
        function CollapseToDepth(d, depth) {
            if (d.type === "param" || d.type === "unknown" || d.type === "unconnected_param") {
                return;
            }
            if (d.depth < depth) {
                d.isMinimized = false;
            }
            else {
                d.isMinimized = true;
            }
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    CollapseToDepth(d.children[i], depth);
                }
            }
        }

        chosenCollapseDepth = newChosenCollapseDepth;
        if (chosenCollapseDepth > zoomedElement.depth) {
            CollapseToDepth(root, chosenCollapseDepth);
        }
        FindRootOfChangeFunction = FindRootOfChangeForCollapseDepth;
        TRANSITION_DURATION = TRANSITION_DURATION_SLOW;
        lastClickWasLeft = false;
        Update();
    }

    function FontSizeSelectChange(fontSize) {
        for (var i = 8; i <= 14; ++i) {
            var newText = (i == fontSize) ? ("<b>" + i + "px</b>") : (i + "px");
            parentDiv.querySelector("#idFontSize" + i + "px").innerHTML = newText;
        }
        N2Layout.fontSizePx = fontSize;
        TRANSITION_DURATION = TRANSITION_DURATION_FAST;
        n2Diag.style.updateSvgStyle(fontSize);
        Update();
    }

    function VerticalResize(height) {
        for (var i = 600; i <= 1000; i += 50) {
            var newText = (i == height) ? ("<b>" + i + "px</b>") : (i + "px");
            parentDiv.querySelector("#idVerticalResize" + i + "px").innerHTML = newText;
        }
        for (var i = 2000; i <= 4000; i += 1000) {
            var newText = (i == height) ? ("<b>" + i + "px</b>") : (i + "px");
            parentDiv.querySelector("#idVerticalResize" + i + "px").innerHTML = newText;
        }
        ClearArrowsAndConnects();
        N2Layout.heightPx = height;
        HEIGHT_PX = height;
        n2Diag.matrix.updateLevelOfDetailThreshold(height);
        WIDTH_N2_PX = height;
        TRANSITION_DURATION = TRANSITION_DURATION_FAST;
        n2Diag.style.updateSvgStyle(N2Layout.fontSizePx);
        Update();
    }

    function ToggleSolverNamesCheckboxChange() {
        N2Layout.toggleSolverNameType();
        // showLinearSolverNames = !showLinearSolverNames;
        parentDiv.querySelector("#toggleSolverNamesButtonId").className = !N2Layout.showLinearSolverNames ? "myButton myButtonToggledOn" : "myButton";
        SetupLegend(d3, n2Diag.d3ContentDiv);
        Update();
    }

    function ShowPathCheckboxChange() {
        showPath = !showPath;
        parentDiv.querySelector("#currentPathId").style.display = showPath ? "block" : "none";
        parentDiv.querySelector("#showCurrentPathButtonId").className = showPath ? "myButton myButtonToggledOn" : "myButton";
    }

    function ToggleLegend() {
        showLegend = !showLegend;
        parentDiv.querySelector("#showLegendButtonId").className = showLegend ? "myButton myButtonToggledOn" : "myButton";
        SetupLegend(d3, n2Diag.d3ContentDiv);
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
        div.querySelector("#uncollapseInViewButtonId").onclick = function () { UncollapseButtonClick(zoomedElement); };
        div.querySelector("#uncollapseAllButtonId").onclick = function () { UncollapseButtonClick(root); };
        div.querySelector("#collapseInViewButtonId").onclick = function () { CollapseOutputsButtonClick(zoomedElement); };
        div.querySelector("#collapseAllButtonId").onclick = function () { CollapseOutputsButtonClick(root); };
        div.querySelector("#clearArrowsAndConnectsButtonId").onclick = ClearArrowsAndConnects;
        div.querySelector("#showCurrentPathButtonId").onclick = ShowPathCheckboxChange;
        div.querySelector("#showLegendButtonId").onclick = ToggleLegend;

        div.querySelector("#toggleSolverNamesButtonId").onclick = ToggleSolverNamesCheckboxChange;

        for (var i = 8; i <= 14; ++i) {
            var f = function (idx) {
                return function () { FontSizeSelectChange(idx); };
            }(i);
            div.querySelector("#idFontSize" + i + "px").onclick = f;
        }

        for (var i = 600; i <= 1000; i += 50) {
            var f = function (idx) {
                return function () { VerticalResize(idx); };
            }(i);
            div.querySelector("#idVerticalResize" + i + "px").onclick = f;
        }
        for (var i = 2000; i <= 4000; i += 1000) {
            var f = function (idx) {
                return function () { VerticalResize(idx); };
            }(i);
            div.querySelector("#idVerticalResize" + i + "px").onclick = f;
        }

        div.querySelector("#saveSvgButtonId").onclick = n2Diag.saveSvg.bind(n2Diag);
        div.querySelector("#helpButtonId").onclick = DisplayModal;
    }

    return {
        GetFontSize: function () { return N2Layout.fontSizePx; },
        ResizeHeight: function (h) { VerticalResize(h); },
        Redraw: function () { Update(); }
    };
}

var zoomedElement = modelData.tree;
var updateFunc;
var mouseOverOffDiagN2;
var mouseOverOnDiagN2;
var mouseOutN2;
var mouseClickN2;
var treeData, connectionList;

let n2Diag = new N2Diagram(modelData);
var app = PtN2Diagram(document.getElementById("ptN2ContentDivId"), modelData);