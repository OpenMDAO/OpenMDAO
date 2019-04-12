function PtN2Diagram(parentDiv, modelData) {
    var root = modelData.tree;
    var conns = modelData.connections_list;
    var abs2prom = modelData.hasOwnProperty("abs2prom") ? modelData.abs2prom : undefined;

    var FONT_SIZE_PX = 11;
    var svgStyleElement = document.createElement("style");
    var outputNamingType = "Absolute";
    var showPath = false; //default off

    var DEFAULT_TRANSITION_START_DELAY = 100;
    var transitionStartDelay = DEFAULT_TRANSITION_START_DELAY;
    var idCounter = 0;
    var maxDepth = 1;

    var maxSystemDepth = 1; // For use with the right hand side solver tree. Only want max depth of solvers, not params,..

    var RIGHT_TEXT_MARGIN_PX = 8; // How much space in px (left and) right of text in partition tree

    //N^2 vars
    var backButtonHistory = [], forwardButtonHistory = [];
    var chosenCollapseDepth = -1;
    var updateRecomputesAutoComplete = true; //default

    var katexInputDivElement = document.getElementById("katexInputDiv");
    var katexInputElement = document.getElementById("katexInput");

    var tooltip = d3.select("body").append("div").attr("class", "tool-tip")
        .style("position", "absolute")
        .style("visibility", "hidden");

    mouseOverOnDiagN2 = MouseoverOnDiagN2;
    mouseOverOffDiagN2 = MouseoverOffDiagN2;
    mouseClickN2 = MouseClickN2;
    mouseOutN2 = MouseoutN2;

    CreateDomLayout();
    CreateToolbar();

    setD3ContentDiv();

    parentDiv.querySelector("#svgId").appendChild(svgStyleElement);
    UpdateSvgCss(svgStyleElement, FONT_SIZE_PX);

    arrowMarker = svg.append("svg:defs").append("svg:marker");

    arrowMarker
        .attr("id", "arrow")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 5)
        .attr("refY", 0)
        .attr("markerWidth", 1)
        .attr("markerHeight", 1)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("class", "arrowHead");

    setN2Group();
    var pTreeGroup = svg.append("g").attr("id", "tree"); // id given just so it is easier to see in Chrome dev tools when debugging
    var pSolverTreeGroup = svg.append("g").attr("id", "solver_tree");

    function updateRootTypes() {
        if (!showParams) return;

        var stack = []
        for (var i = 0; i < root.children.length; ++i) {
            stack.push(root.children[i]);
        }

        while (stack.length > 0) {
            var cur_ele = stack.pop();
            if (cur_ele.type === "param") {
                if (!hasInputConnection(cur_ele.absPathName) && !hasOutputConnection(cur_ele.absPathName)) {
                    cur_ele.type = "unconnected_param";
                }
            }

            if (cur_ele.hasOwnProperty('children')) {
                for (var j = 0; j < cur_ele.children.length; ++j) {
                    stack.push(cur_ele.children[j]);
                }
            }
        }
    }

    function hasInputConnection(target) {
        for (i = 0; i < conns.length; ++i) {
            if (conns[i].tgt === target) {
                return true;
            }
        }

        return false;
    }

    function hasOutputConnection(target) {
        for (i = 0; i < conns.length; ++i) {
            if (conns[i].src === target) {
                return true;
            }
        }
    }
    hasInputConn = hasInputConnection;

    var n2BackgroundRectR0 = -1, n2BackgroundRectC0 = -1;
    var newConnsDict = {};
    function PrintConnects() {
        var text = "Connections:";
        for (var key in newConnsDict) {
            var d = newConnsDict[key];
            var param = d3RightTextNodesArrayZoomed[d.c],
                unknown = d3RightTextNodesArrayZoomed[d.r];
            var paramName = (zoomedElement.promotions && zoomedElement.promotions[param.absPathName] !== undefined) ?
                "<b>" + zoomedElement.promotions[param.absPathName] + "</b>" :
                ((zoomedElement === root) ? param.absPathName : param.absPathName.slice(zoomedElement.absPathName.length + 1));
            var unknownName = (zoomedElement.promotions && zoomedElement.promotions[unknown.absPathName] !== undefined) ?
                "<b>" + zoomedElement.promotions[unknown.absPathName] + "</b>" :
                ((zoomedElement === root) ? unknown.absPathName : unknown.absPathName.slice(zoomedElement.absPathName.length + 1));

            text += "<br />self.connect(\"" + unknownName + "\", \"" + paramName + "\")";
        }
        parentDiv.querySelector("#connectionId").innerHTML = "";
    }
    var n2BackgroundRect = n2Group.append("rect")
        .attr("class", "background")
        .attr("width", WIDTH_N2_PX)
        .attr("height", HEIGHT_PX)
        .on("click", function () {
            if (!showParams) return;
            var coords = d3.mouse(this);
            var c = Math.floor(coords[0] * d3RightTextNodesArrayZoomed.length / WIDTH_N2_PX);
            var r = Math.floor(coords[1] * d3RightTextNodesArrayZoomed.length / HEIGHT_PX);
            if (r == c || r < 0 || c < 0 || r >= d3RightTextNodesArrayZoomed.length || c >= d3RightTextNodesArrayZoomed.length) return;
            if (matrix[r + "_" + c] !== undefined) return;

            var param = d3RightTextNodesArrayZoomed[c],
                unknown = d3RightTextNodesArrayZoomed[r];
            if (param.type !== "param" && unknown.type !== "unknown") return;

            var newClassName = "n2_hover_elements_" + r + "_" + c;
            var selection = n2Group.selectAll("." + newClassName);
            if (selection.size() > 0) {
                delete newConnsDict[r + "_" + c];
                selection.remove();
            }
            else {
                newConnsDict[r + "_" + c] = { "r": r, "c": c };
                n2Group.selectAll("path.n2_hover_elements, circle.n2_hover_elements")
                    .attr("class", newClassName);
            }
            PrintConnects();
        })
        .on("mouseover", function () {
            n2BackgroundRectR0 = -1;
            n2BackgroundRectC0 = -1;
            n2Group.selectAll(".n2_hover_elements").remove();
            PrintConnects();
        })
        .on("mouseleave", function () {
            n2BackgroundRectR0 = -1;
            n2BackgroundRectC0 = -1;
            n2Group.selectAll(".n2_hover_elements").remove();
            PrintConnects();
        })
        .on("mousemove", function () {
            if (!showParams) return;
            var coords = d3.mouse(this);
            var c = Math.floor(coords[0] * d3RightTextNodesArrayZoomed.length / WIDTH_N2_PX);
            var r = Math.floor(coords[1] * d3RightTextNodesArrayZoomed.length / HEIGHT_PX);
            if (r == c || r < 0 || c < 0 || r >= d3RightTextNodesArrayZoomed.length || c >= d3RightTextNodesArrayZoomed.length) return;
            if (matrix[r + "_" + c] !== undefined) return;
            if (n2BackgroundRectR0 == r && n2BackgroundRectC0 == c) return;
            //n2Group.selectAll(".n2_hover_elements_" + n2BackgroundRectR0 + "_" + n2BackgroundRectC0).remove();
            n2Group.selectAll(".n2_hover_elements").remove();
            n2BackgroundRectR0 = r;
            n2BackgroundRectC0 = c;

            var lineWidth = Math.min(5, n2Dx * .5, n2Dy * .5);
            arrowMarker.attr("markerWidth", lineWidth * .4)
                .attr("markerHeight", lineWidth * .4);

            var param = d3RightTextNodesArrayZoomed[c],
                unknown = d3RightTextNodesArrayZoomed[r];
            if (param.type !== "param" && unknown.type !== "unknown") return;
            var leftTextWidthR = d3RightTextNodesArrayZoomed[r].nameWidthPx,
                leftTextWidthC = d3RightTextNodesArrayZoomed[c].nameWidthPx;
            DrawRect(-leftTextWidthR - PTREE_N2_GAP_PX, n2Dy * r, leftTextWidthR, n2Dy, "blue"); //highlight var name
            DrawRect(-leftTextWidthC - PTREE_N2_GAP_PX, n2Dy * c, leftTextWidthC, n2Dy, "blue"); //highlight var name

            PrintConnects();

            if (newConnsDict[r + "_" + c] === undefined) {
                var paramName = (zoomedElement.promotions && zoomedElement.promotions[param.absPathName] !== undefined) ?
                    "<b>" + zoomedElement.promotions[param.absPathName] + "</b>" :
                    ((zoomedElement === root) ? param.absPathName : param.absPathName.slice(zoomedElement.absPathName.length + 1));
                var unknownName = (zoomedElement.promotions && zoomedElement.promotions[unknown.absPathName] !== undefined) ?
                    "<b>" + zoomedElement.promotions[unknown.absPathName] + "</b>" :
                    ((zoomedElement === root) ? unknown.absPathName : unknown.absPathName.slice(zoomedElement.absPathName.length + 1));

            }
        });

    setN2ElementsGroup();
    var zoomedElement0 = root;
    var lastRightClickedElement = root;

    ExpandColonVars(root);
    FlattenColonGroups(root);
    InitTree(root, null, 1);
    updateRootTypes();
    ComputeLayout();
    ComputeConnections();
    ComputeMatrixN2();

    var collapseDepthElement = parentDiv.querySelector("#idCollapseDepthDiv");
    for (var i = 2; i <= maxDepth; ++i) {
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

    Update();
    SetupLegend(d3, d3ContentDiv);

    function Update() {
        parentDiv.querySelector("#currentPathId").innerHTML = "PATH: root" + ((zoomedElement.parent) ? "." : "") + zoomedElement.absPathName;

        parentDiv.querySelector("#backButtonId").disabled = (backButtonHistory.length == 0) ? "disabled" : false;
        parentDiv.querySelector("#forwardButtonId").disabled = (forwardButtonHistory.length == 0) ? "disabled" : false;
        parentDiv.querySelector("#upOneLevelButtonId").disabled = (zoomedElement === root) ? "disabled" : false;
        parentDiv.querySelector("#returnToRootButtonId").disabled = (zoomedElement === root) ? "disabled" : false;

        // Compute the new tree layout.
        ComputeLayout(); //updates d3NodesArray
        ComputeMatrixN2();

        for (var i = 2; i <= maxDepth; ++i) {
            parentDiv.querySelector("#idCollapseDepthOption" + i + "").style.display = (i <= zoomedElement.depth) ? "none" : "block";
        }

        if (xScalerPTree0 != null) {//not first run.. store previous
            kx0 = kx;
            ky0 = ky;
            xScalerPTree0 = xScalerPTree.copy();
            yScalerPTree0 = yScalerPTree.copy();

            kxSolver0 = kxSolver;
            kySolver0 = kySolver;
            xScalerPSolverTree0 = xScalerPSolverTree.copy();
            yScalerPSolverTree0 = yScalerPSolverTree.copy();
        }

        kx = (zoomedElement.x ? widthPTreePx - PARENT_NODE_WIDTH_PX : widthPTreePx) / (1 - zoomedElement.x);
        ky = HEIGHT_PX / zoomedElement.height;
        xScalerPTree.domain([zoomedElement.x, 1]).range([zoomedElement.x ? PARENT_NODE_WIDTH_PX : 0, widthPTreePx]);
        yScalerPTree.domain([zoomedElement.y, zoomedElement.y + zoomedElement.height]).range([0, HEIGHT_PX]);

        kxSolver = (zoomedElement.xSolver ? widthPSolverTreePx - PARENT_NODE_WIDTH_PX : widthPSolverTreePx) / (1 - zoomedElement.xSolver);
        kySolver = HEIGHT_PX / zoomedElement.heightSolver;
        xScalerPSolverTree.domain([zoomedElement.xSolver, 1]).range([zoomedElement.xSolver ? PARENT_NODE_WIDTH_PX : 0, widthPSolverTreePx]);
        yScalerPSolverTree.domain([zoomedElement.ySolver, zoomedElement.ySolver + zoomedElement.heightSolver]).range([0, HEIGHT_PX]);

        if (xScalerPTree0 == null) { //first run.. duplicate
            kx0 = kx;
            ky0 = ky;
            xScalerPTree0 = xScalerPTree.copy();
            yScalerPTree0 = yScalerPTree.copy();

            kxSolver0 = kxSolver;
            kySolver0 = kySolver;
            xScalerPSolverTree0 = xScalerPSolverTree.copy();
            yScalerPSolverTree0 = yScalerPSolverTree.copy();

            //Update svg dimensions before ComputeLayout() changes widthPTreePx
            svgDiv.style("width", (widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + widthPSolverTreePx + 2 * SVG_MARGIN + PTREE_N2_GAP_PX) + "px")
                  .style("height", (HEIGHT_PX + 2 * SVG_MARGIN) + "px");
            svg.attr("width", widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + widthPSolverTreePx + 2 * SVG_MARGIN + PTREE_N2_GAP_PX)
               .attr("height", HEIGHT_PX + 2 * SVG_MARGIN);

            n2Group.attr("transform", "translate(" + (widthPTreePx + PTREE_N2_GAP_PX + SVG_MARGIN) + "," + SVG_MARGIN + ")");
            pTreeGroup.attr("transform", "translate(" + SVG_MARGIN + "," + SVG_MARGIN + ")");

            pSolverTreeGroup.attr("transform", "translate(" + (widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + SVG_MARGIN + PTREE_N2_GAP_PX) + "," + SVG_MARGIN + ")");
        }

        sharedTransition = d3.transition().duration(TRANSITION_DURATION).delay(transitionStartDelay); //do this after intense computation
        transitionStartDelay = DEFAULT_TRANSITION_START_DELAY;

        //Update svg dimensions with transition after ComputeLayout() changes widthPTreePx
        svgDiv.transition(sharedTransition).style("width", (widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + widthPSolverTreePx + 2 * SVG_MARGIN + PTREE_N2_GAP_PX) + "px")
            .style("height", (HEIGHT_PX + 2 * SVG_MARGIN) + "px");
        svg.transition(sharedTransition).attr("width", widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + widthPSolverTreePx + 2 * SVG_MARGIN + PTREE_N2_GAP_PX)
            .attr("height", HEIGHT_PX + 2 * SVG_MARGIN);

        n2Group.transition(sharedTransition).attr("transform", "translate(" + (widthPTreePx + PTREE_N2_GAP_PX + SVG_MARGIN) + "," + SVG_MARGIN + ")");
        pTreeGroup.transition(sharedTransition).attr("transform", "translate(" + SVG_MARGIN + "," + SVG_MARGIN + ")");
        n2BackgroundRect.transition(sharedTransition).attr("width", WIDTH_N2_PX).attr("height", HEIGHT_PX);

        pSolverTreeGroup.transition(sharedTransition).attr("transform", "translate(" + (widthPTreePx + PTREE_N2_GAP_PX + WIDTH_N2_PX + SVG_MARGIN + PTREE_N2_GAP_PX) + "," + SVG_MARGIN + ")");

        var sel = pTreeGroup.selectAll(".partition_group")
            .data(d3NodesArray, function (d) {
                return d.id;
            });

        var nodeEnter = sel.enter().append("svg:g")
            .attr("class", function (d) {
                return "partition_group " + GetClass(d);
            })
            .attr("transform", function (d) {
                return "translate(" + xScalerPTree0(d.x0) + "," + yScalerPTree0(d.y0) + ")";
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
            .on("mousemove", function(){
                if (abs2prom != undefined) {
                    return tooltip.style("top", (d3.event.pageY-30)+"px")
                                  .style("left",(d3.event.pageX+5)+"px");
                }
            });

        nodeEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.width0 * kx0;//0;//
            })
            .attr("height", function (d) {
                return d.height0 * ky0;
            });

        nodeEnter.append("svg:text")
            .attr("dy", ".35em")
            //.attr("text-anchor", "end")
            .attr("transform", function (d) {
                var anchorX = d.width0 * kx0 - RIGHT_TEXT_MARGIN_PX;
                //var anchorX = -RIGHT_TEXT_MARGIN_PX;
                return "translate(" + anchorX + "," + d.height0 * ky0 / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity0;
            })
            .text(GetText);

        var nodeUpdate = nodeEnter.merge(sel).transition(sharedTransition)
            .attr("class", function (d) {
                return "partition_group " + GetClass(d);
            })
            .attr("transform", function (d) {
                return "translate(" + xScalerPTree(d.x) + "," + yScalerPTree(d.y) + ")";
            });

        nodeUpdate.select("rect")
            .attr("width", function (d) {
                return d.width * kx;
            })
            .attr("height", function (d) {
                return d.height * ky;
            });

        nodeUpdate.select("text")
            .attr("transform", function (d) {
                var anchorX = d.width * kx - RIGHT_TEXT_MARGIN_PX;
                return "translate(" + anchorX + "," + d.height * ky / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(GetText);


        // Transition exiting nodes to the parent's new position.
        var nodeExit = sel.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + xScalerPTree(d.x) + "," + yScalerPTree(d.y) + ")";
            })
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                return d.width * kx;//0;//
            })
            .attr("height", function (d) {
                return d.height * ky;
            });

        nodeExit.select("text")
            .attr("transform", function (d) {
                var anchorX = d.width * kx - RIGHT_TEXT_MARGIN_PX;
                return "translate(" + anchorX + "," + d.height * ky / 2 + ")";
                //return "translate(8," + d.height * ky / 2 + ")";
            })
            .style("opacity", 0);


       var selSolver = pSolverTreeGroup.selectAll(".solver_group")
            .data(d3SolverNodesArray, function (d) {
                return d.id;
            });

        function getSolverClass(showLinearSolverNames, linear_solver_name, nonlinear_solver_name){
            if (showLinearSolverNames){
                if (linearSolverNames.indexOf(linear_solver_name) >= 0){
                    solver_class = linearSolverClasses[linear_solver_name]
                } else {
                    solver_class = linearSolverClasses["other"]; // user must have defined their own solver that we do not know about
                }
            } else {
                if (nonLinearSolverNames.indexOf(nonlinear_solver_name) >= 0){
                    solver_class = nonLinearSolverClasses[nonlinear_solver_name]
                } else {
                    solver_class = nonLinearSolverClasses["other"]; // user must have defined their own solver that we do not know about
                }
            }
            return solver_class;
        }

        var nodeSolverEnter = selSolver.enter().append("svg:g")
            .attr("class", function (d) {
                solver_class = getSolverClass(showLinearSolverNames, d.linear_solver, d.nonlinear_solver) ;
                return solver_class + " " + "solver_group " + GetClass(d) ;
            })
            .attr("transform", function (d) {
                x = 1.0 - d.xSolver0 - d.widthSolver0; // The magic for reversing the blocks on the right side
                // The solver tree goes from the root on the right and expands to the left
                return "translate(" + xScalerPSolverTree0(x) + "," + yScalerPSolverTree0(d.ySolver0) + ")";
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
            .on("mousemove", function(){
                if (abs2prom != undefined) {
                    return tooltip.style("top", (d3.event.pageY-30)+"px")
                                  .style("left",(d3.event.pageX+5)+"px");
                }
            });

        nodeSolverEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.widthSolver0 * kxSolver0;//0;//
            })
            .attr("height", function (d) {
                return d.heightSolver0 * kySolver0;
            });

        nodeSolverEnter.append("svg:text")
            .attr("dy", ".35em")
            .attr("transform", function (d) {
                var anchorX = d.widthSolver0 * kxSolver0 - RIGHT_TEXT_MARGIN_PX;
                return "translate(" + anchorX + "," + d.heightSolver0 * kySolver0 / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity0;
            })
            .text(GetSolverText);

        var nodeSolverUpdate = nodeSolverEnter.merge(selSolver).transition(sharedTransition)
            .attr("class", function (d) {
                solver_class = getSolverClass(showLinearSolverNames, d.linear_solver, d.nonlinear_solver) ;
                return solver_class + " " + "solver_group " + GetClass(d) ;
            })
            .attr("transform", function (d) {
                x = 1.0 - d.xSolver - d.widthSolver; // The magic for reversing the blocks on the right side
                return "translate(" + xScalerPSolverTree(x) + "," + yScalerPSolverTree(d.ySolver) + ")";
            });

        nodeSolverUpdate.select("rect")
            .attr("width", function (d) {
                return d.widthSolver * kxSolver;
            })
            .attr("height", function (d) {
                return d.heightSolver * kySolver;
            });

        nodeSolverUpdate.select("text")
            .attr("transform", function (d) {
                var anchorX = d.widthSolver * kxSolver - RIGHT_TEXT_MARGIN_PX;
                return "translate(" + anchorX + "," + d.heightSolver * kySolver / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(GetSolverText);


        // Transition exiting nodes to the parent's new position.
        var nodeSolverExit = selSolver.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + xScalerPSolverTree(d.xSolver) + "," + yScalerPSolverTree(d.ySolver) + ")";
            })
            .remove();

        nodeSolverExit.select("rect")
            .attr("width", function (d) {
                return d.widthSolver * kxSolver;//0;//
            })
            .attr("height", function (d) {
                return d.heightSolver * kySolver;
            });

        nodeSolverExit.select("text")
            .attr("transform", function (d) {
                var anchorX = d.widthSolver * kxSolver - RIGHT_TEXT_MARGIN_PX;
                return "translate(" + anchorX + "," + d.heightSolver * kySolver / 2 + ")";
            })
            .style("opacity", 0);


        ClearArrowsAndConnects()
        DrawMatrix();
    }

    updateFunc = Update;

    function ClearArrows() {
        n2Group.selectAll("[class^=n2_hover_elements]").remove();
    }

    function ClearArrowsAndConnects() {
        ClearArrows();
        newConnsDict = {};
        PrintConnects();
    }

    function ExpandColonVars(d) {
        function findNameInIndex(arr, name) {
            for (var i = 0; i < arr.length; ++i) {
                if (arr[i].name === name) return i;
            }
            return -1;
        }

        function addChildren(originalParent, parent, arrayOfNames, arrayOfNamesIndex, type) {
            if (arrayOfNames.length == arrayOfNamesIndex) return;

            var name = arrayOfNames[arrayOfNamesIndex];

            if (!parent.hasOwnProperty("children")) {
                parent.children = [];
            }

            var parentI = findNameInIndex(parent.children, name);
            if (parentI == -1) { //new name not found in parent, create new
                var newObj = {
                    "name": name,
                    "type": type,
                    "splitByColon": true,
                    "originalParent": originalParent
                };
                if (type === "param" && type === "unconnected_param") {
                    parent.children.splice(0, 0, newObj);
                }
                else {
                    parent.children.push(newObj);
                }
                addChildren(originalParent, newObj, arrayOfNames, arrayOfNamesIndex + 1, type);
            } else { //new name already found in parent, keep traversing
                addChildren(originalParent, parent.children[parentI], arrayOfNames, arrayOfNamesIndex + 1, type);
            }
        }

        if (!d.children) return;
        for (var i = 0; i < d.children.length; ++i) {

            var splitArray = d.children[i].name.split(":");
            if (splitArray.length > 1) {
                if (!d.hasOwnProperty("subsystem_type") || d.subsystem_type !== "component") {
                    alert("error: there is a colon named object whose parent is not a component");
                    return;
                }
                var type = d.children[i].type;
                d.children.splice(i--, 1);
                addChildren(d, d, splitArray, 0, type);
            }
        }
        for (var i = 0; i < d.children.length; ++i) {
            ExpandColonVars(d.children[i]);
        }
    }

    function FlattenColonGroups(d) {
        if (!d.children) return;
        while (d.splitByColon && d.children && d.children.length == 1 && d.children[0].splitByColon) {
            //alert("combine " + d.name + " " + d.children[0].name);
            var child = d.children[0];
            d.name += ":" + child.name;
            d.children = (child.hasOwnProperty("children") && child.children.length >= 1) ? child.children : null; //absorb childs children
            if (d.children == null) delete d.children;
        }
        if (!d.children) return;
        for (var i = 0; i < d.children.length; ++i) {
            FlattenColonGroups(d.children[i]);
        }
    }

    function GetText(d) {
        var retVal = d.name;
        if (outputNamingType === "Promoted" && (d.type === "unknown" || d.type === "param" || d.type === "unconnected_param") && zoomedElement.promotions && zoomedElement.promotions[d.absPathName] !== undefined) {
            retVal = zoomedElement.promotions[d.absPathName];
        }
        if (d.splitByColon && d.children && d.children.length > 0) retVal += ":";
        return retVal;
    }

    function GetSolverText(d) {
        var retVal = showLinearSolverNames ? d.linear_solver : d.nonlinear_solver;
        return retVal;
    }

    //Sets parents, depth, and nameWidthPx of all nodes.  Also finds and sets maxDepth.
    function InitTree(d, parent, depth) {
        d.numLeaves = 0; //for nested params
        d.depth = depth;
        d.parent = parent;
        d.id = ++idCounter; //id starts at 1 for if comparision
        d.absPathName = "";
        if (d.parent) { //not root node? d.parent.absPathName : "";
            if (d.parent.absPathName !== "") {
                d.absPathName += d.parent.absPathName;
                d.absPathName += (d.parent.splitByColon) ? ":" : ".";
            }
            d.absPathName += d.name;
        }
        if (d.type === "unknown" || d.type === "param" || d.type === "unconnected_param") {
            var parentComponent = (d.originalParent) ? d.originalParent : d.parent;
            if (parentComponent.type === "subsystem" && parentComponent.subsystem_type === "component") {
                d.parentComponent = parentComponent;
            }
            else {
                alert("error: there is a param or unknown without a parent component!");
            }
        }
        if (d.splitByColon) {
            d.colonName = d.name;
            for (var obj = d.parent; obj.splitByColon; obj = obj.parent) {
                d.colonName = obj.name + ":" + d.colonName;
            }
        }
        maxDepth = Math.max(depth, maxDepth);

        if (d.type === "subsystem") {
            maxSystemDepth = Math.max(depth, maxSystemDepth);
        }

        if (d.children) {
            for (var i = 0; i < d.children.length; ++i) {
                var implicit = InitTree(d.children[i], d, depth + 1);
                if (implicit) {
                    d.implicit = true;
                }
            }
        }
        return (d.implicit) ? true : false;
    }


    function ComputeLayout() {
        var columnWidthsPx = new Array(maxDepth + 1).fill(0.0),// since depth is one based
            columnLocationsPx = new Array(maxDepth + 1).fill(0.0);

        var columnSolverWidthsPx = new Array(maxDepth + 1).fill(0.0),// since depth is one based
            columnSolverLocationsPx = new Array(maxDepth + 1).fill(0.0);

        var textWidthGroup = svg.append("svg:g").attr("class", "partition_group");
        var textWidthText = textWidthGroup.append("svg:text")
            .text("")
            .attr("x", -50); // Put text off screen
        var textWidthTextNode = textWidthText.node();

        var autoCompleteSetNames = {}, autoCompleteSetPathNames = {};

        function PopulateAutoCompleteList(d) {
            if (d.children && !d.isMinimized) { //depth first, dont go into minimized children
                for (var i = 0; i < d.children.length; ++i) {
                    PopulateAutoCompleteList(d.children[i]);
                }
            }
            if (d === zoomedElement) return;
            if (!showParams && (d.type === "param" || d.type === "unconnected_param")) return;

            var n = d.name;
            if (d.splitByColon && d.children && d.children.length > 0) n += ":";
            if ((d.type !== "param" && d.type !== "unconnected_param") && d.type !== "unknown") n += ".";
            var namesToAdd = [n];

            if (d.splitByColon) namesToAdd.push(d.colonName + ((d.children && d.children.length > 0) ? ":" : ""));

            namesToAdd.forEach(function (name) {
                if (!autoCompleteSetNames.hasOwnProperty(name)) {
                    autoCompleteSetNames[name] = true;
                    autoCompleteListNames.push(name);
                }
            });

            var localPathName = (zoomedElement === root) ? d.absPathName : d.absPathName.slice(zoomedElement.absPathName.length + 1);
            if (!autoCompleteSetPathNames.hasOwnProperty(localPathName)) {
                autoCompleteSetPathNames[localPathName] = true;
                autoCompleteListPathNames.push(localPathName);
            }
        }

        function GetTextWidth(s) {
            textWidthText.text(s);
            return textWidthTextNode.getBoundingClientRect().width;
        }

        function UpdateTextWidths(d) {
            if ((!showParams && (d.type === "param" || d.type === "unconnected_param")) || d.varIsHidden) return;
            d.nameWidthPx = GetTextWidth(GetText(d)) + 2 * RIGHT_TEXT_MARGIN_PX;
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    UpdateTextWidths(d.children[i]);
                }
            }
        }

        function UpdateSolverTextWidths(d) {
            if ((d.type === "param" || d.type === "unconnected_param") || d.varIsHidden) return;
            d.nameSolverWidthPx = GetTextWidth(GetSolverText(d)) + 2 * RIGHT_TEXT_MARGIN_PX;
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    UpdateSolverTextWidths(d.children[i]);
                }
            }
        }

        function ComputeColumnWidths(d) {
            var greatestDepth = 0;
            var leafWidthsPx = new Array(maxDepth + 1).fill(0.0);

            function DoComputeColumnWidths(d) {
                if ((!showParams && (d.type === "param" || d.type === "unconnected_param")) || d.varIsHidden) return;

                var heightPx = HEIGHT_PX * d.numLeaves / zoomedElement.numLeaves;
                d.textOpacity0 = d.hasOwnProperty('textOpacity') ? d.textOpacity : 0;
                d.textOpacity = (heightPx > FONT_SIZE_PX) ? 1 : 0;
                var hasVisibleDetail = (heightPx >= 2.0);
                var widthPx = 1e-3;
                if (hasVisibleDetail) widthPx = MIN_COLUMN_WIDTH_PX;
                if (d.textOpacity > 0.5) widthPx = d.nameWidthPx;

                greatestDepth = Math.max(greatestDepth, d.depth);

                if (d.children && !d.isMinimized) { //not leaf
                    columnWidthsPx[d.depth] = Math.max(columnWidthsPx[d.depth], widthPx);
                    for (var i = 0; i < d.children.length; ++i) {
                        DoComputeColumnWidths(d.children[i]);
                    }
                }
                else { //leaf
                    leafWidthsPx[d.depth] = Math.max(leafWidthsPx[d.depth], widthPx);
                }
            }

            DoComputeColumnWidths(d);


            var sum = 0;
            var lastColumnWidth = 0;
            for (var i = leafWidthsPx.length - 1; i >= zoomedElement.depth; --i) {
                sum += columnWidthsPx[i];
                var lastWidthNeeded = leafWidthsPx[i] - sum;
                lastColumnWidth = Math.max(lastWidthNeeded, lastColumnWidth);
            }
            columnWidthsPx[zoomedElement.depth - 1] = PARENT_NODE_WIDTH_PX;
            columnWidthsPx[greatestDepth] = lastColumnWidth;

        }

        function ComputeSolverColumnWidths(d) {
            var greatestDepth = 0;
            var leafSolverWidthsPx = new Array(maxDepth + 1).fill(0.0);

            function DoComputeSolverColumnWidths(d) {
                var heightPx = HEIGHT_PX * d.numSolverLeaves / zoomedElement.numSolverLeaves;
                d.textOpacity0 = d.hasOwnProperty('textOpacity') ? d.textOpacity : 0;
                d.textOpacity = (heightPx > FONT_SIZE_PX) ? 1 : 0;
                var hasVisibleDetail = (heightPx >= 2.0);
                var widthPx = 1e-3;
                if (hasVisibleDetail) widthPx = MIN_COLUMN_WIDTH_PX;
                if (d.textOpacity > 0.5) widthPx = d.nameSolverWidthPx;

                greatestDepth = Math.max(greatestDepth, d.depth);

                if (d.subsystem_children && !d.isMinimized) { //not leaf
                    columnSolverWidthsPx[d.depth] = Math.max(columnSolverWidthsPx[d.depth], widthPx);
                    for (var i = 0; i < d.subsystem_children.length; ++i) {
                        DoComputeSolverColumnWidths(d.subsystem_children[i]);
                    }
                }
                else { //leaf
                    leafSolverWidthsPx[d.depth] = Math.max(leafSolverWidthsPx[d.depth], widthPx);
                }
            }

            DoComputeSolverColumnWidths(d);


            var sum = 0;
            var lastColumnWidth = 0;
            for (var i = leafSolverWidthsPx.length - 1; i >= zoomedElement.depth; --i) {
                sum += columnSolverWidthsPx[i];
                var lastWidthNeeded = leafSolverWidthsPx[i] - sum;
                lastColumnWidth = Math.max(lastWidthNeeded, lastColumnWidth);
            }
            columnSolverWidthsPx[zoomedElement.depth - 1] = PARENT_NODE_WIDTH_PX;
            columnSolverWidthsPx[greatestDepth] = lastColumnWidth;

        }


        function ComputeLeaves(d) {
            if ((!showParams && (d.type === "param" || d.type === "unconnected_params")) || d.varIsHidden) {
                d.numLeaves = 0;
                return;
            }
            var doRecurse = d.children && !d.isMinimized;
            d.numLeaves = doRecurse ? 0 : 1; //no children: init to 0 because will be added later
            if (!doRecurse) return;

            for (var i = 0; i < d.children.length; ++i) {
                ComputeLeaves(d.children[i]);
                d.numLeaves += d.children[i].numLeaves;
            }
        }

        function ComputeSolverLeaves(d) {
            if ((!showParams && (d.type === "param" || d.type === "unconnected_params")) || d.varIsHidden) {
                d.numSolverLeaves = 0;
                return;
            }

            var doRecurse = d.children && !d.isMinimized;
            d.numSolverLeaves = doRecurse ? 0 : 1; //no children: init to 0 because will be added later
            if (!doRecurse) return;

            for (var i = 0; i < d.children.length; ++i) {
                ComputeSolverLeaves(d.children[i]);
                d.numSolverLeaves += d.children[i].numSolverLeaves;
            }
        }

        function ComputeNormalizedPositions(d, leafCounter, isChildOfZoomed, earliestMinimizedParent) {
            isChildOfZoomed = (isChildOfZoomed) ? true : (d === zoomedElement);
            if (earliestMinimizedParent == null && isChildOfZoomed) {
                if ((showParams || (d.type !== "param" && d.type !== "unconnected_param")) && !d.varIsHidden) d3NodesArray.push(d);
                if (!d.children || d.isMinimized) { //at a "leaf" node
                    if ((showParams || (d.type !== "param" && d.type !== "unconnected_param")) && !d.varIsHidden) d3RightTextNodesArrayZoomed.push(d);
                    earliestMinimizedParent = d;
                }
            }
            var node = (earliestMinimizedParent) ? earliestMinimizedParent : d;
            d.rootIndex0 = d.hasOwnProperty('rootIndex') ? d.rootIndex : leafCounter;
            d.rootIndex = leafCounter;
            d.x0 = d.hasOwnProperty('x') ? d.x : 1e-6;
            d.y0 = d.hasOwnProperty('y') ? d.y : 1e-6;
            d.width0 = d.hasOwnProperty('width') ? d.width : 1e-6;
            d.height0 = d.hasOwnProperty('height') ? d.height : 1e-6;
            d.x = columnLocationsPx[node.depth] / widthPTreePx;
            d.y = leafCounter / root.numLeaves;
            d.width = (d.children && !d.isMinimized) ? (columnWidthsPx[node.depth] / widthPTreePx) : 1 - node.x;//1-d.x;
            d.height = node.numLeaves / root.numLeaves;
            if ((!showParams && (d.type === "param" || d.type === "unconnected_param")) || d.varIsHidden) { //param or hidden leaf leaving
                d.x = columnLocationsPx[d.parentComponent.depth + 1] / widthPTreePx;
                d.y = d.parentComponent.y;
                d.width = 1e-6;
                d.height = 1e-6;
            }

            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    ComputeNormalizedPositions(d.children[i], leafCounter, isChildOfZoomed, earliestMinimizedParent);
                    if (earliestMinimizedParent == null) { //numleaves is only valid passed nonminimized nodes
                        leafCounter += d.children[i].numLeaves;
                    }
                }
            }
        }


        function ComputeSolverNormalizedPositions(d, leafCounter, isChildOfZoomed, earliestMinimizedParent) {
            isChildOfZoomed = (isChildOfZoomed) ? true : (d === zoomedElement);
            if (earliestMinimizedParent == null && isChildOfZoomed) {
                if (d.type === "subsystem" || d.type === "root" ) d3SolverNodesArray.push(d);
                if (!d.children || d.isMinimized) { //at a "leaf" node
                    if ((d.type !== "param" && d.type !== "unconnected_param") && !d.varIsHidden) d3SolverRightTextNodesArrayZoomed.push(d);
                    earliestMinimizedParent = d;
                }
            }
            var node = (earliestMinimizedParent) ? earliestMinimizedParent : d;
            d.rootIndex0 = d.hasOwnProperty('rootIndex') ? d.rootIndex : leafCounter;
            d.xSolver0 = d.hasOwnProperty('xSolver') ? d.xSolver : 1e-6;
            d.ySolver0 = d.hasOwnProperty('ySolver') ? d.ySolver : 1e-6;
            d.widthSolver0 = d.hasOwnProperty('widthSolver') ? d.widthSolver : 1e-6;
            d.heightSolver0 = d.hasOwnProperty('heightSolver') ? d.heightSolver : 1e-6;
            d.xSolver = columnSolverLocationsPx[node.depth] / widthPSolverTreePx;
            d.ySolver = leafCounter / root.numSolverLeaves;
            d.widthSolver = (d.subsystem_children && !d.isMinimized) ? (columnSolverWidthsPx[node.depth] / widthPSolverTreePx) : 1 - node.xSolver;//1-d.x;

            d.heightSolver = node.numSolverLeaves / root.numSolverLeaves; 111






            if ((!showParams && (d.type === "param" || d.type === "unconnected_param")) || d.varIsHidden) { //param or hidden leaf leaving
                d.xSolver = columnLocationsPx[d.parentComponent.depthqqq + 1] / widthPTreePx;
                d.ySolver = d.parentComponent.y;
                d.widthSolver = 1e-6;
                d.heightSolver = 1e-6;
            }






            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    ComputeSolverNormalizedPositions(d.children[i], leafCounter, isChildOfZoomed, earliestMinimizedParent);
                    if (earliestMinimizedParent == null) { //numleaves is only valid passed nonminimized nodes
                        leafCounter += d.children[i].numSolverLeaves;
                    }
                }
            }
        }

        UpdateTextWidths(zoomedElement);

        UpdateSolverTextWidths(zoomedElement);

        ComputeLeaves(root);

        ComputeSolverLeaves(root);

        d3NodesArray = [];
        d3RightTextNodesArrayZoomed = [];

        d3SolverNodesArray = [];
        d3SolverRightTextNodesArrayZoomed = [];

        ComputeColumnWidths(zoomedElement);

        function InitSubSystemChildren(d){
            for (var i = 0; i < d.children.length; ++i) {
                var child = d.children[i];
                if (child.type === "subsystem"){
                    if (!d.hasOwnProperty("subsystem_children")) {
                        d.subsystem_children = [];
                    }
                    d.subsystem_children.push(child);
                    InitSubSystemChildren(child);
                }
            }
        }

        InitSubSystemChildren(root);

        ComputeSolverColumnWidths(zoomedElement);

        // Now the column_width array is relative to the zoomedElement
        //    and the computation of the widths only includes visible items after the zoom
        widthPTreePx = 0;
        for (var depth = 1; depth <= maxDepth; ++depth) {
            columnLocationsPx[depth] = widthPTreePx;
            widthPTreePx += columnWidthsPx[depth];
        }

        widthPSolverTreePx = 0;
        for (var depth = 1; depth <= maxDepth; ++depth) {
            columnSolverLocationsPx[depth] = widthPSolverTreePx;
            widthPSolverTreePx += columnSolverWidthsPx[depth];
        }

        ComputeNormalizedPositions(root, 0, false, null);
        if (zoomedElement.parent) {
            d3NodesArray.push(zoomedElement.parent);
        }

        ComputeSolverNormalizedPositions(root, 0, false, null);
        if (zoomedElement.parent) {
            d3SolverNodesArray.push(zoomedElement.parent);
        }

        if (updateRecomputesAutoComplete) {
            autoCompleteListNames = [];
            autoCompleteListPathNames = [];
            PopulateAutoCompleteList(zoomedElement);
        }
        updateRecomputesAutoComplete = true; //default

        enterIndex = exitIndex = 0;
        if (lastClickWasLeft) { //left click
            if (leftClickIsForward) {
                exitIndex = lastLeftClickedElement.rootIndex - zoomedElement0.rootIndex;
            }
            else {
                enterIndex = zoomedElement0.rootIndex - lastLeftClickedElement.rootIndex;
            }
        }

        textWidthGroup.remove();
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
            Update(d);
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
        zoomedElement0 = zoomedElement;
        zoomedElement = d;
        TRANSITION_DURATION = TRANSITION_DURATION_FAST;
    }

    //left click => navigate
    function LeftClick(d, ele) {
        if (!d.children) return;
        if (d3.event.button != 0) return;
        backButtonHistory.push({ "el": zoomedElement });
        forwardButtonHistory = [];
        SetupLeftClick(d);
        Update();
        d3.event.preventDefault();
        d3.event.stopPropagation();
    }

    function BackButtonPressed() {
        if (backButtonHistory.length == 0) return;
        var d = backButtonHistory.pop().el;
        parentDiv.querySelector("#backButtonId").disabled = (backButtonHistory.length == 0) ? "disabled" : false;
        for (var obj = d; obj != null; obj = obj.parent) { //make sure history item is not minimized
            if (obj.isMinimized) return;
        }
        forwardButtonHistory.push({ "el": zoomedElement });
        SetupLeftClick(d);
        Update();
    }

    function ForwardButtonPressed() {
        if (forwardButtonHistory.length == 0) return;
        var d = forwardButtonHistory.pop().el;
        parentDiv.querySelector("#forwardButtonId").disabled = (forwardButtonHistory.length == 0) ? "disabled" : false;
        for (var obj = d; obj != null; obj = obj.parent) { //make sure history item is not minimized
            if (obj.isMinimized) return;
        }
        backButtonHistory.push({ "el": zoomedElement });
        SetupLeftClick(d);
        Update();
    }

    function GetClass(d) {
        if (d.isMinimized) return "minimized";
        if (d.type === "param") {
            if (d.children && d.children.length > 0) return "param_group";
            return "param";
        }
        if (d.type === "unconnected_param") {
            if (d.children && d.children.length > 0) return "param_group";
            return "unconnected_param"
        }
        if (d.type === "unknown") {
            if (d.children && d.children.length > 0) return "unknown_group";
            if (d.implicit) return "unknown_implicit";
            return "unknown";
        }
        if (d.type === "root") return "subsystem";
        if (d.type === "subsystem") {
            if (d.subsystem_type === "component") return "component";
            return "subsystem";
        }
        alert("class not found");
    }

    function Toggle(d) {

        if (d.isMinimized)
            d.isMinimized = false;
        else
            d.isMinimized = true;
    }

    function ComputeConnections() {
        function GetObjectInTree(d, nameArray, nameIndex) {
            if (nameArray.length == nameIndex) {
                return d;
            }
            if (!d.children) {
                return null;
            }

            for (var i = 0; i < d.children.length; ++i) {
                if (d.children[i].name === nameArray[nameIndex]) {
                    return GetObjectInTree(d.children[i], nameArray, nameIndex + 1);
                }
                else {
                    var numNames = d.children[i].name.split(":").length;
                    if (numNames >= 2 && nameIndex + numNames <= nameArray.length) {
                        var mergedName = nameArray[nameIndex];
                        for (var j = 1; j < numNames; ++j) {
                            mergedName += ":" + nameArray[nameIndex + j];
                        }
                        if (d.children[i].name === mergedName) {
                            return GetObjectInTree(d.children[i], nameArray, nameIndex + numNames);
                        }
                    }
                }
            }
            return null;
        }

        function RemoveDuplicates(d) { //remove redundant elements in every objects' sources and targets arrays
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    RemoveDuplicates(d.children[i]);
                }
            }

            function unique(elem, pos, arr) {
                return arr.indexOf(elem) == pos;
            }

            if (d.targetsParamView) {
                //numElementsBefore += d.targetsParamView.length;
                var uniqueArray = d.targetsParamView.filter(unique);
                d.targetsParamView = uniqueArray;
                //numElementsAfter += d.targetsParamView.length;
            }
            if (d.targetsHideParams) {
                //numElementsBefore += d.targetsHideParams.length;
                var uniqueArray = d.targetsHideParams.filter(unique);
                d.targetsHideParams = uniqueArray;
                //numElementsAfter += d.targetsHideParams.length;
            }
        }

        function AddLeaves(d, objArray) {
            if (d.type !== "param" && d.type !== "unconnected_param") {
                objArray.push(d);
            }
            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    AddLeaves(d.children[i], objArray);
                }
            }
        }

        function ClearConnections(d) {
            // d.targetsParamView = [];
            // d.targetsHideParams = [];
            d.targetsParamView = new Set();
            d.targetsHideParams = new Set();

            if (d.children) {
                for (var i = 0; i < d.children.length; ++i) {
                    ClearConnections(d.children[i]);
                }
            }
        }

        ClearConnections(root);

        for (var i = 0; i < conns.length; ++i) {
            var srcSplitArray = conns[i].src.split(/\.|:/);
            var srcObj = GetObjectInTree(root, srcSplitArray, 0);
            if (srcObj == null) {
                alert("error: cannot find connection source " + conns[i].src);
                return;
            }
            var srcObjArray = [srcObj];
            if (srcObj.type !== "unknown") { //source obj must be unknown
                alert("error: there is a source that is not an unknown.");
                return;
            }
            if (srcObj.children) { //source obj must be unknown
                alert("error: there is a source that has children.");
                return;
            }
            for (var obj = srcObj.parent; obj != null; obj = obj.parent) {
                srcObjArray.push(obj);
            }

            var tgtSplitArray = conns[i].tgt.split(/\.|:/);
            var tgtObj = GetObjectInTree(root, tgtSplitArray, 0);
            if (tgtObj == null) {
                alert("error: cannot find connection target " + conns[i].tgt);
                return;
            }
            var tgtObjArrayParamView = [tgtObj];
            var tgtObjArrayHideParams = [tgtObj];
            if (tgtObj.type !== "param" && tgtObj.type !== "unconnected_param") { //target obj must be a param
                alert("error: there is a target that is NOT a param.");
                return;
            }
            if (tgtObj.children) {
                alert("error: there is a target that has children.");
                return;
            }
            AddLeaves(tgtObj.parentComponent, tgtObjArrayHideParams); //contaminate
            for (var obj = tgtObj.parent; obj != null; obj = obj.parent) {
                tgtObjArrayParamView.push(obj);
                tgtObjArrayHideParams.push(obj);
            }


            for (var j = 0; j < srcObjArray.length; ++j) {
                if (!srcObjArray[j].hasOwnProperty('targetsParamView')) srcObjArray[j].targetsParamView = new Set();
                if (!srcObjArray[j].hasOwnProperty('targetsHideParams')) srcObjArray[j].targetsHideParams = new Set();

                tgtObjArrayParamView.forEach(item => srcObjArray[j].targetsParamView.add(item));
                tgtObjArrayHideParams.forEach(item => srcObjArray[j].targetsHideParams.add(item));



                // if (!srcObjArray[j].hasOwnProperty('targetsParamView')) srcObjArray[j].targetsParamView = [];
                // if (!srcObjArray[j].hasOwnProperty('targetsHideParams')) srcObjArray[j].targetsHideParams = [];
                // srcObjArray[j].targetsParamView = srcObjArray[j].targetsParamView.concat(tgtObjArrayParamView);
                // srcObjArray[j].targetsHideParams = srcObjArray[j].targetsHideParams.concat(tgtObjArrayHideParams);
            }

            var cycleArrowsArray = [];
            if (conns[i].cycle_arrows && conns[i].cycle_arrows.length > 0) {
                var cycleArrows = conns[i].cycle_arrows;
                for (var j = 0; j < cycleArrows.length; ++j) {
                    var cycleArrowsSplitArray = cycleArrows[j].split(" ");
                    if (cycleArrowsSplitArray.length != 2) {
                        alert("error: cycleArrowsSplitArray length not 2: got " + cycleArrowsSplitArray.length);
                        return;
                    }
                    var splitArray = cycleArrowsSplitArray[0].split(/\.|:/);
                    var arrowBeginObj = GetObjectInTree(root, splitArray, 0);
                    if (arrowBeginObj == null) {
                        alert("error: cannot find cycle arrows begin object " + cycleArrowsSplitArray[0]);
                        return;
                    }
                    splitArray = cycleArrowsSplitArray[1].split(/\.|:/);
                    var arrowEndObj = GetObjectInTree(root, splitArray, 0);
                    if (arrowEndObj == null) {
                        alert("error: cannot find cycle arrows end object " + cycleArrowsSplitArray[1]);
                        return;
                    }
                    cycleArrowsArray.push({ "begin": arrowBeginObj, "end": arrowEndObj });
                }
            }
            if (cycleArrowsArray.length > 0) {
                if (!tgtObj.parent.hasOwnProperty("cycleArrows")) {
                    tgtObj.parent.cycleArrows = [];
                }
                tgtObj.parent.cycleArrows.push({ "src": srcObj, "arrows": cycleArrowsArray });
            }

        }
        // RemoveDuplicates(root);
    }

    function ComputeMatrixN2() {
        matrix = {};
        if (d3RightTextNodesArrayZoomed.length < LEVEL_OF_DETAIL_THRESHOLD) {
            for (var si = 0; si < d3RightTextNodesArrayZoomed.length; ++si) {
                var srcObj = d3RightTextNodesArrayZoomed[si];
                matrix[si + "_" + si] = { "r": si, "c": si, "obj": srcObj, "id": srcObj.id + "_" + srcObj.id };
                var targets = (showParams) ? srcObj.targetsParamView : srcObj.targetsHideParams;
                for (let tgtObj of targets) {
                // for (var j = 0; j < targets.length; ++j) {
                //     var tgtObj = targets[j];
                    var ti = d3RightTextNodesArrayZoomed.indexOf(tgtObj);
                    if (ti != -1) {
                        matrix[si + "_" + ti] = { "r": si, "c": ti, "obj": srcObj, "id": srcObj.id + "_" + tgtObj.id }; //matrix[si][ti].z = 1;
                    }
                }
                if (showParams && (srcObj.type === "param" || srcObj.type === "unconnected_param")) {
                    for (var j = si + 1; j < d3RightTextNodesArrayZoomed.length; ++j) {
                        var tgtObj = d3RightTextNodesArrayZoomed[j];
                        if (srcObj.parentComponent !== tgtObj.parentComponent) break;
                        if (tgtObj.type === "unknown") {
                            var ti = j;
                            matrix[si + "_" + ti] = { "r": si, "c": ti, "obj": srcObj, "id": srcObj.id + "_" + tgtObj.id };
                        }
                    }
                }
            }
        }
        n2Dx0 = n2Dx;
        n2Dy0 = n2Dy;

        n2Dx = WIDTH_N2_PX / d3RightTextNodesArrayZoomed.length;
        n2Dy = HEIGHT_PX / d3RightTextNodesArrayZoomed.length;

        symbols_scalar = [];
        symbols_vector = [];
        symbols_group = [];
        symbols_scalarScalar = [];
        symbols_scalarVector = [];
        symbols_vectorScalar = [];
        symbols_vectorVector = [];
        symbols_scalarGroup = [];
        symbols_groupScalar = [];
        symbols_vectorGroup = [];
        symbols_groupVector = [];
        symbols_groupGroup = [];

        for (var key in matrix) {
            var d = matrix[key];
            var tgtObj = d3RightTextNodesArrayZoomed[d.c], srcObj = d3RightTextNodesArrayZoomed[d.r];
            //alert(tgtObj.name + " " + srcObj.name);
            if (d.c == d.r) { //on diagonal
                if (srcObj.type === "subsystem") { //group
                    symbols_group.push(d);
                } else if (srcObj.type === "unknown" || (showParams && (srcObj.type === "param" || srcObj.type === "unconnected_param"))) {
                    if (srcObj.dtype === "ndarray") { //vector
                        symbols_vector.push(d);
                    } else { //scalar
                        symbols_scalar.push(d);
                    }
                }

            }
            else if (srcObj.type === "subsystem") {
                if (tgtObj.type === "subsystem") { //groupGroup
                    symbols_groupGroup.push(d);
                }
                else if (tgtObj.type === "unknown" || (showParams && (tgtObj.type === "param" || tgtObj.type === "unconnected_param"))) {
                    if (tgtObj.dtype === "ndarray") {//groupVector
                        symbols_groupVector.push(d);
                    }
                    else {//groupScalar
                        symbols_groupScalar.push(d);
                    }
                }
            }
            else if (srcObj.type === "unknown" || (showParams && (srcObj.type === "param" || srcObj.type === "unconnected_param"))) {
                if (srcObj.dtype === "ndarray") {
                    if (tgtObj.type === "unknown" || (showParams && (tgtObj.type === "param" || tgtObj.type === "unconnected_param"))) {
                        if (tgtObj.dtype === "ndarray" || (showParams && (tgtObj.type === "param" || tgtObj.type === "unconnected_param"))) {//vectorVector
                            symbols_vectorVector.push(d);
                        }
                        else {//vectorScalar
                            symbols_vectorScalar.push(d);
                        }

                    }
                    else if (tgtObj.type === "subsystem") { //vectorGroup
                        symbols_vectorGroup.push(d);
                    }
                }
                else { //if (srcObj.dtype !== "ndarray"){
                    if (tgtObj.type === "unknown" || (showParams && (tgtObj.type === "param" || tgtObj.type === "unconnected_param"))) {
                        if (tgtObj.dtype === "ndarray") {//scalarVector
                            symbols_scalarVector.push(d);
                        }
                        else {//scalarScalar
                            symbols_scalarScalar.push(d);
                        }

                    }
                    else if (tgtObj.type === "subsystem") { //scalarGroup
                        symbols_scalarGroup.push(d);
                    }
                }

            }
        }

        var currentBox = { "startI": 0, "stopI": 0 };
        d3RightTextNodesArrayZoomedBoxInfo = [currentBox];
        for (var ri = 1; ri < d3RightTextNodesArrayZoomed.length; ++ri) {
            //boxes
            var el = d3RightTextNodesArrayZoomed[ri];
            var startINode = d3RightTextNodesArrayZoomed[currentBox.startI];
            if (startINode.parentComponent && el.parentComponent && startINode.parentComponent === el.parentComponent) {
                ++currentBox.stopI;
            }
            else {
                currentBox = { "startI": ri, "stopI": ri };
            }
            d3RightTextNodesArrayZoomedBoxInfo.push(currentBox);
        }

        drawableN2ComponentBoxes = [];
        for (var i = 0; i < d3RightTextNodesArrayZoomedBoxInfo.length; ++i) { //draw grid lines last so that they will always be visible
            var box = d3RightTextNodesArrayZoomedBoxInfo[i];
            if (box.startI == box.stopI) continue;
            var el = d3RightTextNodesArrayZoomed[box.startI];
            if (!el.parentComponent) alert("parent component not found in box"); //continue;
            box.obj = el.parentComponent;
            i = box.stopI;
            drawableN2ComponentBoxes.push(box);
        }

        //do this so you save old index for the exit()
        gridLines = [];
        if (d3RightTextNodesArrayZoomed.length < LEVEL_OF_DETAIL_THRESHOLD) {
            for (var i = 0; i < d3RightTextNodesArrayZoomed.length; ++i) {
                var obj = d3RightTextNodesArrayZoomed[i];
                var gl = { "i": i, "obj": obj };
                gridLines.push(gl);
            }
        }
    }


    function FindRootOfChangeForShowParams(d) {
        return (d.hasOwnProperty("parentComponent")) ? d.parentComponent : d;
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
            for (var obj = d; obj != null; obj = obj.parent) {
                if (obj === toMatchObj) {
                    return true;
                }
            }
            return HasObjectInChildren(d, toMatchObj);
        }

        var lineWidth = Math.min(5, n2Dx * .5, n2Dy * .5);
        arrowMarker.attr("markerWidth", lineWidth * .4)
            .attr("markerHeight", lineWidth * .4);
        var src = d3RightTextNodesArrayZoomed[d.r];
        var tgt = d3RightTextNodesArrayZoomed[d.c];
        var boxEnd = d3RightTextNodesArrayZoomedBoxInfo[d.c];
        if (d.r > d.c) { //bottom left
            DrawPathTwoLines(
                n2Dx * d.r, //x1
                n2Dy * d.r + n2Dy * .5, //y1
                n2Dx * d.c + n2Dx * .5, //left x2
                n2Dy * d.r + n2Dy * .5, //left y2
                n2Dx * d.c + n2Dx * .5, //up x3
                (showParams) ? n2Dy * d.c + n2Dy - 1e-2 : n2Dy * (boxEnd.stopI) + n2Dy - 1e-2, //up y3
                RED_ARROW_COLOR, lineWidth, true);

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
                                if (showParams) {
                                    DrawArrowsParamView(firstBeginIndex, firstEndIndex);
                                }
                                else {
                                    DrawArrows(firstBeginIndex, firstEndIndex);
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (d.r < d.c) { //top right
            DrawPathTwoLines(
                n2Dx * d.r + n2Dx, //x1
                n2Dy * d.r + n2Dy * .5, //y1
                n2Dx * d.c + n2Dx * .5, //right x2
                n2Dy * d.r + n2Dy * .5, //right y2
                n2Dx * d.c + n2Dx * .5, //down x3
                (showParams) ? n2Dy * d.c + 1e-2 : n2Dy * boxEnd.startI + 1e-2, //down y3
                RED_ARROW_COLOR, lineWidth, true);
        }
        var leftTextWidthR = d3RightTextNodesArrayZoomed[d.r].nameWidthPx,
            leftTextWidthC = d3RightTextNodesArrayZoomed[d.c].nameWidthPx;
        DrawRect(-leftTextWidthR - PTREE_N2_GAP_PX, n2Dy * d.r, leftTextWidthR, n2Dy, RED_ARROW_COLOR); //highlight var name
        DrawRect(-leftTextWidthC - PTREE_N2_GAP_PX, n2Dy * d.c, leftTextWidthC, n2Dy, GREEN_ARROW_COLOR); //highlight var name
    }

    function MouseoverOnDiagN2(d) {
        //d=hovered element
        var hoveredIndexRC = d.c; //d.x == d.y == row == col
        var leftTextWidthHovered = d3RightTextNodesArrayZoomed[hoveredIndexRC].nameWidthPx;

        // Loop over all elements in the matrix looking for other cells in the same column as
        var lineWidth = Math.min(5, n2Dx * .5, n2Dy * .5);
        arrowMarker.attr("markerWidth", lineWidth * .4)
            .attr("markerHeight", lineWidth * .4);
        DrawRect(-leftTextWidthHovered - PTREE_N2_GAP_PX, n2Dy * hoveredIndexRC, leftTextWidthHovered, n2Dy, HIGHLIGHT_HOVERED_COLOR); //highlight hovered
        for (var i = 0; i < d3RightTextNodesArrayZoomed.length; ++i) {
            var leftTextWidthDependency = d3RightTextNodesArrayZoomed[i].nameWidthPx;
            var box = d3RightTextNodesArrayZoomedBoxInfo[i];
            if (matrix[hoveredIndexRC + "_" + i] !== undefined) { //if (matrix[hoveredIndexRC][i].z > 0) { //i is column here
                if (i < hoveredIndexRC) { //column less than hovered
                    if (showParams) {
                        DrawPathTwoLines(
                            n2Dx * hoveredIndexRC, //x1
                            n2Dy * (hoveredIndexRC + .5), //y1
                            (i + .5) * n2Dx, //left x2
                            n2Dy * (hoveredIndexRC + .5), //left y2
                            (i + .5) * n2Dx, //up x3
                            (i + 1) * n2Dy, //up y3
                            GREEN_ARROW_COLOR, lineWidth, true);
                    }
                    else if (i == box.startI) {
                        DrawPathTwoLines(
                            n2Dx * hoveredIndexRC, //x1
                            n2Dy * hoveredIndexRC + n2Dy * .5, //y1
                            (box.startI + (box.stopI - box.startI) * .5) * n2Dx + n2Dx * .5, //left x2
                            n2Dy * hoveredIndexRC + n2Dy * .5, //left y2
                            (box.startI + (box.stopI - box.startI) * .5) * n2Dx + n2Dx * .5, //up x3
                            n2Dy * box.stopI + n2Dy, //up y3
                            GREEN_ARROW_COLOR, lineWidth, true);
                    }
                    DrawRect(-leftTextWidthDependency - PTREE_N2_GAP_PX, n2Dy * i, leftTextWidthDependency, n2Dy, GREEN_ARROW_COLOR); //highlight var name

                } else if (i > hoveredIndexRC) { //column greater than hovered
                    if (showParams) {
                        DrawPathTwoLines(
                            n2Dx * hoveredIndexRC + n2Dx, //x1
                            n2Dy * (hoveredIndexRC + .5), //y1
                            (i + .5) * n2Dx, //right x2
                            n2Dy * (hoveredIndexRC + .5), //right y2
                            (i + .5) * n2Dx, //down x3
                            n2Dy * i, //down y3
                            GREEN_ARROW_COLOR, lineWidth, true); //vertical down
                    }
                    else if (i == box.startI) {
                        DrawPathTwoLines(
                            n2Dx * hoveredIndexRC + n2Dx, //x1
                            n2Dy * hoveredIndexRC + n2Dy * .5, //y1
                            (box.startI + (box.stopI - box.startI) * .5) * n2Dx + n2Dx * .5, //right x2
                            n2Dy * hoveredIndexRC + n2Dy * .5, //right y2
                            (box.startI + (box.stopI - box.startI) * .5) * n2Dx + n2Dx * .5, //down x3
                            n2Dy * box.startI, //down y3
                            GREEN_ARROW_COLOR, lineWidth, true); //vertical down
                    }
                    DrawRect(-leftTextWidthDependency - PTREE_N2_GAP_PX, n2Dy * i, leftTextWidthDependency, n2Dy, GREEN_ARROW_COLOR); //highlight var name
                }
            }

            if (matrix[i + "_" + hoveredIndexRC] !== undefined) { //if (matrix[i][hoveredIndexRC].z > 0) { //i is row here
                if (i < hoveredIndexRC) { //row less than hovered
                    DrawPathTwoLines(
                        n2Dx * i + n2Dx, //x1
                        n2Dy * i + n2Dy * .5, //y1
                        n2Dx * hoveredIndexRC + n2Dx * .5, //right x2
                        n2Dy * i + n2Dy * .5, //right y2
                        n2Dx * hoveredIndexRC + n2Dx * .5, //down x3
                        n2Dy * hoveredIndexRC, //down y3
                        RED_ARROW_COLOR, lineWidth, true); //vertical down
                    DrawRect(-leftTextWidthDependency - PTREE_N2_GAP_PX, n2Dy * i, leftTextWidthDependency, n2Dy, RED_ARROW_COLOR); //highlight var name
                } else if (i > hoveredIndexRC) { //row greater than hovered
                    DrawPathTwoLines(
                        n2Dx * i, //x1
                        n2Dy * i + n2Dy * .5, //y1
                        n2Dx * hoveredIndexRC + n2Dx * .5, //left x2
                        n2Dy * i + n2Dy * .5, //left y2
                        n2Dx * hoveredIndexRC + n2Dx * .5, //up x3
                        n2Dy * hoveredIndexRC + n2Dy, //up y3
                        RED_ARROW_COLOR, lineWidth, true);
                    DrawRect(-leftTextWidthDependency - PTREE_N2_GAP_PX, n2Dy * i, leftTextWidthDependency, n2Dy, RED_ARROW_COLOR); //highlight var name
                }
            }
        }
    }

    function MouseoutN2() {
        n2Group.selectAll(".n2_hover_elements").remove();
    }

    function MouseClickN2(d) {
        var newClassName = "n2_hover_elements_" + d.r + "_" + d.c;
        var selection = n2Group.selectAll("." + newClassName);
        if (selection.size() > 0) {
            selection.remove();
        }
        else {
            n2Group.selectAll("path.n2_hover_elements, circle.n2_hover_elements")
                .attr("class", newClassName);
        }
    }

    function ReturnToRootButtonClick() {
        backButtonHistory.push({ "el": zoomedElement });
        forwardButtonHistory = [];
        SetupLeftClick(root);
        Update();
    }

    function UpOneLevelButtonClick() {
        if (zoomedElement === root) return;
        backButtonHistory.push({ "el": zoomedElement });
        forwardButtonHistory = [];
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
        FONT_SIZE_PX = fontSize;
        TRANSITION_DURATION = TRANSITION_DURATION_FAST;
        UpdateSvgCss(svgStyleElement, FONT_SIZE_PX);
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
        HEIGHT_PX = height;
        LEVEL_OF_DETAIL_THRESHOLD = HEIGHT_PX / 3;
        WIDTH_N2_PX = HEIGHT_PX;
        TRANSITION_DURATION = TRANSITION_DURATION_FAST;
        UpdateSvgCss(svgStyleElement, FONT_SIZE_PX);
        Update();
    }

    function ShowParamsCheckboxChange() {
        if (zoomedElement.type === "param" || zoomedElement.type === "unconnected_param") return;
        showParams = !showParams;
        parentDiv.querySelector("#showParamsButtonId").className = !showParams ? "myButton myButtonToggledOn" : "myButton";

        FindRootOfChangeFunction = FindRootOfChangeForShowParams;
        lastClickWasLeft = false;
        TRANSITION_DURATION = TRANSITION_DURATION_SLOW;
        transitionStartDelay = 500;
        SetupLegend(d3, d3ContentDiv);
        Update();
    }

    function ToggleSolverNamesCheckboxChange() {
        showLinearSolverNames = !showLinearSolverNames;
        parentDiv.querySelector("#toggleSolverNamesButtonId").className = !showLinearSolverNames ? "myButton myButtonToggledOn" : "myButton";
        SetupLegend(d3, d3ContentDiv);
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
        SetupLegend(d3, d3ContentDiv);
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
        div.querySelector("#showParamsButtonId").onclick = ShowParamsCheckboxChange;

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
            var f = function(idx) {
                return function () { VerticalResize(idx); };
            }(i);
            div.querySelector("#idVerticalResize" + i + "px").onclick = f;
        }

        div.querySelector("#saveSvgButtonId").onclick = function () { SaveSvg(parentDiv) };
        div.querySelector("#helpButtonId").onclick = DisplayModal;
    }

    return {
        GetFontSize: function () { return FONT_SIZE_PX; },
        ResizeHeight: function (h) { VerticalResize(h); },
        Redraw: function () { Update(); }
    };
}

var zoomedElement = modelData['tree'];
var updateFunc;
var mouseOverOffDiagN2;
var mouseOverOnDiagN2;
var mouseOutN2;
var mouseClickN2;
var hasInputConn;
var treeData, connectionList;
modelData.tree.name = 'model'; //Change 'root' to 'model'
function ChangeBlankSolverToNone(d) {
    if (d.linear_solver === "") d.linear_solver = "None";
    if (d.nonlinear_solver === "") d.nonlinear_solver = "None";
    if (d.children) {
        for (var i = 0; i < d.children.length; ++i) {
            ChangeBlankSolverToNone(d.children[i]);
        }
    }
}
ChangeBlankSolverToNone(modelData.tree);

console.time("PtN2Diagram");

var app = PtN2Diagram(document.getElementById("ptN2ContentDivId"), modelData);

console.timeEnd("PtN2Diagram");
