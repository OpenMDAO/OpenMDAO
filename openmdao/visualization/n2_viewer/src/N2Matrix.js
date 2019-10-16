/**
 * Use the model tree to build a matrix, display, and perform operations with it.
 * @typedef N2Matrix
 * @property {Object[]} nodes Reference to nodes that will be drawn.
 * @property {number} levelOfDetailThreshold Don't draw elements below this size in pixels.
 * @property {Object} n2Groups References to <g> SVG elements managed by N2Diagram.
 */
class N2Matrix {

    /**
     * Render the matrix of visible elements in the model.
     * @param {Object} visibleNodes Nodes that will be drawn.
     * @param {ModelData} model The pre-processed model data.
     * @param {Object} n2Groups References to <g> SVG elements created by N2Diagram.
     */
    constructor(visibleNodes, model, n2Groups) {
        this.nodes = visibleNodes;
        this.n2Groups = n2Groups;

        n2Dx0 = n2Dx;
        n2Dy0 = n2Dy;

        n2Dx = WIDTH_N2_PX / this.nodes.length;
        n2Dy = N2Layout.heightPx / this.nodes.length;

        this.updateLevelOfDetailThreshold(N2Layout.heightPx);
        this.buildStructure(model);
        this.setupSymbolArrays();
        this.drawingPrep();
    }

    /**
     * Determine if a node exists at the specified location.
     * @param {number} row Row number of the node
     * @param {number} col Column number of the node
     * @returns False if the row doesn't exist or a column doesn't exist
     *  in the row; true otherwise.
     */
    exists(row, col) {
        if (this.matrix[row] !== undefined &&
            this.matrix[row][col] !== undefined) { return true; }
        return false;
    }

    /**
     * Safe method to access a node at the specified location.
     * @param {number} row Row number of the node
     * @param {number} col Column number of the node
     * @param {boolean} doThrow Whether to throw an exception if node undefined.
     * @returns The node if it exists, undefined otherwise.
     */
    node(row, col, doThrow = false) {
        if (this.exists(row, col)) {
            return this.matrix[row][col];
        }
        else if (doThrow) {
            throw "No node in matrix at (" + row + ", " + col + ").";
        }

        return undefined;
    }

    /**
     * Compute the new minimum element size when the diagram height changes.
     * @param {number} height In pixels.
     */
    updateLevelOfDetailThreshold(height) {
        this.levelOfDetailThreshold = height / 3;
    }

    /**
     * Set up nested objects resembling a two-dimensional array as the
     * matrix, but not an actual two dimensional array because most of
     * it would be unused.
     */
    buildStructure(model) {
        this.matrix = {};

        if (this.nodes.length >= this.levelOfDetailThreshold) return;

        for (let srcIdx = 0; srcIdx < this.nodes.length; ++srcIdx) {
            let srcObj = this.nodes[srcIdx];

            // New row
            if (!this.exists(srcIdx)) this.matrix[srcIdx] = {};

            // On the diagonal
            this.matrix[srcIdx][srcIdx] = new N2MatrixNode(srcIdx, srcIdx, srcObj, srcObj, model);

            let targets = srcObj.targetsParamView;

            for (let tgtObj of targets) {
                let tgtIdx = indexFor(this.nodes, tgtObj);
                if (tgtIdx != -1) {
                    this.matrix[srcIdx][tgtIdx] = new N2MatrixNode(srcIdx, tgtIdx, srcObj, tgtObj, model);
                }
            }

            // Solver nodes
            if (srcObj.isParam()) {
                for (let j = srcIdx + 1; j < this.nodes.length; ++j) {
                    let tgtObj = this.nodes[j];
                    if (srcObj.parentComponent !== tgtObj.parentComponent) break;

                    if (tgtObj.type == "unknown") {
                        let tgtIdx = j;
                        this.matrix[srcIdx][tgtIdx] = new N2MatrixNode(srcIdx, tgtIdx, srcObj, tgtObj, model);
                    }
                }
            }
        }
    }

    /** Decide what object each node will be drawn as, based on its
     * location in the matrix, type, source, target, and/or other conditions.
     * Add the node to the appropriate array for drawing later.
     */
    setupSymbolArrays() {
        this.symbols = {
            'scalar': [],
            'vector': [],
            'group': [],
            'scalarScalar': [],
            'scalarVector': [],
            'vectorScalar': [],
            'vectorVector': [],
            'scalarGroup': [],
            'groupScalar': [],
            'vectorGroup': [],
            'groupVector': [],
            'groupGroup': [],
            'declaredPartials': {
                'vectorVector': [],
                'scalarScalar': [],
                'vectorScalar': [],
                'scalarVector': []
            }
        };

        for (let row in this.matrix) {
            for (let col in this.matrix[row]) {
                let node = this.matrix[row][col];
     
                this.symbols[node.symbolType.name].push(node);
                if (node.symbolType.declaredPartial)
                    this.symbols.declaredPartials[node.symbolType.name].push(node);
            }
        }
    }

    drawingPrep() {
        let currentBox = { "startI": 0, "stopI": 0 };

        d3RightTextNodesArrayZoomedBoxInfo = [currentBox];

        // Find which component box each of the parameters belong to
        for (let ri = 1; ri < this.nodes.length; ++ri) {
            //boxes
            let el = this.nodes[ri];
            let startINode = this.nodes[currentBox.startI];
            if (startINode.parentComponent &&
                el.parentComponent &&
                startINode.parentComponent === el.parentComponent) {
                ++currentBox.stopI;
            }
            else {
                currentBox = { "startI": ri, "stopI": ri };
            }
            d3RightTextNodesArrayZoomedBoxInfo.push(currentBox);
        }

        drawableN2ComponentBoxes = [];
        // draw grid lines last so that they will always be visible
        for (let i = 0; i < d3RightTextNodesArrayZoomedBoxInfo.length; ++i) {
            let box = d3RightTextNodesArrayZoomedBoxInfo[i];
            if (box.startI == box.stopI) continue;
            let el = this.nodes[box.startI];
            if (!el.parentComponent) {
                throw "Parent component not found in box.";
            }
            box.obj = el.parentComponent;
            i = box.stopI;
            drawableN2ComponentBoxes.push(box);
        }

        //do this so you save old index for the exit()
        gridLines = [];
        if (this.nodes.length < this.levelOfDetailThreshold) {
            for (let i = 0; i < this.nodes.length; ++i) {
                let obj = this.nodes[i];
                var gl = { "i": i, "obj": obj };
                gridLines.push(gl);
            }
        }
        /*
        console.log("d3RightTextNodesArrayZoomedBoxInfo: ", d3RightTextNodesArrayZoomedBoxInfo);
        console.log("currentBox:", currentBox);
        console.log("drawableN2ComponentBoxes:", drawableN2ComponentBoxes);
        console.log("gridLines:", gridLines);
        */
    }

    draw() {
        let u0 = n2Dx0 * .5,
            v0 = n2Dy0 * .5,
            u = n2Dx * .5,
            v = n2Dy * .5; //(0,0) = center of cell... (u,v) = bottom right of cell... (-u,-v) = top left of cell

        let classes = [
            "cell_scalar",
            "cell_vector",
            "cell_group",
            "cell_scalarScalar",
            "cell_scalarVector",
            "cell_vectorScalar",
            "cell_vectorVector",
            "cell_scalarGroup",
            "cell_groupScalar",
            "cell_vectorGroup",
            "cell_groupVector",
            "cell_groupGroup"
        ];

        let symSrcObj = (modelData.options.use_declare_partial_info) ?
            this.symbols.declaredPartials : this.symbols;

        let datas = [
            this.symbols.scalar,
            this.symbols.vector,
            this.symbols.group,
            symSrcObj.scalarScalar,
            symSrcObj.scalarVector,
            symSrcObj.vectorScalar,
            symSrcObj.vectorVector,
            this.symbols.scalarGroup,
            this.symbols.groupScalar,
            this.symbols.vectorGroup,
            this.symbols.groupVector,
            this.symbols.groupGroup
        ];

        let drawFunctions = [
            DrawScalar,
            DrawVector,
            DrawGroup,
            DrawScalar,
            DrawVector,
            DrawVector,
            DrawVector,
            DrawGroup,
            DrawGroup,
            DrawGroup,
            DrawGroup,
            DrawGroup
        ];

        for (var i = 0; i < classes.length; ++i) {
            var sel = this.n2Groups.elements.selectAll("." + classes[i])
                .data(datas[i], function (d) {
                    return d.id;
                });
            var gEnter = sel.enter().append("g")
                .attr("class", classes[i])
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(" + (n2Dx0 * (d.col - enterIndex) + u0) + "," + (n2Dy0 * (d.row - enterIndex) + v0) + ")";
                    var roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index0 = roc.rootIndex0 - zoomedElement.rootIndex0;
                        return "translate(" + (n2Dx0 * index0 + u0) + "," + (n2Dy0 * index0 + v0) + ")";
                    }
                    throw("enter transform not found");
                });
            drawFunctions[i](gEnter, u0, v0, (i < 3) ? getOnDiagonalCellColor : N2Style.color.connection, false)
                .on("mouseover", (i < 3) ? mouseOverOnDiagN2 : mouseOverOffDiagN2)
                .on("mouseleave", mouseOutN2)
                .on("click", mouseClickN2);


            var gUpdate = gEnter.merge(sel).transition(sharedTransition)
                .attr("transform", function (d) {
                    return "translate(" + (n2Dx * (d.col) + u) + "," + (n2Dy * (d.row) + v) + ")";
                });
            drawFunctions[i](gUpdate, u, v, (i < 3) ? getOnDiagonalCellColor : N2Style.color.connection, true);


            var nodeExit = sel.exit().transition(sharedTransition)
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(" + (n2Dx * (d.col - exitIndex) + u) + "," + (n2Dy * (d.row - exitIndex) + v) + ")";
                    var roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index = roc.rootIndex - zoomedElement.rootIndex;
                        return "translate(" + (n2Dx * index + u) + "," + (n2Dy * index + v) + ")";
                    }
                    throw("exit transform not found");
                })
                .remove();
            drawFunctions[i](nodeExit, u, v, (i < 3) ? getOnDiagonalCellColor : N2Style.color.connection, true);
        }

        {
            var sel = this.n2Groups.gridlines.selectAll(".horiz_line")
                .data(gridLines, function (d) {
                    return d.obj.id;
                });

            var gEnter = sel.enter().append("g")
                .attr("class", "horiz_line")
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(0," + (n2Dy0 * (d.i - enterIndex)) + ")";
                    var roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index0 = roc.rootIndex0 - zoomedElement.rootIndex0;
                        return "translate(0," + (n2Dy0 * index0) + ")";
                    }
                    throw("enter transform not found");
                });
            gEnter.append("line")
                .attr("x2", WIDTH_N2_PX);

            var gUpdate = gEnter.merge(sel).transition(sharedTransition)
                .attr("transform", function (d) {
                    return "translate(0," + (n2Dy * d.i) + ")";
                });
            gUpdate.select("line")
                .attr("x2", WIDTH_N2_PX);

            var nodeExit = sel.exit().transition(sharedTransition)
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(0," + (n2Dy * (d.i - exitIndex)) + ")";
                    var roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index = roc.rootIndex - zoomedElement.rootIndex;
                        return "translate(0," + (n2Dy * index) + ")";
                    }
                    throw("exit transform not found");
                })
                .remove();
        }

        {
            var sel = this.n2Groups.gridlines.selectAll(".vert_line")
                .data(gridLines, function (d) {
                    return d.obj.id;
                });
            var gEnter = sel.enter().append("g")
                .attr("class", "vert_line")
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(" + (n2Dx0 * (d.i - enterIndex)) + ")rotate(-90)";
                    var roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var i0 = roc.rootIndex0 - zoomedElement.rootIndex0;
                        return "translate(" + (n2Dx0 * i0) + ")rotate(-90)";
                    }
                    throw("enter transform not found");
                });
            gEnter.append("line")
                .attr("x1", -HEIGHT_PX);

            var gUpdate = gEnter.merge(sel).transition(sharedTransition)
                .attr("transform", function (d) {
                    return "translate(" + (n2Dx * d.i) + ")rotate(-90)";
                });
            gUpdate.select("line")
                .attr("x1", -HEIGHT_PX);

            var nodeExit = sel.exit().transition(sharedTransition)
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(" + (n2Dx * (d.i - exitIndex)) + ")rotate(-90)";
                    var roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var i = roc.rootIndex - zoomedElement.rootIndex;
                        return "translate(" + (n2Dx * i) + ")rotate(-90)";
                    }
                    throw("exit transform not found");
                })
                .remove();
        }

        {
            var sel = this.n2Groups.componentBoxes.selectAll(".component_box")
                .data(drawableN2ComponentBoxes, function (d) {
                    return d.obj.id;
                });
            var gEnter = sel.enter().append("g")
                .attr("class", "component_box")
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(" + (n2Dx0 * (d.startI - enterIndex)) + "," + (n2Dy0 * (d.startI - enterIndex)) + ")";
                    var roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index0 = roc.rootIndex0 - zoomedElement.rootIndex0;
                        return "translate(" + (n2Dx0 * index0) + "," + (n2Dy0 * index0) + ")";
                    }
                    throw("enter transform not found");
                });

            gEnter.append("rect")
                .attr("width", function (d) {
                    if (lastClickWasLeft) return n2Dx0 * (1 + d.stopI - d.startI);
                    return n2Dx0;
                })
                .attr("height", function (d) {
                    if (lastClickWasLeft) return n2Dy0 * (1 + d.stopI - d.startI);
                    return n2Dy0;
                });

            var gUpdate = gEnter.merge(sel).transition(sharedTransition)
                .attr("transform", function (d) {
                    return "translate(" + (n2Dx * d.startI) + "," + (n2Dy * d.startI) + ")";
                });

            gUpdate.select("rect")
                .attr("width", function (d) {
                    return n2Dx * (1 + d.stopI - d.startI);
                })
                .attr("height", function (d) {
                    return n2Dy * (1 + d.stopI - d.startI);
                });


            var nodeExit = sel.exit().transition(sharedTransition)
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(" + (n2Dx * (d.startI - exitIndex)) + "," + (n2Dy * (d.startI - exitIndex)) + ")";
                    var roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index = roc.rootIndex - zoomedElement.rootIndex;
                        return "translate(" + (n2Dx * index) + "," + (n2Dy * index) + ")";
                    }
                    throw("exit transform not found");
                })
                .remove();

            nodeExit.select("rect")
                .attr("width", function (d) {
                    if (lastClickWasLeft) return n2Dx * (1 + d.stopI - d.startI);
                    return n2Dx;
                })
                .attr("height", function (d) {
                    if (lastClickWasLeft) return n2Dy * (1 + d.stopI - d.startI);
                    return n2Dy;
                });
        }
    }
}

function getOnDiagonalCellColor(d) {
    let rt = n2Diag.matrix.nodes[d.col];
    if (rt === undefined) { console.log(d); }
    if (rt.isMinimized) return N2Style.color.collapsed;
    if (rt.type === "param") return N2Style.color.param;
    if (rt.type === "unconnected_param") return N2Style.color.unconnectedParam;
    return (rt.implicit) ? N2Style.color.unknownImplicit : N2Style.color.unknownExplicit;
}