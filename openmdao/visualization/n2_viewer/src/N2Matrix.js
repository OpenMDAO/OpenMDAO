/**
 * Use the model tree to build a matrix, display, and perform operations with it.
 * @typedef N2Matrix
 * @property {N2TreeNodes[]} nodes Reference to nodes that will be drawn.
 * @property {number} levelOfDetailThreshold Don't draw elements below this size in pixels.
 * @property {Object} n2Groups References to <g> SVG elements managed by N2Diagram.
 */
class N2Matrix {
    /**
     * Render the matrix of visible elements in the model.
     * @param {N2TreeNodes[]} visibleNodes Nodes that will be drawn.
     * @param {ModelData} model The pre-processed model data.
     * @param {N2Layout} layout Pre-computed layout of the diagram.
     * @param {Object} n2Groups References to <g> SVG elements created by N2Diagram.
     * @param {Object} [prevNodeSize = {'width': 0, 'height': 0}] Previous node
     *  width & height for transition purposes.
     */
    constructor(visibleNodes, model, layout, n2Groups,
        prevNodeSize = {'width': 0, 'height': 0}) {
        this.nodes = visibleNodes;
        this.layout = layout;
        this.n2Groups = n2Groups;

        this.previousNodeSize = prevNodeSize;
        this.nodeSize = {
            'width': layout.size.diagram.width / this.nodes.length,
            'height': layout.size.diagram.height / this.nodes.length,
        }

        this.updateLevelOfDetailThreshold(layout.size.diagram.height);
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
        if (this.matrix[row] && this.matrix[row][col]) { return true; }
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
        let u0 = this.previousNodeSize.width * .5,
            v0 = this.previousNodeSize.height * .5,
            u = this.nodeSize.width * .5,
            v = this.nodeSize.height * .5; //(0,0) = center of cell... (u,v) = bottom right of cell... (-u,-v) = top left of cell

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
                    if (lastClickWasLeft) return "translate(" + (this.previousNodeSize.width * (d.col - enterIndex) + u0) + "," + (this.previousNodeSize.height * (d.row - enterIndex) + v0) + ")";
                    var roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index0 = roc.rootIndex0 - zoomedElement.rootIndex0;
                        return "translate(" + (this.previousNodeSize.width * index0 + u0) + "," + (this.previousNodeSize.height * index0 + v0) + ")";
                    }
                    throw("enter transform not found");
                }.bind(this));
            drawFunctions[i](gEnter, u0, v0, (i < 3) ? getOnDiagonalCellColor : N2Style.color.connection, false)
                .on("mouseover", (i < 3) ? mouseOverOnDiagN2 : mouseOverOffDiagN2)
                .on("mouseleave", mouseOutN2)
                .on("click", mouseClickN2);


            var gUpdate = gEnter.merge(sel).transition(sharedTransition)
                .attr("transform", function (d) {
                    return "translate(" + (this.nodeSize.width * (d.col) + u) + "," + (this.nodeSize.height * (d.row) + v) + ")";
                }.bind(this));
            drawFunctions[i](gUpdate, u, v, (i < 3) ? getOnDiagonalCellColor : N2Style.color.connection, true);


            var nodeExit = sel.exit().transition(sharedTransition)
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(" + (this.nodeSize.width * (d.col - exitIndex) + u) + "," + (this.nodeSize.height * (d.row - exitIndex) + v) + ")";
                    var roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index = roc.rootIndex - zoomedElement.rootIndex;
                        return "translate(" + (this.nodeSize.width * index + u) + "," + (this.nodeSize.height * index + v) + ")";
                    }
                    throw("exit transform not found");
                }.bind(this))
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
                    if (lastClickWasLeft) return "translate(0," + (this.previousNodeSize.height * (d.i - enterIndex)) + ")";
                    var roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index0 = roc.rootIndex0 - zoomedElement.rootIndex0;
                        return "translate(0," + (this.previousNodeSize.height * index0) + ")";
                    }
                    throw("enter transform not found");
                }.bind(this));
            gEnter.append("line")
                .attr("x2", this.layout.size.diagram.width);

            var gUpdate = gEnter.merge(sel).transition(sharedTransition)
                .attr("transform", function (d) {
                    return "translate(0," + (this.nodeSize.height * d.i) + ")";
                }.bind(this));
            gUpdate.select("line")
                .attr("x2", this.layout.size.diagram.width);

            var nodeExit = sel.exit().transition(sharedTransition)
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(0," + (this.nodeSize.height * (d.i - exitIndex)) + ")";
                    var roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index = roc.rootIndex - zoomedElement.rootIndex;
                        return "translate(0," + (this.nodeSize.height * index) + ")";
                    }
                    throw("exit transform not found");
                }.bind(this))
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
                    if (lastClickWasLeft) return "translate(" + (this.previousNodeSize.width * (d.i - enterIndex)) + ")rotate(-90)";
                    var roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var i0 = roc.rootIndex0 - zoomedElement.rootIndex0;
                        return "translate(" + (this.previousNodeSize.width * i0) + ")rotate(-90)";
                    }
                    throw("enter transform not found");
                }.bind(this));
            gEnter.append("line")
                .attr("x1", -this.layout.size.diagram.height);

            var gUpdate = gEnter.merge(sel).transition(sharedTransition)
                .attr("transform", function (d) {
                    return "translate(" + (this.nodeSize.width * d.i) + ")rotate(-90)";
                }.bind(this));
            gUpdate.select("line")
                .attr("x1", -this.layout.size.diagram.height);

            var nodeExit = sel.exit().transition(sharedTransition)
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(" + (this.nodeSize.width * (d.i - exitIndex)) + ")rotate(-90)";
                    var roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var i = roc.rootIndex - zoomedElement.rootIndex;
                        return "translate(" + (this.nodeSize.width * i) + ")rotate(-90)";
                    }
                    throw("exit transform not found");
                }.bind(this))
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
                    if (lastClickWasLeft) return "translate(" + (this.previousNodeSize.width * (d.startI - enterIndex)) + "," + (this.previousNodeSize.height * (d.startI - enterIndex)) + ")";
                    var roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index0 = roc.rootIndex0 - zoomedElement.rootIndex0;
                        return "translate(" + (this.previousNodeSize.width * index0) + "," + (this.previousNodeSize.height * index0) + ")";
                    }
                    throw("enter transform not found");
                }.bind(this));

            gEnter.append("rect")
                .attr("width", function (d) {
                    if (lastClickWasLeft) return this.previousNodeSize.width * (1 + d.stopI - d.startI);
                    return this.previousNodeSize.width;
                }.bind(this))
                .attr("height", function (d) {
                    if (lastClickWasLeft) return this.previousNodeSize.height * (1 + d.stopI - d.startI);
                    return this.previousNodeSize.height;
                }.bind(this));

            var gUpdate = gEnter.merge(sel).transition(sharedTransition)
                .attr("transform", function (d) {
                    return "translate(" + (this.nodeSize.width * d.startI) + "," + (this.nodeSize.height * d.startI) + ")";
                }.bind(this));

            gUpdate.select("rect")
                .attr("width", function (d) {
                    return this.nodeSize.width * (1 + d.stopI - d.startI);
                }.bind(this))
                .attr("height", function (d) {
                    return this.nodeSize.height * (1 + d.stopI - d.startI);
                }.bind(this));


            var nodeExit = sel.exit().transition(sharedTransition)
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(" + (this.nodeSize.width * (d.startI - exitIndex)) + "," + (this.nodeSize.height * (d.startI - exitIndex)) + ")";
                    var roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index = roc.rootIndex - zoomedElement.rootIndex;
                        return "translate(" + (this.nodeSize.width * index) + "," + (this.nodeSize.height * index) + ")";
                    }
                    throw("exit transform not found");
                }.bind(this))
                .remove();

            nodeExit.select("rect")
                .attr("width", function (d) {
                    if (lastClickWasLeft) return this.nodeSize.width * (1 + d.stopI - d.startI);
                    return this.nodeSize.width;
                }.bind(this))
                .attr("height", function (d) {
                    if (lastClickWasLeft) return this.nodeSize.height * (1 + d.stopI - d.startI);
                    return this.nodeSize.height;
                }.bind(this));
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