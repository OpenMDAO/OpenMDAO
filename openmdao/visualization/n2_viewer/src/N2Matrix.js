/**
 * Use the model tree to build a matrix, display, and perform
 * operations with it.
 */
class N2Matrix {

    constructor(nodes) {
        this.nodes = nodes;

        n2Dx0 = n2Dx;
        n2Dy0 = n2Dy;

        n2Dx = WIDTH_N2_PX / this.nodes.length;
        n2Dy = HEIGHT_PX / this.nodes.length;

        this.buildStructure();
        this.determineSymbolTypes();
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
     * Set up nested objects resembling a two-dimensional array as the
     * matrix, but not an actual two dimensional array because most of
     * it would be unused.
     * @param {Object[]} nodes The pre-discovered leaves in the model.
     */
    buildStructure() {
        this.matrix = {};

        if (this.nodes.length >= LEVEL_OF_DETAIL_THRESHOLD) return;

        for (let srcIdx = 0; srcIdx < this.nodes.length; ++srcIdx) {
            let srcObj = this.nodes[srcIdx];

            // These nodes are on the diagonal.
            if (!this.exists(srcIdx)) { this.matrix[srcIdx] = {}; }
            this.matrix[srcIdx][srcIdx] = {
                "row": srcIdx,
                "col": srcIdx,
                "obj": srcObj,
                "id": srcObj.id + "_" + srcObj.id
            };

            let targets = srcObj.targetsParamView;

            for (let tgtObj of targets) {
                let tgtIdx = indexFor(this.nodes, tgtObj);
                if (tgtIdx != -1) {
                    this.matrix[srcIdx][tgtIdx] = {
                        "row": srcIdx,
                        "col": tgtIdx,
                        "obj": srcObj,
                        "id": srcObj.id + "_" + tgtObj.id
                    };
                }
            }

            if (srcObj.type === "param" || srcObj.type === "unconnected_param") {
                for (let j = srcIdx + 1; j < this.nodes.length; ++j) {
                    let tgtObj = this.nodes[j];
                    if (srcObj.parentComponent !== tgtObj.parentComponent) break;

                    if (tgtObj.type === "unknown") {
                        let tgtIdx = j;
                        this.matrix[srcIdx][tgtIdx] = {
                            "row": srcIdx,
                            "col": tgtIdx,
                            "obj": srcObj,
                            "id": srcObj.id + "_" + tgtObj.id
                        };
                    }
                }
            }
        }
    }

    /** Decide what object ach node will be drawn as, based on its
     * location in the matrix, type, source, target, and/or other conditions.
     */
    determineSymbolTypes() {
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

        let regex = /^unknown$|^param$|^unconnected_param$/;

        for (let row in this.matrix) {
            for (let col in this.matrix[row]) {
                let d = this.matrix[row][col];
                let tgtObj = this.nodes[d.col],
                    srcObj = this.nodes[d.row];

                if (d.col == d.row) { // on diagonal
                    if (srcObj.type === "subsystem") { // group
                        this.symbols.group.push(d);
                    } else if (srcObj.type.match(regex)) {
                        if (srcObj.dtype === "ndarray") { // vector
                            this.symbols.vector.push(d);
                        } else { // scalar
                            this.symbols.scalar.push(d);
                        }
                    }
                }
                else if (srcObj.type === "subsystem") {
                    if (tgtObj.type === "subsystem") { // groupGroup
                        this.symbols.groupGroup.push(d);
                    }
                    else if (tgtObj.type.match(regex)) {
                        if (tgtObj.dtype === "ndarray") { // groupVector
                            this.symbols.groupVector.push(d);
                        }
                        else { // groupScalar
                            this.symbols.groupScalar.push(d);
                        }
                    }
                }
                else if (srcObj.type.match(regex)) {
                    if (srcObj.dtype === "ndarray") {
                        if (tgtObj.type.match(regex)) {
                            if (tgtObj.dtype === "ndarray" ||
                                tgtObj.type.match(/^param$|^unconnected_param$/)) { // vectorVector
                                this.symbols.vectorVector.push(d);

                                let partials_string = tgtObj.absPathName + " > " + srcObj.absPathName;
                                if (modelData.declare_partials_list.includes(partials_string)) {
                                    this.symbols.declaredPartials.vectorVector.push(d);
                                }

                            }
                            else { // vectorScalar
                                this.symbols.vectorScalar.push(d);
                                let partials_string = tgtObj.absPathName + " > " + srcObj.absPathName;
                                if (modelData.declare_partials_list.includes(partials_string)) {
                                    this.symbols.declaredPartials.vectorScalar.push(d);
                                }
                            }
                        }
                        else if (tgtObj.type === "subsystem") { // vectorGroup
                            this.symbols.vectorGroup.push(d);
                        }
                    }
                    else {
                        if (tgtObj.type.match(regex)) {
                            if (tgtObj.dtype === "ndarray") { // scalarVector
                                this.symbols.scalarVector.push(d);
                                let partials_string = tgtObj.absPathName + " > " + srcObj.absPathName;
                                if (modelData.declare_partials_list.includes(partials_string)) {
                                    this.symbols.declaredPartials.scalarVector.push(d);
                                }
                            }
                            else { // scalarScalar
                                this.symbols.scalarScalar.push(d);
                                let partials_string = tgtObj.absPathName + " > " + srcObj.absPathName;
                                if (modelData.declare_partials_list.includes(partials_string)) {
                                    this.symbols.declaredPartials.scalarScalar.push(d);
                                }
                            }
                        }
                        else if (tgtObj.type === "subsystem") { // scalarGroup
                            this.symbols.scalarGroup.push(d);
                        }
                    }
                }
            }
        }
    }

    drawingPrep() {
        let currentBox = { "startI": 0, "stopI": 0 };

        d3RightTextNodesArrayZoomedBoxInfo = [currentBox];

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
        if (this.nodes.length < LEVEL_OF_DETAIL_THRESHOLD) {
            for (let i = 0; i < this.nodes.length; ++i) {
                let obj = this.nodes[i];
                var gl = { "i": i, "obj": obj };
                gridLines.push(gl);
            }
        }
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
            var sel = n2ElementsGroup.selectAll("." + classes[i])
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
                    alert("error: enter transform not found");
                });
            drawFunctions[i](gEnter, u0, v0, (i < 3) ? getOnDiagonalCellColor : CONNECTION_COLOR, false)
                .on("mouseover", (i < 3) ? mouseOverOnDiagN2 : mouseOverOffDiagN2)
                .on("mouseleave", mouseOutN2)
                .on("click", mouseClickN2);


            var gUpdate = gEnter.merge(sel).transition(sharedTransition)
                .attr("transform", function (d) {
                    return "translate(" + (n2Dx * (d.col) + u) + "," + (n2Dy * (d.row) + v) + ")";
                });
            drawFunctions[i](gUpdate, u, v, (i < 3) ? getOnDiagonalCellColor : CONNECTION_COLOR, true);


            var nodeExit = sel.exit().transition(sharedTransition)
                .attr("transform", function (d) {
                    if (lastClickWasLeft) return "translate(" + (n2Dx * (d.col - exitIndex) + u) + "," + (n2Dy * (d.row - exitIndex) + v) + ")";
                    var roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                    if (roc) {
                        var index = roc.rootIndex - zoomedElement.rootIndex;
                        return "translate(" + (n2Dx * index + u) + "," + (n2Dy * index + v) + ")";
                    }
                    alert("error: exit transform not found");
                })
                .remove();
            drawFunctions[i](nodeExit, u, v, (i < 3) ? getOnDiagonalCellColor : CONNECTION_COLOR, true);
        }

        {
            var sel = n2GridLinesGroup.selectAll(".horiz_line")
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
                    alert("error: enter transform not found");
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
                    alert("error: exit transform not found");
                })
                .remove();
        }

        {
            var sel = n2GridLinesGroup.selectAll(".vert_line")
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
                    alert("error: enter transform not found");
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
                    alert("error: exit transform not found");
                })
                .remove();
        }

        {
            var sel = n2ComponentBoxesGroup.selectAll(".component_box")
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
                    alert("error: enter transform not found");
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
                    alert("error: exit transform not found");
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
    let rt = matrix.nodes[d.col];
    if (rt === undefined) { console.log(d); }
    if (rt.isMinimized) return COLLAPSED_COLOR;
    if (rt.type === "param") return PARAM_COLOR;
    if (rt.type === "unconnected_param") return UNCONNECTED_PARAM_COLOR
    return (rt.implicit) ? UNKNOWN_IMPLICIT_COLOR : UNKNOWN_EXPLICIT_COLOR;
}