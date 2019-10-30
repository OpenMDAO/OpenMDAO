/**
 * Use the model tree to build the matrix of parameters and connections, display, and
 * perform operations with it.
 * @typedef N2Matrix
 * @property {N2TreeNodes[]} nodes Reference to nodes that will be drawn.
 * @property {ModelData} model Reference to the pre-processed model.
 * @property {N2Layout} layout Reference to object managing columns widths and such.
 * @property {Object} n2Groups References to <g> SVG elements created by N2Diagram.
 * @property {number} levelOfDetailThreshold Don't draw elements below this size in pixels.
 * @property {Object} nodeSize Width and height of each node in the matrix.
 * @property {Object} prevNodeSize Width and height of each node in the previous matrix.
 * @property {Object} cellDims nodeSize with computed coordinates.
 * @property {Object} prevCellDims prevNodeSize with computed coordinates.
 * @property {Object[][]} grid Object keys corresponding to rows and columns.
 * @property {Array} allCells One-dimensional array of all cells, for D3 processing.
 * @property {Array} boxInfo Component box dimensions.
 */
class N2Matrix {
    /**
     * Render the matrix of visible elements in the model.
     * @param {N2TreeNodes[]} visibleNodes Nodes that will be drawn.
     * @param {ModelData} model The pre-processed model data.
     * @param {N2Layout} layout Pre-computed layout of the diagram.
     * @param {N2MatrixCell[][]} grid N2MatrixCell objects in row,col order.
     * @param {Object} n2Groups References to <g> SVG elements created by N2Diagram.
     * @param {Object} [prevNodeSize = {'width': 0, 'height': 0}] Previous node
     *  width & height for transition purposes.
     */
    constructor(visibleNodes, model, layout, n2Groups,
        prevNodeSize = { 'width': 0, 'height': 0 }) {

        this.nodes = visibleNodes;
        this.layout = layout;
        this.n2Groups = n2Groups;

        this.prevNodeSize = prevNodeSize;
        this.nodeSize = {
            'width': layout.size.diagram.width / this.nodes.length,
            'height': layout.size.diagram.height / this.nodes.length,
        }

        this.cellDims = {
            'size': this.nodeSize,
            'bottomRight': {
                'x': this.nodeSize.width * .5,
                'y': this.nodeSize.height * .5
            },
            'topLeft': {
                'x': -(this.nodeSize.width * .5),
                'y': -(this.nodeSize.width * .5)
            }
        }

        this.prevCellDims = {
            'size': this.prevNodeSize,
            'bottomRight': {
                'x': this.prevNodeSize.width * .5,
                'y': this.prevNodeSize.height * .5
            },
            'topLeft': {
                'x': -(this.prevNodeSize.width * .5),
                'y': -(this.prevNodeSize.width * .5)
            }
        }

        this.updateLevelOfDetailThreshold(layout.size.diagram.height);

        console.time('N2Matrix._buildGrid');
        this._buildGrid(model);
        console.timeEnd('N2Matrix._buildGrid');

        console.time('N2Matrix._setupComponentBoxesAndGridLines');
        this._setupComponentBoxesAndGridLines();
        console.timeEnd('N2Matrix._setupComponentBoxesAndGridLines');

    }

    /**
     * Determine if a node exists at the specified location.
     * @param {number} row Row number of the node
     * @param {number} col Column number of the node
     * @returns False if the row doesn't exist or a column doesn't exist
     *  in the row; true otherwise.
     */
    exists(row, col) {
        if (this.grid[row] && this.grid[row][col]) { return true; }
        return false;
    }

    /**
     * Safe method to access a node/cell at the specified location.
     * Renamed from node() which is confusing with D3.
     * @param {number} row Row number of the node
     * @param {number} col Column number of the node
     * @param {boolean} doThrow Whether to throw an exception if node undefined.
     * @returns {N2MatrixCell} The node if it exists, undefined otherwise.
     */
    cell(row, col, doThrow = false) {
        if (this.exists(row, col)) {
            return this.grid[row][col];
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
     * Add a cell to the grid Object and allCells array, with special
     * handling for declared partials.
     * @param {N2MatrixCell} newCell Cell created in _buildGrid().
     * @param {number} row Index of the row in the grid to place the cell.
     * @param {number} col Index of the column in the grid to place the cell.
     */
    _addCell(newCell, row, col) {
        if (modelData.options.use_declare_partial_info &&
            newCell.symbolType.potentialDeclaredPartial &&
            !newCell.symbolType.declaredPartial) {

            return;
        }

        this.grid[row][col] = newCell;
        this.allCells.push(newCell);
    }

    /**
     * Set up N2MatrixCell arrays resembling a two-dimensional grid as the
     * matrix, but not an actual two dimensional array because most of
     * it would be unused.
     */
    _buildGrid(model) {
        this.grid = {};
        this.allCells = [];

        if (this.nodes.length >= this.levelOfDetailThreshold) return;

        for (let srcIdx = 0; srcIdx < this.nodes.length; ++srcIdx) {
            let srcObj = this.nodes[srcIdx];

            // New row
            if (!this.exists(srcIdx)) this.grid[srcIdx] = {};

            // On the diagonal
            let newCell = new N2MatrixCell(srcIdx, srcIdx, srcObj, srcObj, model,
                this.cellDims, this.prevCellDims);
            this._addCell(newCell, srcIdx, srcIdx);

            let targets = srcObj.targetsParamView;

            for (let tgtObj of targets) {
                let tgtIdx = indexFor(this.nodes, tgtObj);
                if (tgtIdx != -1) {
                    let newCell = new N2MatrixCell(srcIdx, tgtIdx, srcObj, tgtObj, model,
                        this.cellDims, this.prevCellDims);
                    this._addCell(newCell, srcIdx, tgtIdx);
                }
            }

            // Solver nodes
            if (srcObj.isParam()) {
                for (let j = srcIdx + 1; j < this.nodes.length; ++j) {
                    let tgtObj = this.nodes[j];
                    if (srcObj.parentComponent !== tgtObj.parentComponent) break;

                    if (tgtObj.isUnknown()) {
                        let tgtIdx = j;
                        let newCell = new N2MatrixCell(srcIdx, tgtIdx, srcObj, tgtObj,
                            model, this.cellDims, this.prevCellDims);
                        this._addCell(newCell, srcIdx, tgtIdx);
                    }
                }
            }
        }
    }

    /**
     * Determine the size of the boxes that will border the parameters of each component.
     */
    _setupComponentBoxesAndGridLines() {
        let currentBox = { "startI": 0, "stopI": 0 };

        this.boxInfo = [currentBox];

        // Find which component box each of the parameters belong in,
        // while finding the bounds of that box. Top and bottom
        // rows recorded for each node in this.boxInfo[].
        for (let ri = 1; ri < this.nodes.length; ++ri) {
            let curNode = this.nodes[ri];
            let startINode = this.nodes[currentBox.startI];
            if (startINode.parentComponent &&
                curNode.parentComponent &&
                startINode.parentComponent === curNode.parentComponent) {
                ++currentBox.stopI;
            }
            else {
                currentBox = { "startI": ri, "stopI": ri };
            }
            this.boxInfo.push(currentBox);
        }

        // Step through this.boxInfo[] and record one set of dimensions
        // for each box in this.componentBoxInfo[].
        this.componentBoxInfo = [];
        for (let i = 0; i < this.boxInfo.length; ++i) {
            let box = this.boxInfo[i];
            if (box.startI == box.stopI) continue;
            let curNode = this.nodes[box.startI];
            if (!curNode.parentComponent) {
                throw "Parent component not found in box.";
            }
            box.obj = curNode.parentComponent;
            i = box.stopI;
            this.componentBoxInfo.push(box);
        }

        //do this so you save old index for the exit()
        this.gridLines = [];
        if (this.nodes.length < this.levelOfDetailThreshold) {
            for (let i = 0; i < this.nodes.length; ++i) {
                let obj = this.nodes[i];
                let gl = { "i": i, "obj": obj };
                this.gridLines.push(gl);
            }
        }
    }

    /**
     * Create an SVG group for each visible element, and have the element
     * render its shape in it. Move the groups around to their correct
     * positions, providing an animated transition from the previous
     * rendering.
     */
    _drawCells() {
        let self = this; // For callbacks that change "this"

        let selection = this.n2Groups.elements.selectAll('.n2cell')
            .data(this.allCells, d => d.id);

        // Use D3 to join N2MatrixCells to SVG groups, and render shapes in them.
        let gEnter = selection.enter().append('g')
            .attr('class', 'n2cell')
            .attr('transform', function (d) {
                if (lastClickWasLeft) {
                    return 'translate(' +
                        (self.prevCellDims.size.width * (d.col - enterIndex) +
                            self.prevCellDims.bottomRight.x) + ',' +
                        (self.prevCellDims.size.height * (d.row - enterIndex) +
                            self.prevCellDims.bottomRight.y) + ')';
                }

                let roc = (d.obj && FindRootOfChangeFunction) ?
                    FindRootOfChangeFunction(d.obj) : null;

                if (roc) {
                    let prevIdx = roc.prevRootIndex -
                        self.layout.zoomedElement.prevRootIndex;
                    return 'translate(' + (self.prevCellDims.size.width * prevIdx +
                        self.prevCellDims.bottomRight.x) + ',' +
                        (self.prevCellDims.size.height * prevIdx +
                            self.prevCellDims.bottomRight.y) + ')';
                }
                throw ('Enter transform not found');
            })
            .each(function (d) {
                d.renderer.renderPrevious(this)
                    .on('mouseover', d.mouseover())
                    .on('mouseleave', n2MouseFuncs.out)
                    .on('click', n2MouseFuncs.click);
            });

        gEnter.merge(selection)
            .transition(sharedTransition)
            .attr('transform', function (d) {
                return 'translate(' + (self.cellDims.size.width * (d.col) +
                    self.cellDims.bottomRight.x) + ',' +
                    (self.cellDims.size.height * (d.row) +
                        self.cellDims.bottomRight.y) + ')';
            })
            .each(function (d) { d.renderer.updateCurrent(this) });

        selection.exit()
            .transition(sharedTransition)
            .attr('transform', function (d) {
                if (lastClickWasLeft)
                    return 'translate(' + (self.cellDims.size.width *
                        (d.col - exitIndex) + self.cellDims.bottomRight.x) + ',' +
                        (self.cellDims.size.height * (d.row - exitIndex) +
                            self.cellDims.bottomRight.y) + ')';

                let roc = (d.obj && FindRootOfChangeFunction) ?
                    FindRootOfChangeFunction(d.obj) : null;

                if (roc) {
                    let index = roc.rootIndex - self.layout.zoomedElement.rootIndex;
                    return 'translate(' + (self.cellDims.size.width * index +
                        self.cellDims.bottomRight.x) + ',' +
                        (self.cellDims.size.height * index + self.cellDims.bottomRight.y) + ')';
                }
                throw ('Exit transform not found');
            })
            .each(function (d) { d.renderer.updateCurrent(this) })
            .remove();
    }

    /** Draw a line above every row in the matrix. */
    _drawHorizontalLines() {
        let sel = this.n2Groups.gridlines.selectAll('.horiz_line')
            .data(this.gridLines, function (d) {
                return d.obj.id;
            });

        let gEnter = sel.enter().append('g')
            .attr('class', 'horiz_line')
            .attr('transform', function (d) {
                if (lastClickWasLeft) return 'translate(0,' +
                    (this.prevCellDims.size.height * (d.i - enterIndex)) + ')';
                let roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let index0 = roc.prevRootIndex - this.layout.zoomedElement.prevRootIndex;
                    return 'translate(0,' + (this.prevCellDims.size.height * index0) + ')';
                }
                throw ('enter transform not found');
            }.bind(this));
        gEnter.append('line')
            .attr('x2', this.layout.size.diagram.width);

        let gUpdate = gEnter.merge(sel).transition(sharedTransition)
            .attr('transform', function (d) {
                return 'translate(0,' + (this.cellDims.size.height * d.i) + ')';
            }.bind(this));
        gUpdate.select('line')
            .attr('x2', this.layout.size.diagram.width);

        sel.exit().transition(sharedTransition)
            .attr('transform', function (d) {
                if (lastClickWasLeft) return 'translate(0,' + (this.cellDims.size.height * (d.i - exitIndex)) + ')';
                let roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let index = roc.rootIndex - this.layout.zoomedElement.rootIndex;
                    return 'translate(0,' + (this.cellDims.size.height * index) + ')';
                }
                throw ('exit transform not found');
            }.bind(this))
            .remove();
    }

    /** Draw a vertical line for every column in the matrix. */
    _drawVerticalLines() {
        let sel = this.n2Groups.gridlines.selectAll(".vert_line")
            .data(this.gridLines, function (d) {
                return d.obj.id;
            });
        let gEnter = sel.enter().append("g")
            .attr("class", "vert_line")
            .attr("transform", function (d) {
                if (lastClickWasLeft) return "translate(" + (this.prevCellDims.size.width * (d.i - enterIndex)) + ")rotate(-90)";
                let roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let i0 = roc.prevRootIndex - this.layout.zoomedElement.prevRootIndex;
                    return "translate(" + (this.prevCellDims.size.width * i0) + ")rotate(-90)";
                }
                throw ("enter transform not found");
            }.bind(this));
        gEnter.append("line")
            .attr("x1", -this.layout.size.diagram.height);

        let gUpdate = gEnter.merge(sel).transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + (this.cellDims.size.width * d.i) + ")rotate(-90)";
            }.bind(this));
        gUpdate.select("line")
            .attr("x1", -this.layout.size.diagram.height);

        sel.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                if (lastClickWasLeft) return "translate(" + (this.cellDims.size.width * (d.i - exitIndex)) + ")rotate(-90)";
                let roc = (FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let i = roc.rootIndex - this.layout.zoomedElement.rootIndex;
                    return "translate(" + (this.cellDims.size.width * i) + ")rotate(-90)";
                }
                throw ("exit transform not found");
            }.bind(this))
            .remove();
    }

    /** Draw boxes around the cells associated with each component. */
    _drawComponentBoxes() {
        let sel = this.n2Groups.componentBoxes.selectAll(".component_box")
            .data(this.componentBoxInfo, function (d) {
                return d.obj.id;
            });
        let gEnter = sel.enter().append("g")
            .attr("class", "component_box")
            .attr("transform", function (d) {
                if (lastClickWasLeft) return "translate(" + (this.prevCellDims.size.width * (d.startI - enterIndex)) + "," + (this.prevCellDims.size.height * (d.startI - enterIndex)) + ")";
                let roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let index0 = roc.prevRootIndex - this.layout.zoomedElement.prevRootIndex;
                    return "translate(" + (this.prevCellDims.size.width * index0) + "," + (this.prevCellDims.size.height * index0) + ")";
                }
                throw ("enter transform not found");
            }.bind(this));

        gEnter.append("rect")
            .attr("width", function (d) {
                if (lastClickWasLeft) return this.prevCellDims.size.width * (1 + d.stopI - d.startI);
                return this.prevCellDims.size.width;
            }.bind(this))
            .attr("height", function (d) {
                if (lastClickWasLeft) return this.prevCellDims.size.height * (1 + d.stopI - d.startI);
                return this.prevCellDims.size.height;
            }.bind(this));

        let gUpdate = gEnter.merge(sel).transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + (this.cellDims.size.width * d.startI) + "," + (this.cellDims.size.height * d.startI) + ")";
            }.bind(this));

        gUpdate.select("rect")
            .attr("width", function (d) {
                return this.cellDims.size.width * (1 + d.stopI - d.startI);
            }.bind(this))
            .attr("height", function (d) {
                return this.cellDims.size.height * (1 + d.stopI - d.startI);
            }.bind(this));


        let nodeExit = sel.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                if (lastClickWasLeft) return "translate(" + (this.cellDims.size.width * (d.startI - exitIndex)) + "," + (this.cellDims.size.height * (d.startI - exitIndex)) + ")";
                let roc = (d.obj && FindRootOfChangeFunction) ? FindRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let index = roc.rootIndex - this.layout.zoomedElement.rootIndex;
                    return "translate(" + (this.cellDims.size.width * index) + "," + (this.cellDims.size.height * index) + ")";
                }
                throw ("exit transform not found");
            }.bind(this))
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                if (lastClickWasLeft) return this.cellDims.size.width * (1 + d.stopI - d.startI);
                return this.cellDims.size.width;
            }.bind(this))
            .attr("height", function (d) {
                if (lastClickWasLeft) return this.cellDims.size.height * (1 + d.stopI - d.startI);
                return this.cellDims.size.height;
            }.bind(this));
    }

    /** Add all the visible elements to the matrix. */
    draw() {
        console.time('N2Matrix.draw');

        this._drawCells();
        this._drawHorizontalLines();
        this._drawVerticalLines();
        this._drawComponentBoxes();

        console.timeEnd('N2Matrix.draw');

    }

    /**
     * When the mouse goes over a cell that's on the diagonal, look for and
     * draw connection arrows, and highlight variable names.
     * @param {N2MatrixCell} cell The cell the event occured on.
     * TODO: move DrawRect() someplace non-global
     */
    mouseOverOnDiagonal(cell) {
        let leftTextWidthHovered = this.layout.visibleNodes[cell.row].nameWidthPx;

        // Loop over all elements in the matrix looking for other cells in the same column as
        let lineWidth = Math.min(5, this.nodeSize.width * .5,
            this.nodeSize.height * .5);

        DrawRect(-leftTextWidthHovered - this.layout.size.partitionTreeGap,
            this.nodeSize.height * cell.row, leftTextWidthHovered,
            this.nodeSize.height, N2Style.color.highlightHovered); //highlight hovered

        for (let col = 0; col < this.layout.visibleNodes.length; ++col) {
            let leftTextWidthDependency = this.layout.visibleNodes[col].nameWidthPx;
            if (this.exists(cell.row, col)) {
                if (col != cell.row) {
                    new N2Arrow({
                        'end': { 'col': col, 'row': col },
                        'start': { 'col': cell.row, 'row': cell.row },
                        'color': N2Style.color.greenArrow,
                        'width': lineWidth
                    }, this.n2Groups, this.nodeSize);

                    //highlight var name
                    DrawRect(-leftTextWidthDependency - this.layout.size.partitionTreeGap,
                        this.nodeSize.height * col, leftTextWidthDependency,
                        this.nodeSize.height, N2Style.color.greenArrow);
                }
            }

            // Now swap row and col
            if (this.exists(col, cell.row)) {
                if (col != cell.row) {
                    new N2Arrow({
                        'start': { 'col': col, 'row': col },
                        'end': { 'col': cell.row, 'row': cell.row },
                        'color': N2Style.color.redArrow,
                        'width': lineWidth
                    }, this.n2Groups, this.nodeSize);

                    //highlight var name
                    DrawRect(-leftTextWidthDependency - this.layout.size.partitionTreeGap,
                        this.nodeSize.height * col, leftTextWidthDependency,
                        this.nodeSize.height, N2Style.color.redArrow);
                }
            }
        }
    }

    /**
     * When the mouse goes over a cell that's not on the diagonal, look for and
     * draw cycle arrows, and highlight variable names.
     * @param {N2MatrixCell} cell The cell the event occured on.
     */
    mouseOverOffDiagonal(cell) {
        let lineWidth = Math.min(5, this.nodeSize.width * .5, this.nodeSize.height * .5);
        let src = this.layout.visibleNodes[cell.row];
        let tgt = this.layout.visibleNodes[cell.col];
        // let boxEnd = this.boxInfo[cell.col]; // not used?

        new N2Arrow({
            'start': { 'col': cell.row, 'row': cell.row },
            'end': { 'col': cell.col, 'row': cell.col },
            'color': N2Style.color.redArrow,
            'width': lineWidth
        }, this.n2Groups, this.nodeSize);

        if (cell.row > cell.col) {
            let targetsWithCycleArrows = [];
            tgt.getObjectsWithCycleArrows(targetsWithCycleArrows);

            for (let twca of targetsWithCycleArrows) {
                for (let ai of twca.cycleArrows) {
                    if (src.hasObject(ai.src)) {
                        for (let si of ai.arrows) {
                            let beginObj = si.begin;
                            let endObj = si.end;
                            let firstBeginIndex = -1, firstEndIndex = -1;

                            // find first begin index
                            for (let mi in this.layout.visibleNodes) {
                                let rtNode = n2Diag.layout.visibleNodes[mi];
                                if (rtNode.hasObject(beginObj)) {
                                    firstBeginIndex = mi;
                                    break;
                                }
                            }
                            if (firstBeginIndex == -1) {
                                throw("Error: first begin index not found");
                            }

                            // find first end index
                            for (let mi in this.layout.visibleNodes) {
                                let rtNode = this.layout.visibleNodes[mi];
                                if (rtNode.hasObject(endObj)) {
                                    firstEndIndex = mi;
                                    break;
                                }
                            }
                            if (firstEndIndex == -1) {
                                throw("Error: first end index not found");
                            }

                            if (firstBeginIndex != firstEndIndex) {
                                DrawArrowsParamView(firstBeginIndex, firstEndIndex, n2Diag.matrix.nodeSize);
                            }
                        }
                    }
                }
            }
        }

        let leftTextWidthR = this.layout.visibleNodes[cell.row].nameWidthPx,
            leftTextWidthC = this.layout.visibleNodes[cell.col].nameWidthPx;

        // highlight var name
        DrawRect(-leftTextWidthR - this.layout.size.partitionTreeGap,
            this.nodeSize.height * cell.row, leftTextWidthR, this.nodeSize.height,
            N2Style.color.redArrow);

        // highlight var name
        DrawRect(-leftTextWidthC - this.layout.size.partitionTreeGap,
            this.nodeSize.height * cell.col, leftTextWidthC, this.nodeSize.height,
            N2Style.color.greenArrow);
    }
}
