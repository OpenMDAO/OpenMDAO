/**
 * Use the model tree to build the matrix of variables and connections, display, and
 * perform operations with it.
 * @typedef N2Matrix
 * @property {N2TreeNodes[]} nodes Reference to nodes that will be drawn.
 * @property {ModelData} model Reference to the pre-processed model.
 * @property {N2Layout} layout Reference to object managing columns widths and such.
 * @property {Object} n2Groups References to <g> SVG elements created by N2Diagram.
 * @property {number} levelOfDetailThreshold Don't draw elements below this size in pixels.
 * @property {Object} nodeSize Width and height of each node in the matrix.
 * @property {Object} prevNodeSize Width and height of each node in the previous matrix.
 * @property {Object[][]} grid Object keys corresponding to rows and columns.
 * @property {Array} visibleCells One-dimensional array of all cells, for D3 processing.
 * @property {Array} boxInfo Component box dimensions.
 */
class N2Matrix {
    /**
     * Render the matrix of visible elements in the model.
     * @param {ModelData} model The pre-processed model data.
     * @param {N2Layout} layout Pre-computed layout of the diagram.
     * @param {Object} n2Groups References to <g> SVG elements created by N2Diagram.
     * @param {N2ArrowManager} arrowMgr Object to create and manage conn. arrows.
     * @param {Boolean} lastClickWasLeft
     * @param {function} findRootOfChangeFunction
     */
    constructor(model, layout, n2Groups, arrowMgr,
        lastClickWasLeft,
        findRootOfChangeFunction,
        prevNodeSize = {
            'width': 0,
            'height': 0
        }) {

        this.model = model;
        this.layout = layout;
        this.diagNodes = layout.visibleNodes;
        this.n2Groups = n2Groups;
        this.arrowMgr = arrowMgr;
        this.lastClickWasLeft = lastClickWasLeft;
        this.findRootOfChangeFunction = findRootOfChangeFunction;

        this.prevNodeSize = prevNodeSize;
        this.nodeSize = {
            'width': layout.size.n2matrix.width / this.diagNodes.length,
            'height': layout.size.n2matrix.height / this.diagNodes.length,
        }
        this.arrowMgr.setNodeSize(this.nodeSize);

        let markerSize = Math.max(2, this.nodeSize.width * .04, this.nodeSize.height * .04);
        d3.select("#arrow").attr("markerWidth", markerSize).attr("markerHeight", markerSize);
        d3.select("#offgridArrow").attr("markerWidth", markerSize * 2).attr("markerHeight", markerSize);

        N2CellRenderer.updateDims(this.nodeSize.width, this.nodeSize.height);
        this.updateLevelOfDetailThreshold(layout.size.n2matrix.height);

        startTimer('N2Matrix._buildGrid');
        this._buildGrid();
        stopTimer('N2Matrix._buildGrid');

        startTimer('N2Matrix._setupComponentBoxesAndGridLines');
        this._setupComponentBoxesAndGridLines();
        stopTimer('N2Matrix._setupComponentBoxesAndGridLines');
    }

    get cellDims() {
        return N2CellRenderer.dims;
    }

    get prevCellDims() {
        return N2CellRenderer.prevDims;
    }

    /**
     * Determine if a node exists at the specified location.
     * @param {number} row Row number of the node
     * @param {number} col Column number of the node
     * @returns False if the row doesn't exist or a column doesn't exist
     *  in the row; true otherwise.
     */
    exists(row, col) {
        if (this.grid[row] && this.grid[row][col]) return true;
        return false;
    }

    /**
     * Make sure the cell is still part of the matrix and not an old one.
     * @param {N2MatrixCell} cell The cell to test.
     * @returns {Boolean} True if this.diagNodes has an object in the
     *   same row and column, and it matches the provided cell.
     */
    cellExists(cell) {
        return (this.exists(cell.row, cell.col) &&
            this.cell(cell.row, cell.col) === cell);
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

    findCellById(cellId) {
        for (const cell of this.visibleCells) {
            if (cell.id == cellId) return cell;
        }

        return undefined;
    }

    /**
     * Given the node ID, determine if one of the cells in the matrix
     * represents it or contains it.
     * @param {Number} nodeId The id of the N2TreeNode to search for
     * @returns {Object} Contains reference to cell if found, and flags describing it.
     */
    findCellByNodeId(nodeId) {
        const node = this.model.nodeIds[nodeId];
        let ret = { 
            'cell': undefined,    // Changed to refer to a related cell if found
            'exactMatch': false,  // True if nodeId matches a cell
            'parentMatch': false, // True if nodeId's ancestor is a cell
            'childMatch': false   // True if nodeId's descendant is a cell
        }

        const debugStr = `${node.absPathName}(${nodeId})`

        // Less expensive to check entire matrix for direct matches first
        for (const row in this.grid) { // Check diagonals only
            const cell = this.grid[row][row];

            // Found directly:
            if (cell.id == N2MatrixCell.makeId(nodeId)) {
                debugInfo(`findCellByNodeId: Found ${debugStr} directly in matrix`)
                ret.cell = cell;
                ret.exactMatch = true
                return ret;
            }
        }

        // Only check for relationships if node not directly visible
        for (const row in this.grid) { // Check diagonals only
            const cell = this.grid[row][row];
            if (node.hasNodeInChildren(cell.obj)) {
                debugInfo(`findCellByNodeId: Found descendant of ${debugStr} in matrix`)
                ret.cell = cell;
                ret.childMatch = true
                return ret;
            }

            if (node.hasParent(cell.obj)) {
                debugInfo(`findCellByNodeId: Found ancestor of ${debugStr} in matrix`)
                ret.cell = cell;
                ret.parentMatch = true
                return ret;
            }
        }
    
        // Shouldn't really get here due to zoomedElement check at top
        debugInfo(`findCellByNodeId: ${debugStr} fell through all checks!`)
        return ret;
    }

    /**
     * Compute the new minimum element size when the diagram height changes.
     * @param {number} height In pixels.
     */
    updateLevelOfDetailThreshold(height) {
        this.levelOfDetailThreshold = height / 3;
    }

    /**
     * Compare the number of visible nodes to the amount allowed by
     * the threshold setting.
     */
    tooMuchDetail() {
        let tooMuch = (this.diagNodes.length >= this.levelOfDetailThreshold);

        if (tooMuch) debugInfo("Too much detail.")

        return tooMuch;
    }

    /**
     * Add a cell to the grid Object and visibleCells array, with special
     * handling for declared partials.
     * @param {number} row Index of the row in the grid to place the cell.
     * @param {number} col Index of the column in the grid to place the cell.
     * @param {N2MatrixCell} newCell Cell created in _buildGrid().
     */
    _addCell(row, col, newCell) {
        if (this.model.useDeclarePartialsList &&
            newCell.symbolType.potentialDeclaredPartial &&
            !newCell.symbolType.declaredPartial) {

            return;
        }

        this.grid[row][col] = newCell;
        this.visibleCells.push(newCell);
    }

    /**
     * For cells that are part of cycles, determine if there are parts
     * of the cycle that are offscreen. If so, record them in the cell.
     * @param {N2MatrixCell} cell The cell to check.
     */
    _findUnseenCycleSources(cell) {
        let node = cell.tgtObj;
        let targetsWithCycleArrows = node.getNodesWithCycleArrows();

        for (let twca of targetsWithCycleArrows) {
            for (let ai of twca.cycleArrows) {
                let found = false;

                // Check visible nodes on the diagonal.
                for (let diagNode of this.diagNodes) {
                    let commonParent = diagNode.nearestCommonParent(ai.src);
                    if (diagNode.hasNode(ai.src, commonParent)) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    for (let tgt of ai.src.targetParentSet) {
                        if (tgt.absPathName == node.absPathName) {
                            debugInfo("Adding offscreen connection from _findUnseenCycleSources().")
                            cell.addOffScreenConn(ai.src, node)
                        }
                    }
                }
            }
        }
    }

    /**
     * Set up N2MatrixCell arrays resembling a two-dimensional grid as the
     * matrix, but not an actual two dimensional array because most of
     * it would be unused.
     */
    _buildGrid() {
        this.visibleCells = [];
        this.grid = {};

        if (this.tooMuchDetail()) return;

        for (let srcIdx = 0; srcIdx < this.diagNodes.length; ++srcIdx) {
            let diagNode = this.diagNodes[srcIdx];

            // New row
            if (!this.grid.propExists(srcIdx)) this.grid[srcIdx] = {};

            // On the diagonal
            let newDiagCell = new N2MatrixCell(srcIdx, srcIdx, diagNode, diagNode, this.model);
            this._addCell(srcIdx, srcIdx, newDiagCell);
            this._findUnseenCycleSources(newDiagCell);

            for (let tgt of diagNode.targetParentSet) {
                let tgtIdx = indexFor(this.diagNodes, tgt);

                if (tgtIdx != -1) {
                    let newCell = new N2MatrixCell(srcIdx, tgtIdx, diagNode, tgt, this.model);
                    this._addCell(srcIdx, tgtIdx, newCell);
                }
                // Make sure tgt isn't descendant of zoomedElement, otherwise it's
                // visible at least as a collapsed node
                else if (tgt.isConnectable() && !this.layout.zoomedElement.hasNode(tgt)) {
                    newDiagCell.addOffScreenConn(diagNode, tgt);
                }
            }

            // Check for missing source part of connections
            for (let src of diagNode.sourceParentSet) {

                if (indexFor(this.diagNodes, src) == -1) {
                    // Make sure src isn't descendant of zoomedElement, otherwise it's
                    // visiable at least as a collapsed node
                    if (src.isConnectable() && !this.layout.zoomedElement.hasNode(src)) {
                        newDiagCell.addOffScreenConn(src, diagNode);
                    }
                }
            }

            // Solver nodes
            if (diagNode.isInput()) {
                for (let j = srcIdx + 1; j < this.diagNodes.length; ++j) {
                    let tgtObj = this.diagNodes[j];
                    if (diagNode.parentComponent !== tgtObj.parentComponent) break;

                    if (tgtObj.isOutput()) {
                        let tgtIdx = j;
                        let newCell = new N2MatrixCell(srcIdx, tgtIdx, diagNode, tgtObj, this.model);
                        this._addCell(srcIdx, tgtIdx, newCell);
                    }
                }
            }
        }

    }

    /**
     * Determine the size of the boxes that will border the variables of each component.
     */
    _setupComponentBoxesAndGridLines() {
        let currentBox = {
            "startI": 0,
            "stopI": 0
        };

        this.boxInfo = [currentBox];

        // Find which component box each of the variables belong in,
        // while finding the bounds of that box. Top and bottom
        // rows recorded for each node in this.boxInfo[].
        for (let ri = 1; ri < this.diagNodes.length; ++ri) {
            let curNode = this.diagNodes[ri];
            let startINode = this.diagNodes[currentBox.startI];
            if (startINode.parentComponent &&
                curNode.parentComponent &&
                startINode.parentComponent === curNode.parentComponent) {
                ++currentBox.stopI;
            }
            else {
                currentBox = {
                    "startI": ri,
                    "stopI": ri
                };
            }
            this.boxInfo.push(currentBox);
        }

        // Step through this.boxInfo[] and record one set of dimensions
        // for each box in this.componentBoxInfo[].
        this.componentBoxInfo = [];
        for (let i = 0; i < this.boxInfo.length; ++i) {
            let box = this.boxInfo[i];
            if (box.startI == box.stopI) continue;
            let curNode = this.diagNodes[box.startI];
            if (!curNode.parentComponent) {
                throw "Parent component not found in box.";
            }
            box.obj = curNode.parentComponent;
            i = box.stopI;
            this.componentBoxInfo.push(box);
        }

        //do this so you save old index for the exit()
        this.gridLines = [];
        if (!this.tooMuchDetail()) {
            for (let i = 0; i < this.diagNodes.length; ++i) {
                let obj = this.diagNodes[i];
                let gl = {
                    "i": i,
                    "obj": obj
                };
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
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        let selection = this.n2Groups.elements.selectAll('.n2cell')
            .data(self.visibleCells, d => d.id);

        // Use D3 to join N2MatrixCells to SVG groups, and render shapes in them.
        let gEnter = selection.enter().append('g')
            .attr('class', 'n2cell')
            .attr('transform', function (d) {
                if (self.lastClickWasLeft) {
                    let tranStr = 'translate(' +
                        (self.prevCellDims.size.width *
                            (d.col - enterIndex) +
                            self.prevCellDims.bottomRight.x) + ',' +
                        (self.prevCellDims.size.height *
                            (d.row - enterIndex) +
                            self.prevCellDims.bottomRight.y) + ')';

                    return tranStr;
                }

                let roc = (d.obj && self.findRootOfChangeFunction) ?
                    self.findRootOfChangeFunction(d.obj) : null;

                if (roc) {
                    let prevIdx = roc.prevRootIndex -
                        self.layout.zoomedElement.prevRootIndex;
                    let tranStr = 'translate(' + (self.prevCellDims.size.width * prevIdx +
                        self.prevCellDims.bottomRight.x) + ',' +
                        (self.prevCellDims.size.height * prevIdx +
                            self.prevCellDims.bottomRight.y) + ')';

                    return tranStr;
                }
                throw ('Enter transform not found');
            })
            .each(function (d) {
                // "this" refers to the element here, so leave it alone:
                d.renderer.renderPrevious(this)
                    .on('mouseover', d.mouseover())
                    .on('mousemove', d.mousemove())
                    .on('mouseleave', n2MouseFuncs.out)
                    .on('click', n2MouseFuncs.click);
            });

        gEnter.merge(selection)
            .transition(sharedTransition)
            .attr('transform', function (d) {
                let tranStr = 'translate(' + (self.cellDims.size.width * d.col +
                    self.cellDims.bottomRight.x) + ',' +
                    (self.cellDims.size.height * d.row +
                        self.cellDims.bottomRight.y) + ')';

                return (tranStr);
            })
            // "this" refers to the element here, so leave it alone:
            .each(function (d) {
                d.renderer.updateCurrent(this)
            });

        selection.exit()
            .transition(sharedTransition)
            .attr('transform', function (d) {
                if (self.lastClickWasLeft) {
                    let tranStr = 'translate(' + (self.cellDims.size.width *
                        (d.col - exitIndex) + self.cellDims.bottomRight.x) + ',' +
                        (self.cellDims.size.height * (d.row - exitIndex) +
                            self.cellDims.bottomRight.y) + ')';
                    return tranStr;
                }

                let roc = (d.obj && self.findRootOfChangeFunction) ?
                    self.findRootOfChangeFunction(d.obj) : null;

                if (roc) {
                    let index = roc.rootIndex - self.layout.zoomedElement.rootIndex;
                    return 'translate(' + (self.cellDims.size.width * index +
                        self.cellDims.bottomRight.x) + ',' +
                        (self.cellDims.size.height * index + self.cellDims.bottomRight.y) + ')';
                }
                throw ('Exit transform not found');
            })
            // "this" refers to the element here, so leave it alone:
            .each(function (d) {
                d.renderer.updateCurrent(this)
            })
            .remove();
    }

    /** Draw a line above every row in the matrix. */
    _drawHorizontalLines() {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        let selection = self.n2Groups.gridlines.selectAll('.horiz_line')
            .data(self.gridLines, function (d) {
                return d.obj.id;
            });

        let gEnter = selection.enter().append('g')
            .attr('class', 'horiz_line')
            .attr('transform', function (d) {
                if (self.lastClickWasLeft) return 'translate(0,' +
                    (self.prevCellDims.size.height * (d.i - enterIndex)) + ')';
                let roc = (self.findRootOfChangeFunction) ?
                    self.findRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let index0 = roc.prevRootIndex - self.layout.zoomedElement.prevRootIndex;
                    return 'translate(0,' + (self.prevCellDims.size.height * index0) + ')';
                }
                throw ('enter transform not found');
            });
        gEnter.append('line')
            .attr('x2', self.layout.size.n2matrix.width);

        let gUpdate = gEnter.merge(selection).transition(sharedTransition)
            .attr('transform', function (d) {
                return 'translate(0,' + (self.cellDims.size.height * d.i) + ')';
            });
        gUpdate.select('line')
            .attr('x2', self.layout.size.n2matrix.width);

        selection.exit().transition(sharedTransition)
            .attr('transform', function (d) {
                if (self.lastClickWasLeft) return 'translate(0,' +
                    (self.cellDims.size.height * (d.i - exitIndex)) + ')';
                let roc = (self.findRootOfChangeFunction) ?
                    self.findRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let index = roc.rootIndex - self.layout.zoomedElement.rootIndex;
                    return 'translate(0,' + (self.cellDims.size.height * index) + ')';
                }
                throw ('exit transform not found');
            })
            .remove();
    }

    /** Draw a vertical line for every column in the matrix. */
    _drawVerticalLines() {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        let selection = self.n2Groups.gridlines.selectAll(".vert_line")
            .data(self.gridLines, function (d) {
                return d.obj.id;
            });
        let gEnter = selection.enter().append("g")
            .attr("class", "vert_line")
            .attr("transform", function (d) {
                if (self.lastClickWasLeft) return "translate(" +
                    (self.prevCellDims.size.width * (d.i - enterIndex)) + ")rotate(-90)";
                let roc = (self.findRootOfChangeFunction) ? self.findRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let i0 = roc.prevRootIndex - self.layout.zoomedElement.prevRootIndex;
                    return "translate(" + (self.prevCellDims.size.width * i0) + ")rotate(-90)";
                }
                throw ("enter transform not found");
            });
        gEnter.append("line")
            .attr("x1", -self.layout.size.n2matrix.height);

        let gUpdate = gEnter.merge(selection).transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + (self.cellDims.size.width * d.i) + ")rotate(-90)";
            });
        gUpdate.select("line")
            .attr("x1", -self.layout.size.n2matrix.height);

        selection.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                if (self.lastClickWasLeft) return "translate(" +
                    (self.cellDims.size.width * (d.i - exitIndex)) + ")rotate(-90)";
                let roc = (self.findRootOfChangeFunction) ? self.findRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let i = roc.rootIndex - self.layout.zoomedElement.rootIndex;
                    return "translate(" + (self.cellDims.size.width * i) + ")rotate(-90)";
                }
                throw ("exit transform not found");
            })
            .remove();
    }

    /** Draw boxes around the cells associated with each component. */
    _drawComponentBoxes() {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        let selection = self.n2Groups.componentBoxes.selectAll(".component_box")
            .data(self.componentBoxInfo, function (d) {
                return d.obj.id;
            });
        let gEnter = selection.enter().append("g")
            .attr("class", "component_box")
            .attr("transform", function (d) {
                if (self.lastClickWasLeft) return "translate(" +
                    (self.prevCellDims.size.width * (d.startI - enterIndex)) + "," +
                    (self.prevCellDims.size.height * (d.startI - enterIndex)) + ")";
                let roc = (d.obj && self.findRootOfChangeFunction) ? self.findRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let index0 = roc.prevRootIndex - self.layout.zoomedElement.prevRootIndex;
                    return "translate(" + (self.prevCellDims.size.width * index0) + "," +
                        (self.prevCellDims.size.height * index0) + ")";
                }
                throw ("enter transform not found");
            })

        gEnter.append("rect")
            .attr("width", function (d) {
                if (self.lastClickWasLeft) return self.prevCellDims.size.width * (1 + d.stopI - d.startI);
                return self.prevCellDims.size.width;
            })
            .attr("height", function (d) {
                if (self.lastClickWasLeft) return self.prevCellDims.size.height * (1 + d.stopI - d.startI);
                return self.prevCellDims.size.height;
            });

        let gUpdate = gEnter.merge(selection).transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + (self.cellDims.size.width * d.startI) + "," +
                    (self.cellDims.size.height * d.startI) + ")";
            });

        gUpdate.select("rect")
            .attr("width", function (d) {
                return self.cellDims.size.width * (1 + d.stopI - d.startI);
            })
            .attr("height", function (d) {
                return self.cellDims.size.height * (1 + d.stopI - d.startI);
            });

        let nodeExit = selection.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                if (self.lastClickWasLeft) return "translate(" +
                    (self.cellDims.size.width * (d.startI - exitIndex)) + "," +
                    (self.cellDims.size.height * (d.startI - exitIndex)) + ")";
                let roc = (d.obj && self.findRootOfChangeFunction) ?
                    self.findRootOfChangeFunction(d.obj) : null;
                if (roc) {
                    let index = roc.rootIndex - self.layout.zoomedElement.rootIndex;
                    return "translate(" + (self.cellDims.size.width * index) + "," +
                        (self.cellDims.size.height * index) + ")";
                }
                throw ("exit transform not found");
            })
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                if (self.lastClickWasLeft) return self.cellDims.size.width * (1 + d.stopI - d.startI);
                return self.cellDims.size.width;
            })
            .attr("height", function (d) {
                if (self.lastClickWasLeft) return self.cellDims.size.height * (1 + d.stopI - d.startI);
                return self.cellDims.size.height;
            });
    }

    /** Add all the visible elements to the matrix. */
    draw() {
        startTimer('N2Matrix.draw');

        let size = this.layout.size;
        d3.select("#n2MatrixClip > rect")
            .transition(sharedTransition)
            .attr('width', size.n2matrix.width + size.svgMargin * 2)
            .attr('height', size.n2matrix.height + size.svgMargin * 2);

        this._drawCells();

        // Draw gridlines:
        if (!this.tooMuchDetail()) {
            debugInfo("Drawing gridlines.")
            this._drawHorizontalLines();
            this._drawVerticalLines();
        }
        else {
            debugInfo("Erasing gridlines.")
            this.n2Groups.gridlines.selectAll('.horiz_line').remove();
            this.n2Groups.gridlines.selectAll(".vert_line").remove();
        }
        this._drawComponentBoxes();

        stopTimer('N2Matrix.draw');

    }

    /**
     * Iterate through all the offscreen connection sets of the
     * hovered cell and draw an arrow/add a tooltip for each.
     */
    _drawOffscreenArrows(cell) {
        if (!cell.offScreen.total) return;

        for (let side in cell.offScreen) {
            for (let dir in cell.offScreen[side]) {
                for (let offscreenNode of cell.offScreen[side][dir]) {
                    this.arrowMgr.addOffGridArrow(cell.id, side, dir, {
                        'cell': {
                            'col': cell.row,
                            'row': cell.row,
                            'srcId': cell.srcObj.id,
                            'tgtId': cell.tgtObj.id
                        },
                        'matrixSize': this.diagNodes.length,
                        'label': offscreenNode.absPathName,
                        'offscreenId': offscreenNode.id
                    });
                }
            }
        }
    }

    /**
     * For a cell that's on the diagonal, look for and draw connection arrows.
     * @param {N2MatrixCell} cell The on-diagonal cell to draw arrows for.
     * @returns {Array} The highlights that can optionally be performed.
     */
    drawOnDiagonalArrows(cell) {
        // Loop over all elements in the matrix looking for other cells in the same column
        this._drawOffscreenArrows(cell);
        let highlights = [{ 'cell': cell, 'varType': 'self', 'direction': 'self' }];

        for (let col = 0; col < this.layout.visibleNodes.length; ++col) {
            if (this.exists(cell.row, col)) {
                if (col != cell.row) {
                    this.arrowMgr.addFullArrow(cell.id, {
                        'end': {
                            'col': col,
                            'row': col,
                            'id': this.grid[col][col].srcObj.id
                        },
                        'start': {
                            'col': cell.row,
                            'row': cell.row,
                            'id': cell.tgtObj.id
                        },
                        'color': N2Style.color.outputArrow,
                    });

                    highlights.push({
                        'cell': this.cell(cell.row, col),
                        'varType': 'target', 'direction': 'output'
                    });
                }
            }

            // Now swap row and col
            if (this.exists(col, cell.row)) {
                if (col != cell.row) {
                    this.arrowMgr.addFullArrow(cell.id, {
                        'start': {
                            'col': col,
                            'row': col,
                            'id': this.grid[col][col].srcObj.id
                        },
                        'end': {
                            'col': cell.row,
                            'row': cell.row,
                            'id': cell.tgtObj.id
                        },
                        'color': N2Style.color.inputArrow,
                    });

                    highlights.push({
                        'cell': this.cell(col, cell.row),
                        'varType': 'source', 'direction': 'input'
                    });
                }
            }
        }

        return highlights;
    }

    /**
     * When the mouse goes over a cell that's on the diagonal, look for and
     * draw connection arrows, and highlight variable names.
     * @param {N2MatrixCell} cell The cell the event occured on.
     */
    mouseOverOnDiagonal(cell) {
        // Don't do anything during transition:
        if (d3.active(cell)) return;
        const highlights = this.drawOnDiagonalArrows(cell);
        for (const h of highlights) h.cell.highlight(h.varType, h.direction);
    }

    drawArrowsInputView(cell, startIndex, endIndex) {
        let boxStart = this.boxInfo[startIndex];
        let boxEnd = this.boxInfo[endIndex];

        // Draw multiple horizontal lines, but no more than one vertical line
        // for box-to-box connections
        let arrows = [];
        for (let startsI = boxStart.startI; startsI <= boxStart.stopI; ++startsI) {
            for (let endsI = boxEnd.startI; endsI <= boxEnd.stopI; ++endsI) {
                if (this.exists(startsI, endsI)) {
                    arrows.push({
                        'start': startsI,
                        'end': endsI
                    });
                }
                /*
                else {
                    throw ("Doesn't exist in matrix: " + startsI + ', ' + endsI);
                } */
            }
        }

        for (let arrow of arrows) {
            this.arrowMgr.addFullArrow(cell.id, {
                'start': {
                    'col': arrow.start,
                    'row': arrow.start,
                    'id': this.grid[arrow.start][arrow.start].srcObj.id
                },
                'end': {
                    'col': arrow.end,
                    'row': arrow.end,
                    'id': this.grid[arrow.end][arrow.end].tgtObj.id
                },
                'color': (startIndex < endIndex) ?
                    N2Style.color.outputArrow : N2Style.color.inputArrow,
            });
        }
    }

    /**
     * Look for and draw cycle arrows of the specified cell.
     * @param {N2MatrixCell} cell The off-diagonal cell to draw arrows for.
     */
    drawOffDiagonalArrows(cell) {
        let src = this.diagNodes[cell.row];
        let tgt = this.diagNodes[cell.col];

        this.arrowMgr.addFullArrow(cell.id, {
            'start': {
                'col': cell.row,
                'row': cell.row,
                'id': cell.srcObj.id
            },
            'end': {
                'col': cell.col,
                'row': cell.col,
                'id': cell.tgtObj.id
            },
            'color': N2Style.color.inputArrow,
        });

        if (cell.row > cell.col) {
            let targetsWithCycleArrows = tgt.getNodesWithCycleArrows();

            for (let twca of targetsWithCycleArrows) {
                for (let ai of twca.cycleArrows) {
                    if (src.hasNode(ai.src)) {
                        for (let arrow of ai.arrows) {
                            let firstBeginIndex = -1,
                                firstEndIndex = -1;

                            // find first begin index
                            for (let mi in this.diagNodes) {
                                let rtNode = this.diagNodes[mi];
                                if (rtNode.hasNode(arrow.begin)) {
                                    firstBeginIndex = mi;
                                    break;
                                }
                            }
                            if (firstBeginIndex == -1) {
                                throw ("Error: first begin index not found");
                            }

                            // find first end index
                            for (let mi in this.diagNodes) {
                                let rtNode = this.diagNodes[mi];
                                if (rtNode.hasNode(arrow.end)) {
                                    firstEndIndex = mi;
                                    break;
                                }
                            }
                            if (firstEndIndex == -1) {
                                throw ("Error: first end index not found");
                            }

                            if (firstBeginIndex != firstEndIndex) {
                                this.drawArrowsInputView(cell, firstBeginIndex, firstEndIndex);
                            }
                        }
                    }
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
        // Don't do anything during transition:
        if (d3.active(cell)) return;

        this.drawOffDiagonalArrows(cell);

        cell.highlight('source', 'input');
        cell.highlight('target', 'output');
    }

    /**
     * Determine if a cell is on the diagonal or not and draw the appropriate
     * connection arrows.
     * @param {N2MatrixCell} cell The cell to operate on.
     */
    drawConnectionArrows(cell) {
        if (cell.row == cell.col) this.drawOnDiagonalArrows(cell);
        else this.drawOffDiagonalArrows(cell);
    }
}
