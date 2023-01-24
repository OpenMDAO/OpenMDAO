// <<hpp_insert gen/MatrixCell.js>>

/**
 * Use the model tree to build the matrix of variables and connections, display, and
 * perform operations with it.
 * @typedef Matrix
 * @property {TreeNodes[]} nodes Reference to nodes that will be drawn.
 * @property {ModelData} model Reference to the pre-processed model.
 * @property {Layout} layout Reference to object managing columns widths and such.
 * @property {Object} diagGroups References to <g> SVG elements created by Diagram.
 * @property {number} levelOfDetailThreshold Don't draw elements below this size in pixels.
 * @property {Object} nodeSize Width and height of each node in the matrix.
 * @property {Object} prevNodeSize Width and height of each node in the previous matrix.
 * @property {Object[][]} grid Object keys corresponding to rows and columns.
 * @property {MatrixCell[]} visibleCells One-dimensional array of all cells, for D3 processing.
 */
class Matrix {
    /**
     * Render the matrix of visible elements in the model.
     * @param {ModelData} model The pre-processed model data.
     * @param {Layout} layout Pre-computed layout of the diagram.
     * @param {Object} diagGroups References to <g> SVG elements created by Diagram.
     * @param {ArrowManager} arrowMgr Object to create and manage conn. arrows.
     * @param {Boolean} lastClickWasLeft
     * @param {function} findRootOfChangeFunction
     */
    constructor(model, layout, diagGroups, arrowMgr,
        lastClickWasLeft,
        findRootOfChangeFunction,
        prevNodeSize = {
            'width': 0,
            'height': 0
        }) {

        this.model = model;
        this.layout = layout;
        this.diagNodes = layout.visibleNodes;
        this.diagGroups = diagGroups;
        this.arrowMgr = arrowMgr;
        this.lastClickWasLeft = lastClickWasLeft;
        this.findRootOfChangeFunction = findRootOfChangeFunction;

        this.prevNodeSize = prevNodeSize;
        this.nodeSize = {
            'width': layout.size.matrix.width / this.diagNodes.length,
            'height': layout.size.matrix.height / this.diagNodes.length,
        }
        this.arrowMgr.setNodeSize(this.nodeSize);

        const markerSize = Math.max(2, this.nodeSize.width * .04, this.nodeSize.height * .04);
        d3.select("#arrow").attr("markerWidth", markerSize).attr("markerHeight", markerSize);
        d3.select("#offgridArrow").attr("markerWidth", markerSize * 2).attr("markerHeight", markerSize);

        this._init();
        
        this.updateLevelOfDetailThreshold(layout.size.matrix.height);

        this._buildGrid();
        this._setupVariableBoxesAndGridLines();
    }

    _init() {
        CellRenderer.updateDims(this.nodeSize.width, this.nodeSize.height);
    }

    get cellDims() { return CellRenderer.dims; }
    get prevCellDims() { return CellRenderer.prevDims; }

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
     * @param {MatrixCell} cell The cell to test.
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
     * @returns {MatrixCell} The node if it exists, undefined otherwise.
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
     * Given the id of the cell, try to find it in the visible matrix.
     * @param {String} cellId Unique string associated with the cell.
     * @returns {MatrixCell} Cell reference if found, otherwise undefined.
     */
    findCellById(cellId) {
        for (const cell of this.visibleCells) {
            if (cell.id == cellId) return cell;
        }

        return undefined;
    }

    /**
     * Given the node ID, determine if one of the cells in the matrix
     * represents it or contains it.
     * @param {Number} nodeId The id of the TreeNode to search for
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
            if (cell.id == MatrixCell.makeId(nodeId)) {
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
        const tooMuch = (this.diagNodes.length >= this.levelOfDetailThreshold);

        if (tooMuch) debugInfo("Too much detail.")

        return tooMuch;
    }

    /**
     * Add a cell to the grid Object and visibleCells array.
     * @param {number} row Index of the row in the grid to place the cell.
     * @param {number} col Index of the column in the grid to place the cell.
     * @param {MatrixCell} newCell Cell created in _buildGrid().
     */
    _addCell(row, col, newCell) {
        this.grid[row][col] = newCell;
        this.visibleCells.push(newCell);
    }

    /**
     * Generate a new MatrixCell object. Can be overridden by subclass.
     * @param {Number} row Vertical coordinate of the cell in the matrix.
     * @param {Number} col Horizontal coordinate of the cell in the matrix.
     * @param {TreeNode} srcObj The node in the model tree this node is associated with.
     * @param {TreeNode} tgtObj The model tree node that this outputs to.
     * @param {ModelData} model Reference to the model to get some info from it.
     * @returns {MatrixCell} Newly created cell.
     */
    _createCell(row, col, srcObj, tgtObj, model) {
        return new MatrixCell(row, col, srcObj, tgtObj, model);
    }

    /**
     * Stub to be overridden by subclasses that need to add more info.
     * @param {Number} srcIdx Index of the diagonal node being processed.
     * @param {MatrixCell} newDiagCell A new cell being placed.
     */
    _customProcessing(srcIdx, newDiagCell) { }

    /**
     * Set up MatrixCell arrays resembling a two-dimensional grid as the matrix, but not
     * an actual two dimensional array because most of it would be unused. Each node
     * appearing on the diagonal is first added to the matrix. It's then checked for connections
     * to see if a connecting cell needs to be added or if the connection is going offscreen.
     * Additional processing tests collapsed and filtered nodes for connections.
     */
    _buildGrid() {
        this.visibleCells = [];
        this.grid = {};

        if (this.tooMuchDetail()) return;

        for (let srcIdx = 0; srcIdx < this.diagNodes.length; ++srcIdx) {
            const diagNode = this.diagNodes[srcIdx];

            // New row
            if (!this.grid.propExists(srcIdx)) this.grid[srcIdx] = {};

            // On the diagonal
            const newDiagCell = this._createCell(srcIdx, srcIdx, diagNode, diagNode, this.model);
            this._addCell(srcIdx, srcIdx, newDiagCell);

            for (const tgt of diagNode.targetParentSet) {
                const tgtNode = tgt.draw.filtered? tgt.draw.filterParent : tgt;
                const tgtIdx = indexFor(this.diagNodes, tgtNode);

                if (tgtIdx != -1 && srcIdx != tgtIdx) {
                    const newCell = this._createCell(srcIdx, tgtIdx, diagNode, tgtNode, this.model);
                    this._addCell(srcIdx, tgtIdx, newCell);
                }
                // Make sure tgt isn't descendant of zoomedElement, otherwise it's
                // visible at least as a collapsed node
                else if (tgt.isConnectable() && !this.layout.zoomedElement.hasNode(tgt)) {
                    newDiagCell.addOffScreenConn(diagNode, tgt);
                }
            }

            // Check for missing source part of connections
            for (const src of diagNode.sourceParentSet) {
                const srcNode = src.draw.filtered? src.draw.filterParent : src;

                if (indexFor(this.diagNodes, srcNode) == -1) {
                    // Make sure src isn't descendant of zoomedElement, otherwise it's
                    // visiable at least as a collapsed node
                    if (src.isConnectable() && !this.layout.zoomedElement.hasNode(src)) {
                        newDiagCell.addOffScreenConn(srcNode, diagNode);
                    }
                }
            }

            this._customProcessing(srcIdx, newDiagCell);
        }
    }

    /**
     * Determine which "box" a diagonal node is in based on common ancestors.
     * @param {Number} [ancestorLevel=2] The nth ancestor to compare.
     * @returns {Array} Box dimensions of the box containing each diagonal node.
     */
    _boxInfo(ancestorLevel = 2) {
        let currentBox = {
            "startI": 0,
            "stopI": 0
        };

        const boxInfo = [currentBox];

        // Find which variable box each of the variables belong in,
        // while finding the bounds of that box. Top and bottom
        // rows recorded for each node.
        for (let ri = 1; ri < this.diagNodes.length; ++ri) {
            const curNode = this.diagNodes[ri];
            const startINode = this.diagNodes[currentBox.startI];

            if (startINode.boxAncestor(ancestorLevel) === curNode.boxAncestor(ancestorLevel)) {
                ++currentBox.stopI;
            }
            else {
                currentBox = {
                    "startI": ri,
                    "stopI": ri
                };
            }
            boxInfo.push(currentBox);
        }

        return boxInfo;
    }

    /** Determine the size of the boxes that will enclose the variables of each group. */
    _setupVariableBoxesAndGridLines() {
        const boxInfo = this._boxInfo();

        // Step through boxInfo[] and record one set of dimensions
        // for each box in this._variableBoxInfo[].
        this._variableBoxInfo = [];
        for (let i = 0; i < boxInfo.length; ++i) {
            const box = boxInfo[i];
            if (box.startI == box.stopI) continue;

            const curNode = this.diagNodes[box.startI];
            if (!curNode.boxAncestor()) { continue; throw "Ancestor not found in box."; }

            box.obj = curNode.boxAncestor();
            if (box.obj.draw.varBoxDims) {
                box.obj.draw.varBoxDims.preserve().count = 1 + box.stopI - box.startI;
            }

            i = box.stopI;
            this._variableBoxInfo.push(box);
        }

        //do this so you save old index for the exit()
        this._gridLines = [];
        if (!this.tooMuchDetail()) {
            for (let i = 0; i < this.diagNodes.length; ++i) {
                const obj = this.diagNodes[i];
                const gl = { "i": i, "obj": obj };
                this._gridLines.push(gl);
            }
        }
    }

    /**
     * Add SVG groups & contents coupled to the visible nodes in the matrix.
     * Select all <g> elements that have class "n2cell". If any already
     * exist, join to their associated cells in the matrix. If no
     * existing <g> matches a displayable cell, add it to the "enter"
     * selection so the <g> can be created. If a <g> exists but there is
     * no longer a displayable cell for it, put it in the "exit" selection so
     * it can be removed.
     * @param {Dimensions} dims Size of a cell and placement within its group.
     */
    _drawCells(dims) {
        const self = this;

        this.diagGroups.elements.selectAll('g.n2cell')
            .data(this.visibleCells, d => d.id)
            .join(
                enter => self._addNewMatrixCells(enter, dims),
                update => self._updateExistingMatrixCells(update, dims),
                exit => self._removeOldMatrixCells(exit, dims)
            )
    }

    /**
     * Using the visible cells in the matrix grid as data points, create SVG objects to
     * represent each one. Location is calculated from a cell's column and row.
     * @param {Selection} enter The selection to add <g> elements and children to.
     * @param {Dimensions} dims Size of a cell and placement within its group.
     */
    _addNewMatrixCells(enter, dims) {
        const enterSelection = enter.append('g')
            .attr('class', 'n2cell');

        enterSelection
            // Initialize at "previous" location...
            .attr('transform', d => {
                const transX = dims.prev.width * (d.col - enterIndex) + dims.prev.x,
                    transY = dims.prev.height * (d.row - enterIndex) + dims.prev.y;

                return `translate(${transX},${transY})`;
            })
            .transition(sharedTransition)
            // ... then smoothly transition to new location to appear to slide in:
            .attr('transform', d => {
                const transX = dims.width * d.col + dims.x,
                    transY = dims.height * d.row + dims.y;

                return `translate(${transX},${transY})`;
            })
            .each(function(d) {
                // "this" refers to the element here, so leave it alone:
                d.renderer.renderCurrent(this)
                    .on('mouseover', d.getMouseoverFunc())
                    .on('mousemove', d.getMousemoveFunc())
                    .on('mouseleave', n2MouseFuncs.out)
                    .on('click', (e,d) => n2MouseFuncs.click(d))
            });
        
        return enterSelection;
    }

    /**
     * Update the geometry for existing SVG groups with a transition.
     * @param {Selection} update The selected group of existing matrix grid <g> elements.
     * @param {Dimensions} dims Size of the cell and placement within its group.
     */
    _updateExistingMatrixCells(update, dims) {
        update.transition(sharedTransition)
            .attr('transform', d => {
                const transX = dims.width * d.col + dims.x,
                    transY = dims.height * d.row + dims.y;

                return `translate(${transX},${transY})`;
            })
            // "this" refers to the element here, so leave it alone:
            .each(function (d) {
                d.renderer.updateCurrent(this)
            });
    }

    /**
     * Remove SVG groups that no longer have displayable cells associated with them, and
     * transition them away.
     * @param {Selection} exit The selected group of matrix grid <g> elements to remove.
     * @param {Dimensions} dims Size of the cell and placement within its group.
     */
    _removeOldMatrixCells(exit, dims) {
        const exitSelection = exit.transition(sharedTransition)
            .attr('transform', d => {
                const transX = dims.width * (d.col - exitIndex) + dims.x,
                    transY = dims.height * (d.row - exitIndex) + dims.y;

                return `translate(${transX},${transY})`
            })
            .each(function (d) { d.renderer.updateCurrent(this); });

        exitSelection.on('end', function() {d3.select(this).remove(); })

        return exitSelection;
    }

    /**
     * Draw a line above every row in the matrix.
     * @param {Dimensions} dims Sizes for a current and previous matrix cell.
     */
    _drawHorizontalLines(dims) {
        const self = this;

        self.diagGroups.gridlines.selectAll('g.horiz_line')
            .data(self._gridLines, d => d.obj.id)
            .join(
                enter => { 
                    const enterGroups = enter.append('g')
                        .attr('class', 'horiz_line')
                        .attr('transform', d => `translate(0, ${dims.prev.height * (d.i - enterIndex)})`);

                    enterGroups.transition(sharedTransition)
                        .attr('transform', d => `translate(0, ${dims.height * d.i})`)

                    enterGroups.append('line')
                        .attr('x2', self.layout.size.matrix.width)
                },
                update => update.transition(sharedTransition)
                    .attr('transform', d => `translate(0, ${dims.height * d.i})`)
                    .select('line')
                    .attr('x2', self.layout.size.matrix.width),
                exit => exit.transition(sharedTransition)
                    .attr('transform', d => `translate(0, ${dims.height * (d.i - exitIndex)})`)
                    .remove()
            )
    }

    /**
     * Draw a vertical line for every column in the matrix.
     * @param {Dimensions} dims Sizes for a current and previous matrix cell.
     */
     _drawVerticalLines(dims) {
        const self = this;

        self.diagGroups.gridlines.selectAll('g.vert_line')
        .data(self._gridLines, d => d.obj.id)
        .join(
            enter => { 
                const enterGroups = enter.append('g')
                    .attr('class', 'vert_line')
                    .attr('transform', d => `translate(${dims.prev.width * (d.i - enterIndex)}) rotate(-90)`);

                enterGroups.transition(sharedTransition)
                    .attr('transform', d => `translate(${dims.width * d.i}) rotate(-90)`)

                enterGroups.append('line')
                    .attr('x1', -self.layout.size.matrix.height)
            },
            update => update.transition(sharedTransition)
                .attr('transform', d => `translate(${dims.width * d.i}) rotate(-90)`)
                .select('line')
                .attr('x1', -self.layout.size.matrix.height),
            exit => exit.transition(sharedTransition)
                .attr('transform', d => `translate(0, ${dims.width * (d.i - exitIndex)}) rotate(-90)`)
                .remove()
        )

    }

    /** Draw boxes around the cells associated with each variable grouping. */
    _drawVariableBoxes(dims) {
        const self = this; 

        self.diagGroups.variableBoxes.selectAll('g.variable_box')
            .data(self._variableBoxInfo, d => d.obj.id)
            .join(
                enter => {
                    const newGroups = enter.append('g')
                        .attr('class', 'variable_box')
                        .attr('transform', d => {
                            const transX = dims.prev.width * (d.startI - enterIndex),
                                transY = dims.prev.height * (d.startI - enterIndex);
                            return `translate(${transX}, ${transY})`;
                        });

                    newGroups.transition(sharedTransition)
                        .attr('transform', d =>
                            `translate(${dims.width * d.startI}, ${dims.height * d.startI})`);

                    newGroups.append('rect')
                        .attr('width', d => dims.prev.width * (1 + d.stopI - d.startI))
                        .attr('height', d => dims.prev.height * (1 + d.stopI - d.startI))
                        .transition(sharedTransition)
                        .attr('width', d => dims.width * (1 + d.stopI - d.startI))
                        .attr('height', d => dims.height * (1 + d.stopI - d.startI));
                },
                update => {
                    update.transition(sharedTransition)
                        .attr('transform', d =>
                            `translate(${dims.width * d.startI}, ${dims.height * d.startI})`)

                    update.select('rect').transition(sharedTransition)
                        .attr('width', d => dims.width * (1 + d.stopI - d.startI))
                        .attr('height', d => dims.height * (1 + d.stopI - d.startI))
                },
                exit => {
                    exit.transition(sharedTransition)
                        .attr('transform', d => {
                            const transX = dims.width * (d.startI - exitIndex),
                              transY = dims.height * (d.startI - exitIndex);
                            return `translate(${transX}, ${transY})`;
                        })
                        .remove();                 
                }
            )
    }

    _preDraw(dims) {

    }

    /** Add all the visible elements to the matrix. */
    draw() {
        // Dimensions used to calculate cell geometry and gridlines
        const cellDims = new Dimensions(
            { // Current cell geometry:
                width: this.cellDims.size.width,
                height: this.cellDims.size.height,
                x: this.cellDims.bottomRight.x,
                y: this.cellDims.bottomRight.y   
            }, 
            null, 
            { // Previous cell geometry:
                width: this.prevCellDims.size.width,
                height: this.prevCellDims.size.height,
                x: this.prevCellDims.bottomRight.x,
                y: this.prevCellDims.bottomRight.y 
            }
        );

        const size = this.layout.size;

        d3.select("#n2MatrixClip > rect")
            .transition(sharedTransition)
            .attr('width', size.matrix.width + size.svgMargin * 2)
            .attr('height', size.matrix.height + size.svgMargin * 2);

        this._preDraw(cellDims);
        this._drawCells(cellDims);

        if (!this.tooMuchDetail()) {
            debugInfo("Drawing gridlines.")
            this._drawHorizontalLines(cellDims);
            this._drawVerticalLines(cellDims);
        }
        else {
            debugInfo("Erasing gridlines.")
            this.diagGroups.gridlines.selectAll('.horiz_line').remove();
            this.diagGroups.gridlines.selectAll(".vert_line").remove();
        }
        this._drawVariableBoxes(cellDims);
    }

    /**
     * Iterate through all the offscreen connection sets of the
     * hovered cell and draw an arrow/add a tooltip for each.
     */
    _drawOffscreenArrows(cell) {
        if (!cell.offScreen.total) return;

        for (const side in cell.offScreen) {
            for (const dir in cell.offScreen[side]) {
                for (const offscreenNode of cell.offScreen[side][dir]) {
                    this.arrowMgr.addOffGridArrow(cell.id, side, dir, {
                        'cell': {
                            'col': cell.row,
                            'row': cell.row,
                            'srcId': cell.srcObj.id,
                            'tgtId': cell.tgtObj.id
                        },
                        'matrixSize': this.diagNodes.length,
                        'label': offscreenNode.path,
                        'offscreenId': offscreenNode.id
                    });
                }
            }
        }
    }

    /**
     * For a cell that's on the diagonal, look for and draw connection arrows.
     * @param {MatrixCell} cell The on-diagonal cell to draw arrows for.
     * @returns {Array} The highlights that can optionally be performed.
     */
    drawOnDiagonalArrows(cell) {
        // Loop over all elements in the matrix looking for other cells in the same column
        this._drawOffscreenArrows(cell);
        const highlights = [{ 'cell': cell, 'varType': 'self', 'direction': 'self' }];

        this.layout.visibleNodes.forEach( (node, col) => {
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
                        'color': Style.color.outputArrow,
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
                        'color': Style.color.inputArrow,
                    });

                    highlights.push({
                        'cell': this.cell(col, cell.row),
                        'varType': 'source', 'direction': 'input'
                    });
                }
            }
        });

        return highlights;
    }

    /**
     * When the mouse goes over a cell that's on the diagonal, look for and
     * draw connection arrows, and highlight variable names.
     * @param {MatrixCell} cell The cell the event occured on.
     */
    mouseOverOnDiagonal(cell) {
        if (! d3.active(cell)) { // Don't do anything during transition
            const highlights = this.drawOnDiagonalArrows(cell);
            highlights.forEach(h => h.cell.highlight(h.varType, h.direction));
        }
    }

    /**
     * Look for and draw cycle arrows of the specified cell.
     * @param {MatrixCell} cell The off-diagonal cell to draw arrows for.
     */
    drawOffDiagonalArrows(cell) {
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
            'color': Style.color.inputArrow,
        });
    }

    /**
     * When the mouse goes over a cell that's not on the diagonal, look for and
     * draw cycle arrows, and highlight variable names.
     * @param {MatrixCell} cell The cell the event occured on.
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
     * @param {MatrixCell} cell The cell to operate on.
     */
    drawConnectionArrows(cell) {
        if (cell.row == cell.col) this.drawOnDiagonalArrows(cell);
        else this.drawOffDiagonalArrows(cell);
    }
}
