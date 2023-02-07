// <<hpp_insert src/OmMatrixCell.js>>
// <<hpp_insert gen/Matrix.js>>

/**
 * Use the model tree to build the matrix of variables and connections, display, and
 * perform operations with it.
 * @typedef OmMatrix
 * @property {OmTreeNodes[]} nodes Reference to nodes that will be drawn.
 * @property {OmModelData} model Reference to the pre-processed model.
 * @property {OmLayout} layout Reference to object managing columns widths and such.
 * @property {Object} diagGroups References to <g> SVG elements created by Diagram.
 * @property {Number} levelOfDetailThreshold Don't draw elements below this size in pixels.
 * @property {Object} nodeSize Width and height of each node in the matrix.
 * @property {Object} prevNodeSize Width and height of each node in the previous matrix.
 * @property {Object[][]} grid Object keys corresponding to rows and columns.
 * @property {OmMatrixCell[]} visibleCells One-dimensional array of all cells, for D3 processing.
 */
class OmMatrix extends Matrix {
    /**
     * Render the matrix of visible elements in the model.
     * @param {OmModelData} model The pre-processed model data.
     * @param {OmLayout} layout Pre-computed layout of the diagram.
     * @param {Object} diagGroups References to <g> SVG elements created by Diagram.
     * @param {ArrowManager} arrowMgr Object to create and manage conn. arrows.
     * @param {Boolean} lastClickWasLeft
     * @param {function} findRootOfChangeFunction
     */
    constructor(model, layout, diagGroups, arrowMgr, lastClickWasLeft, findRootOfChangeFunction,
        prevNodeSize = { 'width': 0, 'height': 0 }) {
        super(model, layout, diagGroups, arrowMgr, lastClickWasLeft, findRootOfChangeFunction, prevNodeSize);

        this._nodeIdxCache = {};
    }

    /**
     * Add a cell to the grid Object and visibleCells array, with special
     * handling for declared partials.
     * @param {number} row Index of the row in the grid to place the cell.
     * @param {number} col Index of the column in the grid to place the cell.
     * @param {OmMatrixCell} newCell Cell created in _buildGrid().
     */
    _addCell(row, col, newCell) {
        if (this.model.useDeclarePartialsList &&
            newCell.symbolType.potentialDeclaredPartial &&
            !newCell.symbolType.declaredPartial) {

            return;
        }

        super._addCell(row, col, newCell);
    }

    /**
     * For cells that are part of cycles, determine if there are parts
     * of the cycle that are offscreen. If so, record them in the cell.
     * @param {OmMatrixCell} cell The cell to check.
     */
    _findUnseenCycleSources(cell) {
        const node = cell.tgtObj;
        if (!(node instanceof OmTreeNode)) return;
        
        const targetsWithCycleArrows = node.getNodesWithCycleArrows();

        const offscreenInit = cell.offScreen.total;

        for (const twca of targetsWithCycleArrows) {
            for (const ai of twca.cycleArrows) {
                let found = false;

                // Check visible nodes on the diagonal.
                for (const diagNode of this.diagNodes) {
                    const commonParent = diagNode.nearestCommonParent(ai.src);
                    if (diagNode.hasNode(ai.src, commonParent)) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    for (const tgt of ai.src.targetParentSet) {
                        if (tgt.absPathName == node.absPathName) {
                            debugInfo("Adding offscreen connection from _findUnseenCycleSources().")
                            debugInfo(`Offscreen cycle conn from ${ai.src.absPathName}(${ai.src.type}) to ${node.absPathName}(${node.type}).`)
                            cell.addOffScreenConn(ai.src, node);
                        }
                    }
                }
            }
        }
    }

    /**
     * Generate a new OmMatrixCell object. Overrides superclass definition.
     * @param {Number} row Vertical coordinate of the cell in the matrix.
     * @param {Number} col Horizontal coordinate of the cell in the matrix.
     * @param {OmTreeNode} srcObj The node in the model tree this node is associated with.
     * @param {OmTreeNode} tgtObj The model tree node that this outputs to.
     * @param {ModelData} model Reference to the model to get some info from it.
     * @returns {OmMatrixCell} Newly created cell.
     */
    _createCell(row, col, srcObj, tgtObj, model) {
        return new OmMatrixCell(row, col, srcObj, tgtObj, model);
    }

    /**
     * Handle solver processing and cycle arrows. Overrides superclass definition.
     * @param {Number} srcIdx Index of the diagonal node being processed.
     * @param {MatrixCell} newDiagCell A new cell being placed.
     */
    _customProcessing(srcIdx, newDiagCell) {
        const diagNode = this.diagNodes[srcIdx]

        // Solver nodes
        let solverNodes = null;
        if (diagNode.isInput()) { solverNodes = [diagNode]; }
        else if (diagNode.isInputFilter() && diagNode.count > 0) {
            solverNodes = diagNode.children;
        }
        else { solverNodes = []; }

        for (const solverNode of solverNodes) {
            for (let j = srcIdx + 1; j < this.diagNodes.length; ++j) {
                const tgtObj = this.diagNodes[j];
                if (solverNode.parentComponent !== tgtObj.parentComponent) break;

                if (tgtObj.isOutput() || tgtObj.isOutputFilter()) {
                    const tgtIdx = j;
                    const newCell = this._createCell(srcIdx, tgtIdx, solverNode, tgtObj, this.model);
                    this._addCell(srcIdx, tgtIdx, newCell);
                }
            }
        }

        this._findUnseenCycleSources(newDiagCell);
    }

    /**
     * Determine whether a connection has already been drawn. If not,
     * record that it has now.
     * @param {Number} startIndex The index of the first diagonal node.
     * @param {Number} endIndex The index of the last diagonal node.
     * @returns {Boolean} True if the connection is already drawn, otherwise false.
     */
    _checkDrawnList(startIndex, endIndex) {
        if (this._drawnList[startIndex] && this._drawnList[startIndex][endIndex]) return true;
        if (this._drawnList[endIndex] && this._drawnList[endIndex][startIndex]) return true;

        if (!this._drawnList[startIndex]) this._drawnList[startIndex] = {};
        this._drawnList[startIndex][endIndex] = true;
        return false;
    }

    /**
     * Determine which cycle arrows to draw in the lower-left corner of the matrix.
     * @param {Number} startIndex The index of the first diagonal node.
     * @param {Number} endIndex The index of the last diagonal node.
     */
    _addArrowsInputView(startIndex, endIndex) {
        if (this._checkDrawnList(startIndex, endIndex)) return;

        const boxInfo = this._boxInfo()
        const boxStart = boxInfo[startIndex],
            boxEnd = boxInfo[endIndex];

        // Draw multiple horizontal lines, but no more than one vertical line
        // for box-to-box connections
        for (let startsI = boxStart.startI; startsI <= boxStart.stopI; ++startsI) {
            for (let endsI = boxEnd.startI; endsI <= boxEnd.stopI; ++endsI) {
                if (startsI != endsI && this.exists(startsI, endsI)) {
                    if (!this._arrows[startsI]) this._arrows[startsI] = {};
                    this._arrows[startsI][endsI] = true;
                }
            }
        }
    }

    /**
     * Draw the accumulated set of arrows in the matrix.
     * @param {OmMatrixCell} cell Reference to the cell where the mouseover was triggered.
     */
    _drawArrowsInputView(cell) {
        for (const start in this._arrows) {
            for (const end in this._arrows[start]) {
                this.arrowMgr.addFullArrow(cell.id, {
                    'start': {
                        'col': start,
                        'row': start,
                        'id': this.grid[start][start].srcObj.id
                    },
                    'end': {
                        'col': end,
                        'row': end,
                        'id': this.grid[end][end].tgtObj.id
                    },
                    'color': (start < end) ?
                        OmStyle.color.outputArrow : OmStyle.color.inputArrow,
                });
            }
        }
    }

    /**
     * Find the index of the first node in this.diagNodes that "has" the supplied node.
     * For some reason Array.findIndex() did not work correctly for this. Found nodes
     * are cached with their index so the search doesn't have to be performed more
     * than once per node.
     * @param {OmTreeNode} node Reference to the node to search for.
     * @returns {Number} The index of the node if found, otherwise -1.
     */
    _findDiagNodeIndex(node) {
        if (this._nodeIdxCache[node.path]) {
            return this._nodeIdxCache[node.path];
        }

        for (const idx in this.diagNodes) {
            if (this.diagNodes[idx].hasNode(node)) {
                this._nodeIdxCache[node.path] = idx;
                return idx;
            }
        }

        return -1;
    }

    /**
     * Look for and draw cycle arrows of the specified cell.
     * @param {OmMatrixCell} cell The off-diagonal cell to draw arrows for.
     */
    drawOffDiagonalArrows(cell) {
        super.drawOffDiagonalArrows(cell);

        /* Cycle arrows are only drawn for cells in the bottom triangle of the diagram */
        if (cell.row > cell.col) {
            const src = this.diagNodes[cell.row],
                tgt = this.diagNodes[cell.col];

            // Accumulate what's been discovered/drawn in these to prevent duplicates
            this._drawnList = {};
            this._arrows = {};

            // Get an array of all the parents and children of the target with cycle arrows
            const relativesWithCycleArrows = tgt.getNodesWithCycleArrows();

            for (const relative of relativesWithCycleArrows) {
                for (const ai of relative.cycleArrows) {
                    if (src.hasNode(ai.src)) {
                        for (const arrow of ai.arrows) {
                            const firstBeginIndex = this._findDiagNodeIndex(arrow.begin);
                            if (firstBeginIndex == -1)
                                throw ("OmMatrix.drawOffDiagonalArrows() error: first begin index not found");

                            const firstEndIndex = this._findDiagNodeIndex(arrow.end);
                            if (firstEndIndex == -1)
                                throw ("OmMatrix.drawOffDiagonalArrows() error: first end index not found");

                            if (firstBeginIndex != firstEndIndex) {
                                this._addArrowsInputView(firstBeginIndex, firstEndIndex);
                            }
                        }
                    }
                }
            }

            this._drawArrowsInputView(cell)
        }
    }
}
