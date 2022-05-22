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
     * Draw the cycle arrows in the lower-left corner of the matrix.
     * @param {OmMatrixCell} cell The focused cell.
     * @param {Number} startIndex The index of the first diagonal node.
     * @param {Number} endIndex The index of the last diagonal node.
     */
    _drawArrowsInputView(cell, startIndex, endIndex) {
        const boxStart = this._boxInfo[startIndex],
            boxEnd = this._boxInfo[endIndex];

        // Draw multiple horizontal lines, but no more than one vertical line
        // for box-to-box connections
        const arrows = [];
        for (let startsI = boxStart.startI; startsI <= boxStart.stopI; ++startsI) {
            for (let endsI = boxEnd.startI; endsI <= boxEnd.stopI; ++endsI) {
                if (this.exists(startsI, endsI)) {
                    arrows.push({
                        'start': startsI,
                        'end': endsI
                    });
                }
            }
        }

        for (const arrow of arrows) {
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
                    OmStyle.color.outputArrow : OmStyle.color.inputArrow,
            });
        }
    }

    /**
     * Look for and draw cycle arrows of the specified cell.
     * @param {OmMatrixCell} cell The off-diagonal cell to draw arrows for.
     */
    drawOffDiagonalArrows(cell) {
        super.drawOffDiagonalArrows(cell);

        /* Cycle arrows are only drawn in the bottom triangle of the diagram */
        if (cell.row > cell.col) {
            const src = this.diagNodes[cell.row],
                tgt = this.diagNodes[cell.col];

            // Get an array of all the parents and children of the target with cycle arrows
            const relativesWithCycleArrows = tgt.getNodesWithCycleArrows();

            for (const relative of relativesWithCycleArrows) {
                for (const ai of relative.cycleArrows) {
                    if (src.hasNode(ai.src)) {
                        for (const arrow of ai.arrows) {
                            const firstBeginIndex =
                                this.diagNodes.findIndex(diagNode => diagNode.hasNode(arrow.begin));
                            if (firstBeginIndex == -1)
                                throw ("OmMatrix.drawOffDiagonalArrows() error: first begin index not found");

                            const firstEndIndex =
                                this.diagNodes.findIndex(diagNode => diagNode.hasNode(arrow.end));
                            if (firstEndIndex == -1)
                                throw ("OmMatrix.drawOffDiagonalArrows() error: first end index not found");

                            if (firstBeginIndex != firstEndIndex) {
                                this._drawArrowsInputView(cell, firstBeginIndex, firstEndIndex);
                            }
                        }
                    }
                }
            }
        }
    }
}
