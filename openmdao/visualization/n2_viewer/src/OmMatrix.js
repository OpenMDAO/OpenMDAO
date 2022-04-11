// <<hpp_insert src/OmMatrixCell.js>>
// <<hpp_insert gen/Matrix.js>>

/**
 * Use the model tree to build the matrix of variables and connections, display, and
 * perform operations with it.
 * @typedef OmMatrix
 * @property {OmTreeNodes[]} nodes Reference to nodes that will be drawn.
 * @property {ModelData} model Reference to the pre-processed model.
 * @property {N2Layout} layout Reference to object managing columns widths and such.
 * @property {Object} n2Groups References to <g> SVG elements created by N2Diagram.
 * @property {number} levelOfDetailThreshold Don't draw elements below this size in pixels.
 * @property {Object} nodeSize Width and height of each node in the matrix.
 * @property {Object} prevNodeSize Width and height of each node in the previous matrix.
 * @property {Object[][]} grid Object keys corresponding to rows and columns.
 * @property {Array} visibleCells One-dimensional array of all cells, for D3 processing.
 * @property {Array} boxInfo Variable box dimensions.
 */
class OmMatrix extends Matrix {
    /**
     * Render the matrix of visible elements in the model.
     * @param {OmModelData} model The pre-processed model data.
     * @param {OmLayout} layout Pre-computed layout of the diagram.
     * @param {Object} n2Groups References to <g> SVG elements created by N2Diagram.
     * @param {ArrowManager} arrowMgr Object to create and manage conn. arrows.
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
        super(model, layout, n2Groups, arrowMgr, lastClickWasLeft, findRootOfChangeFunction, prevNodeSize);
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

    _createCell(row, col, srcObj, tgtObj, model) {
        return new OmMatrixCell(row, col, srcObj, tgtObj, model);
    }

    _extraProcessing(srcIdx, newDiagCell) {
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
     * Look for and draw cycle arrows of the specified cell.
     * @param {OmMatrixCell} cell The off-diagonal cell to draw arrows for.
     */
    drawOffDiagonalArrows(cell) {
        super.drawOffDiagonalArrows(cell);

        /* Cycle arrows are only drawn in the bottom triangle of the diagram */
        if (cell.row > cell.col) {
            const src = this.diagNodes[cell.row];
            const tgt = this.diagNodes[cell.col];

            // Get an array of all the parents and children of the target with cycle arrows
            const relativesWithCycleArrows = tgt.getNodesWithCycleArrows();

            for (const relative of relativesWithCycleArrows) {
                for (const ai of relative.cycleArrows) {
                    if (src.hasNode(ai.src)) {
                        for (const arrow of ai.arrows) {
                            let firstBeginIndex = -1,
                                firstEndIndex = -1;

                            // find first begin index
                            for (let diagIdx in this.diagNodes) {
                                const diagNode = this.diagNodes[diagIdx];
                                if (diagNode.hasNode(arrow.begin)) {
                                    firstBeginIndex = diagIdx;
                                    break;
                                }
                            }
                            if (firstBeginIndex == -1) {
                                throw ("Error: first begin index not found");
                            }

                            // find first end index
                            for (let diagIdx in this.diagNodes) {
                                const diagNode = this.diagNodes[diagIdx];
                                if (diagNode.hasNode(arrow.end)) {
                                    firstEndIndex = diagIdx;
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
}
