// <<hpp_insert gen/ArrowCache.js>>

/**
 * Manage all connection arrow operations. Create new arrows, maintain
 * caches of hovered/pinned arrows, move arrows between them, and
 * transition on updates.
 * @typedef ArrowManager
 * @prop {Object} diagGroups DOM elements referenced by Diagram
 * @prop {ArrowCache} hoverArrows Arrows that disappear if the mouse moves away from the cell
 * @prop {ArrowCache} pinnedArrows Arrows that persist and are redrawn during updates
 * @prop {Object} nodeSize Matrix cell width and height
 * @prop {Number} lineWidth Width of the arrows, calculated from nodeSize.
 * @prop {Object} arrowDirClasses The various offscreen Arrow derived classes.
 */
 class ArrowManager {
    constructor(diagGroups) {
        this.diagGroups = diagGroups;
        this.hoverArrows = new ArrowCache();
        this.pinnedArrows = new ArrowCache();
        this.nodeSize = { 'width': -1, 'height': -1 };
        this._lineWidth = -1;

        this.arrowDirClasses = {
            'top': {
                'incoming': OffGridDownArrow,
                'outgoing': OffGridRightArrow
            },
            'bottom': {
                'incoming': OffGridUpArrow,
                'outgoing': OffGridLeftArrow
            }
        };
    }

    /** Make sure _lineWidth has been manually set */
    get lineWidth() {
        if (this._lineWidth < 0) {
            throw ("addFullArrow: this._lineWidth is unset")
        }

        return this._lineWidth;
    }

    /**
     * Update the node size and calculate the line width.
     * @param {Object} newNodeSize Matrix cell width and height
     */
    setNodeSize(newNodeSize) {
        this.nodeSize = newNodeSize;
        this._lineWidth = Math.min(5, newNodeSize.width * .5, newNodeSize.height * .5);
    }

    /**
     * True if the arrow is referenced in either cache
     * @param {String} arrowId The ID of the arrow to find.
     */
    arrowExists(arrowId) {
        return (this.pinnedArrows.hasArrow(arrowId) ||
            this.hoverArrows.hasArrow(arrowId));
    }

    /**
     * Create a new BentArrow object. This may replace existing elements
     * on the screen with new dimensions and colors. However, the arrow may
     * already exist in one of the caches, in which case it's replaced.
     * @param {String} cellId The ID of the cell that triggered the event.
     * @param {Object} attribs Values to pass to the Arrow constructor.
     * @returns {BentArrow} The newly created arrow object.
     */
    addFullArrow(cellId, attribs) {
        attribs.width = this.lineWidth;
        const newArrow = new BentArrow(attribs, this.diagGroups, this.nodeSize);

        // Add or replace the cache entry with the new arrow
        if (this.pinnedArrows.hasArrow(newArrow.id)) {
            this.pinnedArrows.add(cellId, newArrow);
        }
        else {
            this.hoverArrows.add(cellId, newArrow);
        }

        return newArrow;
    }

    /**
     * Create a new OffGridArrow-derived object. This may replace existing
     * elements on the screen with new dimensions and colors. However, the
     * arrow may already exist in one of the caches, in which case it's not
     * added again.
     * @param {String} cellId The ID of the cell that triggered the event.
     * @param {String} side Whether the arrow is in the top or bottom.
     * @param {String} dir Whether the arrow is incoming or outgoing.
     * @param {Object} attribs Values to pass to the Arrow constructor.
     * @returns {BentArrow} The newly created arrow object.
     */
    addOffGridArrow(cellId, side, dir, attribs) {
        attribs.width = this.lineWidth;
        attribs.cellId = cellId;
        debugInfo("addOffGridArrow(): ", side, dir, attribs)
        const newArrow = new (this.arrowDirClasses[side][dir])(attribs,
            this.diagGroups, this.nodeSize);

        // Add or replace the cache entry with the new arrow
        if (this.pinnedArrows.hasArrow(newArrow.id)) {
            this.pinnedArrows.add(cellId, newArrow);
            this.pinnedArrows.arrows[newArrow.id].cellId = cellId;
        }
        else {
            this.hoverArrows.add(cellId, newArrow);
        }

        return newArrow;
    }

    /**
     * Both endpoints are visible, so draw a full arrow between them.
     * @param {Arrow} arrow The arrow object to transition.
     * @param {MatrixCell} startCell Cell at the beginning of the arrow.
     * @param {MatrixCell} endCell Cell at the end of the arrow.
     */
    _transitionFullArrow(arrow, startCell, endCell) {
        debugInfo(`transition: Found both sides of ${arrow.id}`)
        let attribs = arrow.attribs;
        attribs.start.col = startCell.col;
        attribs.start.row = startCell.row;
        attribs.end.col = endCell.col;
        attribs.end.row = endCell.row;
        attribs.width = this.lineWidth;
        this.pinnedArrows.arrows[arrow.id] =
            new BentArrow(attribs, this.diagGroups, this.nodeSize);
    }

    /**
     * Only the starting cell is visible, so draw an arrow from that
     * heading offscreen in the direction the end would be.
     * @param {Arrow} arrow The arrow object to transition.
     * @param {MatrixCell} startCell Cell at the beginning of the arrow.
     * @param {Matrix} matrix Reference to the matrix object.
     */
    _transitionStartArrow(arrow, startCell, matrix) {
        debugInfo(`transition: Only found start cell for ${arrow.id}`)
        const side = (arrow.attribs.start.id > arrow.attribs.end.id)?
            'bottom' : 'top';
        const attribs = {
            'cell': {
                'col': startCell.col,
                'row': startCell.row,
                'srcId': startCell.srcObj.id,
                'tgtId': startCell.tgtObj.id
            },
            'width': this.lineWidth,
            'cellId': arrow.cellId,
            'matrixSize': matrix.diagNodes.length,
            'offscreenId': arrow.attribs.end.id,
            'label': matrix.model.nodeIds[arrow.attribs.end.id].path,
            'color': arrow.attribs.color
        }
        this.pinnedArrows.arrows[arrow.id] =
            new (this.arrowDirClasses[side]['outgoing'])(attribs,
                    this.diagGroups, this.nodeSize);
    }

    /**
     * Only the ending cell is visible, so draw an arrow to that
     * from offscreen in the direction the starting cell would be.
     * @param {Arrow} arrow The arrow object to transition.
     * @param {MatrixCell} endCell Cell at the end of the arrow.
     * @param {Matrix} matrix Reference to the matrix object.
     */
    _transitionEndArrow(arrow, endCell, matrix) {
        debugInfo(`transition: Only found end cell for ${arrow.id}`)
        const side = (arrow.attribs.start.id > arrow.attribs.end.id)?
            'bottom' : 'top';
        const attribs = {
            'cell': {
                'col': endCell.col,
                'row': endCell.row,
                'srcId': endCell.srcObj.id,
                'tgtId': endCell.tgtObj.id
            },
            'width': this.lineWidth,
            'cellId': arrow.cellId,
            'matrixSize': matrix.diagNodes.length,
            'offscreenId': arrow.attribs.start.id,
            'label': matrix.model.nodeIds[arrow.attribs.start.id].path,
            'color': arrow.attribs.color
        }
        this.pinnedArrows.arrows[arrow.id] =
            new (this.arrowDirClasses[side]['incoming'])(attribs,
                this.diagGroups, this.nodeSize);
    }

    /**
     * Handle nodes that were uncollapsed with pinned arrows by pinning arrows
     * to their visible child nodes. This is done after the rest of the arrow
     * transitions because new arrows are added to the cache.
     * @param {Array} uncollapsedNodeIds List of nodeIds that were uncollapsed.
     * @param {Matrix} matrix Reference to the matrix object.
     */
    _transitionUncollapsedNodes(uncollapsedNodeIds, matrix) {
        for (const row in matrix.grid) {
            const cell = matrix.grid[row][row]; // Diagonal cells only
            for (const nodeId of uncollapsedNodeIds) {
                if (cell.obj.hasParent(matrix.model.nodeIds[nodeId])) {
                    matrix.drawOnDiagonalArrows(cell);
                    this.togglePin(cell.id, true);
                }
            }
        }
    }

    /**
     * Redraw all the visible arrows in the pinned arrow cache, and remove
     * the ones for which neither endpoint is visible. Full arrows may
     * need to transition to offgrid arrows and vice versa.
     * @param {Matrix} matrix The matrix to operate with.
     */
    transition(matrix) {
        let uncollapsedNodeIds = [];

        for (const arrowId in this.pinnedArrows.arrows) {
            const arrow = this.pinnedArrows.arrows[arrowId];
            const startCellInfo = matrix.findCellByNodeId(arrow.attribs.start.id);
            const endCellInfo = matrix.findCellByNodeId(arrow.attribs.end.id);
            const startCell = startCellInfo.cell;
            const endCell = endCellInfo.cell;

            if (startCell === endCell) { // Both undefined, or same cell
                if (startCell === undefined)
                    debugInfo(`transition: No visible endpoints for ${arrowId}`)
                else
                    debugInfo(`transition: ${arrowId} points to and from` +
                        ` collapsed cell ${startCell.id}`)

                this.pinnedArrows.removeArrowFromScreen(arrowId);
            }
            else if (startCellInfo.childMatch || endCellInfo.childMatch) {
                debugInfo(`transition: ${arrowId} endpoint was previously collapsed, \
                    pinning arrows for all children.`)
                if (startCellInfo.childMatch)
                    uncollapsedNodeIds.push(arrow.attribs.start.id);
                if (endCellInfo.childMatch)
                    uncollapsedNodeIds.push(arrow.attribs.end.id);

                this.pinnedArrows.removeArrowFromScreen(arrowId);
                this.pinnedArrows.removeArrowFromCache(arrowId);
            }
            else if ((startCellInfo.exactMatch || startCellInfo.parentMatch) &&
                (endCellInfo.exactMatch || endCellInfo.parentMatch)) {
                // Both endpoint cells are visible
                this._transitionFullArrow(arrow, startCell, endCell);
            }
            else if (startCell) {
                // Only the non-pointy end is visible
                this._transitionStartArrow(arrow, startCell, matrix);
            }
            else if (endCell) {
                // Only the pointy end is visible
                this._transitionEndArrow(arrow, endCell, matrix);
            }
        }

        // Adding arrows, so do after iteration through arrow list
        this._transitionUncollapsedNodes(uncollapsedNodeIds, matrix);
    }

    /**
     * If arrows are hovering, then pin them, and vice versa.
     * @param {String} cellId The ID of the MatrixCell to operate on.
     * @param {Boolean} [ pinOnly = false] If true, don't unpin anything.
     */
    togglePin(cellId, pinOnly = false) {
        const cellClassName = "n2_hover_elements_" + cellId;
        if (this.pinnedArrows.hasEventCell(cellId)) { // Arrows already pinned
            if (pinOnly) return;
            debugInfo(`Unpinning ${cellId} arrows`)
            this.hoverArrows.migrateCell(cellId, this.pinnedArrows,
                'n2_hover_elements', cellClassName);
            this.removeAllHovered();
        }
        else if (this.hoverArrows.hasEventCell(cellId)) { // Arrows just "hovered"
            debugInfo(`Pinning ${cellId} arrows`)
            this.pinnedArrows.migrateCell(cellId, this.hoverArrows,
                cellClassName, 'n2_hover_elements');
        }
    }

    /** Remove all arrows in the hoverArrow cache */
    removeAllHovered() {
        const removedArrowIds = Object.keys(this.hoverArrows.arrows);
        this.hoverArrows.removeAll();
        debugInfo(`removeAllHovered(): Removed ${removedArrowIds.length} arrows`)
    }

    /** Remove all arrows in the pinnedArrow cache */
    removeAllPinned() {
        const removedArrowIds = Object.keys(this.pinnedArrows.arrows);
        this.pinnedArrows.removeAll();
        debugInfo(`removeAllPinned(): Removed ${removedArrowIds.length} arrows`)
    }

    /* Save all pinnedArrows to a dictionary for saving the view. */
    savePinnedArrows() {
        let pinned = this.pinnedArrows;
        let data = {};

        for (const arrowId in pinned.arrows) {
            const arrow = pinned.arrows[arrowId];

            if (arrow.cell !== undefined) {
                // Off screen connection.
                data[arrowId] = [arrow.cellId, arrow.direction,
                                 arrow.cell.col, arrow.cell.row,
                                 arrow.cell.srcId, arrow.cell.tgtId,
                                 arrow.attribs.matrixSize, arrow.attribs.label, arrow.attribs.offscreenId];
            }
            else {
                // On screen connection.
                data[arrowId] = [arrow.cellId,
                                 arrow.start.col, arrow.start.row, arrow.start.id,
                                 arrow.end.col, arrow.end.row, arrow.end.id,
                                 arrow.color];
            }

        }
        return data;
    }

    /* Restore all pinnedArrows that were saved in a view. */
    loadPinnedArrows(arrows) {
        this.pinnedArrows.removeAll();

        for(const arrowID in arrows) {
            const arrow = arrows[arrowID];

            if (arrow.length == 9) {
                // Off screen arrow.
                let arrowClasses = {
                    'down': OffGridDownArrow,
                    'up': OffGridUpArrow,
                    'left': OffGridLeftArrow,
                    'right': OffGridRightArrow,
                };

                let attribs = {
                    'cell': {
                        'col': arrow[2],
                        'row': arrow[3],
                        'srcId': arrow[4],
                        'tgtId': arrow[5],
                    },
                    'matrixSize': arrow[6],
                    'label': arrow[7],
                    'offscreenId': arrow[8],
                };

                attribs.width = this.lineWidth;
                const newArrow = new (arrowClasses[arrow[1]])(attribs,
                    this.diagGroups, this.nodeSize);
                this.pinnedArrows.add(arrow[0], newArrow);
            }
            else {
                // On screen arrow.
                let attribs = {
                    'start': {
                        'col': arrow[1],
                        'row': arrow[2],
                        'id': arrow[3]
                    },
                    'end': {
                        'col': arrow[4],
                        'row': arrow[5],
                        'id': arrow[6]
                    },
                    'color': arrow[7],
                };
                attribs.width = this.lineWidth;
                const newArrow = new BentArrow(attribs, this.diagGroups, this.nodeSize);
                this.pinnedArrows.add(arrow[0], newArrow);
            }
        }
    }
}
