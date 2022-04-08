// <<hpp_insert gen/Arrow.js>>

/**
 * Maintain a set of arrows, usually for pinning/unpinning.
 * @typedef ArrowCache
 * @property {Object} arrows Every individual arrow keyed by it's ID
 * @property {Object} eventCells Cells that triggered the arrows, with a Set
 *   of associated arrow IDs.
 */
class ArrowCache {
    /**
     * Initialize with empty cache.
     */
    constructor() {
        this.arrows = {};
        this.eventCells = {};
    }

    /**
     * Determine if the arrow exists in this cache.
     * @param {String} arrowId The ID of the arrow to look for.
     * @returns {Boolean} True if found.
     */
    hasArrow(arrowId) {
        return exists(this.arrows[arrowId]);
    }

    /**
     * Find an arrow that is pointing to or from the specified cell.
     * @param {String} cellId The ID of the cell to search for.
     * @returns {String} True if any arrow found, otherwise false.
     */
    hasCell(cellId) {
        for (const id in this.arrows) {
            if (this.arrows[id].connectsToCell(cellId)) return true;
        }

        return false;
    }

    /**
     * Add an individual arrow to the cache.
     * @param {String} cellId The ID of the cell that triggered the event.
     * @param {Arrow} arrow The arrow object to cache.
     * @param {Boolean} [allowReplace = true] Replacing an existing arrow is OK.
     */
    add(cellId, arrow, allowReplace = true) {
        arrow.cellId = cellId;

        if (this.hasArrow(arrow.id) && !allowReplace) {
            console.warn(`ArrowCache.add(): Not adding arrow ${arrow.id} to cache
                    since it already exists.`)
        }
        else {
            this.arrows[arrow.id] = arrow;
            this.addEventCellArrow(cellId, arrow.id);
        }
    }

    /**
     * Associate a cell with an arrow that a mouse event triggered.
     * @param {String} cellId The ID of the event-triggering cell.
     * @param {String} arrowId The ID of the arrow.
     */
    addEventCellArrow(cellId, arrowId) {
        if (!exists(this.eventCells[cellId])) this.eventCells[cellId] = new Set();
        this.eventCells[cellId].add(arrowId);
    }

    /**
     * Determine if the cell triggered any cached arrows.
     * @param {String} cellId The ID of the cell to check.
     * @returns {Boolean} True if the cellId is tracked.
     */
    hasEventCell(cellId) { return exists(this.eventCells[cellId]); }

    /**
     * Disassociate an arrow with a trigger cell. If the cell has no more arrows,
     * remove the entry.
     * @param {String} cellId The ID of the event-triggering cell.
     * @param {String} arrowId The ID of the arrow.
     */
    removeEventCellArrow(cellId, arrowId) {
        if (!exists(this.eventCells[cellId])) {
            console.warn(`removeEventCellArrow: no tracked cell with ID ${cellId}`)
        }
        else if (!this.eventCells[cellId].has(arrowId)) {
            console.warn(`removeEventCellArrow: no arrow with ID ${arrowId}
                associated with cell ID ${cellId}`)
        }
        else {
            this.eventCells[cellId].delete(arrowId)
            if (this.eventCells[cellId].size == 0) delete this.eventCells[cellId];
        }
    }

    /**
     * Delete all associations for the cell and stop tracking it.
     * @param {String} cellId The ID of the event-triggering cell.
     */
    removeEventCell(cellId) {
        if (!exists(this.eventCells[cellId])) {
            console.warn(`removeEventCellArrow: no tracked cell with ID ${cellId}`)
        }
        else {
            this.eventCells[cellId].clear();
            delete this.eventCells[cellId];
        }
    }

    /**
     * Remove drawn arrow from the screen, but keep the info cached.
     * @param {String} arrowId The ID of the single arrow to remove.
     */
    removeArrowFromScreen(arrowId) {
        if (!this.hasArrow(arrowId)) {
            console.warn(`removeArrowFromScreen: Arrow ${arrowId} doesn't exist.`)
        }
        else {
            this.arrows[arrowId].group.remove();
        }
    }

    /**
     * Remove arrow info from cache, but leave the elements on the screen.
     * @param {String} arrowId The ID of the single arrow to remove.
     */
    removeArrowFromCache(arrowId) {
        if (!this.hasArrow(arrowId)) {
            console.warn(`removeArrowFromCache: Arrow ${arrowId} doesn't exist.`)
        }
        else {
            delete this.arrows[arrowId];
            for (const cellId in this.eventCells) {
                if (this.eventCells[cellId].has(arrowId))
                    this.eventCells[cellId].delete(arrowId);
            }
        }
    }

    /**
     * Remove everything we're tracking from screen and cache.
     */
    removeAll() {
        for (const arrowId in this.arrows) {
            this.removeArrowFromScreen(arrowId);
            this.removeArrowFromCache(arrowId);
        }

        this.arrows = {};
        this.eventCells = {};
    }

    /**
     * Move an arrow from one cache to another and change its CSS classes.
     * @param {String} cellId The cell ID that triggered the event.
     * @param {String} arrowId The ID of the single arrow.
     * @param {ArrowCache} oldCache The cache to move the arrow from.
     * @param {String} newClassName The CSS class to add to HTML elements.
     * @param {String} oldClassName The CSS class to remove from HTML elements.
     */
    migrateArrow(cellId, arrowId, oldCache, newClassName, oldClassName) {
        const arrow = oldCache.arrows[arrowId];
        this.add(cellId, arrow);
        oldCache.removeArrowFromCache(arrowId);

        debugInfo(`migrateArrow(): Moving ${arrowId} from class ${oldClassName} to ${newClassName}`);

        arrow.cssClass = newClassName;
        arrow.group
            .classed(oldClassName, false)
            .classed(newClassName, true)
            .selectAll('.' + oldClassName)
            .classed(oldClassName, false)
            .classed(newClassName, true);
    }

    /**
     * Migrate all arrows associated with a cell into this cache from another.
     * @param {String} cellId The ID of the cell to operate on.
     * @param {ArrowCache} oldCache The cache to move the arrows from.
     * @param {String} newClassName The CSS class to add to HTML elements.
     * @param {String} oldClassName The CSS class to remove from HTML elements.
     */
    migrateCell(cellId, oldCache, newClassName, oldClassName) {
        if (!oldCache.hasEventCell(cellId)) {
            console.warn(`migrateCell: Event cell ${cellId} isn't tracked in
                the old cache, can't migrate arrows.`)
        }
        else {
            for (const arrowId of oldCache.eventCells[cellId]) {
                this.migrateArrow(cellId, arrowId, oldCache, newClassName, oldClassName);
            }
            delete oldCache.eventCells[cellId];
        }
    }
}
