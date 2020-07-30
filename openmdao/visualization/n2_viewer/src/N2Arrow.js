/**
 * Base class for all types of N2 arrows.
 * @typedef N2Arrow
 * @prop {Number} attribs.start.row Row # of starting node
 * @prop {Number} attribs.start.col Col # of starting ndoe
 * @prop {String} attribs.start.id ID of the start node
 * @prop {Number} attribs.end.row Row # of end node (arrow points to)
 * @prop {Number} attribs.end.col Col # of end node (arrow points to)
 * @prop {String} attribs.end.id ID of the end node
 * @prop {String} color Color of the line/circle (arrowhead is black)
 * @prop {Number} width Width of the arrow path
 * @prop {Number} nodeSize.width Width of a matrix cell in pixels.
 * @prop {Number} nodeSize.height Height of a matrix cell in pixels.
 */
class N2Arrow {
    /**
     * Determine whether arrow elements already exist, and create them if not.
     * @param {Object} attribs All of the specific arrow properties.
     * @param {Object} n2Groups References to <g> SVG elements.
     * @param {Object} nodeSize The dimensions of each cell in the matrix.
     */
    constructor(attribs, n2Groups, nodeSize) {
        this.color = attribs.color;
        this.width = attribs.width;
        this.nodeSize = nodeSize;
        this.attribs = attribs;
        this._genPath = this._angledPath;

        /*
         * Generate a CSS id for the arrow. This is based on the ids of the
         * N2TreeNodes it points to rather than the cells, so if it's pinned
         * and the matrix is redrawn, we know whether to transition it. The color
         * is included because there can be identical arrows of different colors
         * stacked on each other.
         */
        this.id = 'arrow-' + attribs.start.id + '-to-'
            + attribs.end.id + '-' + this.color.replace(/#/, '');

        const existingArrow = d3.select('g#' + this.id);
        if (existingArrow.empty()) {
            debugInfo(`N2Arrow(): Creating new ${this.id}`);
            this.doTransition = false;
            this.cssClass = 'n2_hover_elements';
            this.group = n2Groups.arrows.append('g')
                .attr('id', this.id)
                .attr('class', this.cssClass)
        }
        else {
            this.doTransition = true;
            this.group = existingArrow;
            const cssClasses = String(this.group.attr('class')).split(' ');
            this.cssClass = cssClasses.find(o => o.match(/n2_hover_elements.*/));
            debugInfo(`N2Arrow(): Using existing ${this.id} with class ${this.cssClass}`);
        }
    }

    connectsToCell(cellId) {
        return (this.attribs.start.id == cellId ||
            this.attribs.end.id == cellId);
    }

    get offsetAbsX() {
        return this.nodeSize.width * N2Arrow.cellOverlap;
    }

    get offsetAbsY() {
        return this.nodeSize.height * N2Arrow.cellOverlap + 3;  // + to account for the arrow size
    }
}

N2Arrow.cellOverlap = .125;

/**
 * Draws a two-segment arrow, using provided beginning and end node locations.
 */
class N2BentArrow extends N2Arrow {
    constructor(attribs, n2Groups, nodeSize) {
        super(attribs, n2Groups, nodeSize);

        this.start = attribs.start;
        this.end = attribs.end;

        this.middle = { 'row': -1, 'col': -1 };

        this.pts = {
            'start': { 'x': -1, 'y': -1 },
            'mid': { 'x': -1, 'y': -1 },
            'end': { 'x': -1, 'y': -1 }
        }

        this._computePts();
        this.draw();
    }

    /**
     * Determine the location of the "middle" node, where the bend is. Calculate
     * the points corresponding to the associated col, row in the matrix.
     */
    _computePts() {
        this.middle.row = this.start.row;
        this.middle.col = this.end.col;
        let offsetAbsX = this.offsetAbsX;
        let offsetAbsY = this.offsetAbsY;

        this.offsetX = (this.start.col < this.end.col) ? offsetAbsX : -offsetAbsX; // Left-to-Right : Right-to-Left
        this.pts.start.x = this.nodeSize.width * this.start.col + this.nodeSize.width * .5 + this.offsetX;
        this.pts.mid.x = this.nodeSize.width * this.middle.col + this.nodeSize.width * .5;
        // this.pts.mid.x = (this.offsetX > 0)? this.nodeSize.width * this.middle.col : this.nodeSize.width * (this.middle.col + 1);
        this.pts.end.x = this.nodeSize.width * this.end.col + this.nodeSize.width * .5;

        let offsetY = (this.start.row < this.end.row) ? -offsetAbsY : offsetAbsY; // Down : Up
        this.pts.start.y = this.nodeSize.height * this.start.row + this.nodeSize.height * .5;
        this.pts.mid.y = this.nodeSize.height * this.middle.row + this.nodeSize.height * .5;
        this.pts.end.y = this.nodeSize.height * this.end.row + this.nodeSize.height * .5 + offsetY;
    }

    /** Create a path string with a quadratic curve at the bend. */
    _curvedPath() {
        const dir = (this.offsetX > 0) ? 1 : -1;
        const s = this.nodeSize.width * .5 * dir;

        return "M" + this.pts.start.x + " " + this.pts.start.y +
            " L" + this.pts.mid.x + " " + this.pts.mid.y +
            ` q${s} 0 ${s} ${s}` +
            " L" + this.pts.end.x + " " + this.pts.end.y;
    }

    /** Generate a path with a 90-degree angle at the bend. */
    _angledPath() {
        return "M" + this.pts.start.x + " " + this.pts.start.y +
            " L" + this.pts.mid.x + " " + this.pts.mid.y +
            " L" + this.pts.end.x + " " + this.pts.end.y;
    }

    /** Set up for a black/gray dot covered by a colored dot at the arrow bend. */
    _createDots() {
        this.bottomCircle = this.group.append("circle")
            .attr('id', 'bottom-circle')
            .attr("class", this.cssClass);

        this.topCircle = this.group.append("circle")
            .attr('id', 'top-circle')
            .attr("class", this.cssClass)
    }

    /** Use SVG to draw the line segments and an arrow at the end-point. */
    draw() {
        if (this.doTransition) {
            // Arrow already exists, size and/or shape needs updated
            this.path = this.group.select('path').transition(sharedTransition);

            if (this.group.classed('off-grid-arrow')) {
                // The arrow was previously going offscreen but now is fully onscreen
                this._createDots();
            }
            else {
                this.bottomCircle = this.group.select('circle#bottom-circle').transition(sharedTransition);
                this.topCircle = this.group.select('circle#top-circle').transition(sharedTransition);
            }
        }
        else {
            // This is an entirely new arrow
            this.path = this.group.append("path")
                .attr('id', 'arrow-path')
                .attr("class", this.cssClass);

            this._createDots();
        }

        this.group.classed('off-grid-arrow', false);

        this.path
            .attr("marker-end", "url(#arrow)")
            .attr('stroke-dasharray', null)
            .attr("d", this._genPath())
            .attr("fill", "none")
            .style("stroke-width", this.width)
            .style("stroke", this.color);

        this.bottomCircle
            .attr("cx", this.pts.mid.x)
            .attr("cy", this.pts.mid.y)
            .attr("r", this.width * 1.0)
            .style("stroke-width", 0)
            .style("fill-opacity", 1)
            .style("fill", N2Style.color.connection);

        this.topCircle
            .attr("cx", this.pts.mid.x)
            .attr("cy", this.pts.mid.y)
            .attr("r", this.width * 1.0)
            .style("stroke-width", 0)
            .style("fill-opacity", .75)
            .style("fill", this.color);
    }
}

/**
 * Draw a straight, dashed arrow with one side connected to an
 * offscreen node. Show a tooltip near the offscreen.
 */
class N2OffGridArrow extends N2Arrow {
    constructor(attribs, n2Groups, nodeSize) {
        super(attribs, n2Groups, nodeSize);

        this.cell = attribs.cell;

        this.label = {
            'text': attribs.label,
            'pts': {},
            'ref': null,
            'labels': {}
        }

        this.pts = {
            'start': { 'x': -1, 'y': -1 },
            'end': { 'x': -1, 'y': -1 }
        }

        /* Get the bounding rect for the N2 matrix background rect (use rect
         * because getting the bounding rect of a group doesn't work well).
         * This is used to compute the locations of the tooltips since they're divs
         * and not SVG objects.
         */
        this.bgRect = d3.select("#backgroundRect").node().getBoundingClientRect();

    }

    /** 
     * Add the pathname of the offscreen node to the label, if it's not
     * already in there. 
     * @returns {Boolean} True if the label didn't already contain the pathname.
    */
    _addToLabel() {
        let visValue = this.label.ref.style('visibility');
        let firstEntry = (!visValue.match(/visible/));
        let tipHTML = this.label.ref.node().innerHTML;

        // Prevent duplicate listings
        if (!firstEntry && tipHTML.match(this.label.text)) {
            debugInfo('Duplicate entry for label to ' + this.label.text);
            this.newLabel = false;
            return false;
        }

        this.newLabel = true;

        if (firstEntry) {
            this.label.ref
                .style('visibility', 'visible')
                .node().innerHTML = this.label.text;
        }
        else {
            this.label.ref.node().innerHTML += '<br>' + this.label.text;
        }

        // After adding the text, store the bounding rect of the label
        // so that the new position can be calculated.
        this.label.rect = this.label.ref.node().getBoundingClientRect();

        return true;
    }

    /** Put the SVG arrow on the screen and position the tooltip. */
    draw() {
        debugInfo('Adding offscreen ' + this.attribs.direction +
            ' arrow connected to ' + this.label.text);

        if (this.doTransition) {
            if (!this.group.classed('off-grid-arrow')) {
                // If it was previously a bent arrow, remove the dots.
                this.group.selectAll('circle').remove();
            }

            this.path = this.group.select('path').transition(sharedTransition);
        }
        else {
            this.path = this.group.insert('path')
                .attr('id', 'arrow-path')
                .attr('class', this.cssClass);
        }

        this.group.classed('off-grid-arrow', true);

        const mid = {
                'x': this.pts.start.x - (this.pts.start.x - this.pts.end.x)/2,
                'y': this.pts.start.y - (this.pts.start.y - this.pts.end.y)/2
            }

        this.path
            .attr('marker-end', 'url(#arrow)')
            .attr('stroke-dasharray', '5,5')
            .attr('d', `M${this.pts.start.x},${this.pts.start.y} 
                 L${mid.x},${mid.y} L${this.pts.end.x},${this.pts.end.y}`)
            .attr('fill', 'none')
            .style('stroke-width', this.width)
            .style('stroke', this.color);

        for (let pos in this.label.pts) {
            this.label.ref.style(pos, this.label.pts[pos] + 'px');
        }
    }
}

/**
 * Draw an arrow on the bottom half of the N2 matrix coming up into
 * the onscreen target cell from the offscreen source.
 */
class N2OffGridUpArrow extends N2OffGridArrow {
    constructor(attribs, n2Groups, nodeSize) {
        super(Object.assign(attribs, {
            'start': { 'id': attribs.offscreenId },
            'end': { 'id': attribs.cell.tgtId },
            'direction': 'up',
            'color': attribs.color ? attribs.color : N2Style.color.inputArrow
        }), n2Groups, nodeSize);

        this.label.ref = d3.select("div#left.offgrid");

        this._addToLabel();
        this._computePts();
        this.draw();
    }

    /**
     * Use the start coordinates and the direction of the arrow to determine
     * the starting coordinates and location of the arrowhead.
     */
    _computePts() {
        let offsetY = this.offsetAbsY;

        // Arrow
        this.pts.start.x = this.pts.end.x =
            this.nodeSize.width * this.cell.col + this.nodeSize.width * .5;
        this.pts.start.y = this.attribs.matrixSize * this.nodeSize.height +
            this.nodeSize.height * .5 + offsetY;
        this.pts.end.y = this.nodeSize.height * this.cell.row +
            this.nodeSize.height * .5 + offsetY

        // Tooltip
        if (this.newLabel) {
            this.label.pts.left = this.pts.start.x + this.bgRect.left -
                this.label.rect.width / 2;
            this.label.pts.top = this.bgRect.bottom + 2;
        }
    }
}

/**
 * Draw arrow on the top half of the N2 matrix coming down into the
 * onscreen target cell from the offscreen source.
 */
class N2OffGridDownArrow extends N2OffGridArrow {
    constructor(attribs, n2Groups, nodeSize) {
        super(Object.assign(attribs, {
            'start': { 'id': attribs.offscreenId },
            'end': { 'id': attribs.cell.tgtId },
            'direction': 'down',
            'color': attribs.color ? attribs.color : N2Style.color.inputArrow
        }), n2Groups, nodeSize);

        this.label.ref = d3.select("div#bottom.offgrid");

        this._addToLabel();
        this._computePts();
        this.draw();
    }

    /**
     * Use the start coordinates and the direction of the arrow to determine
     * the starting coordinates and location of the arrowhead.
     */
    _computePts() {
        let offsetY = this.offsetAbsY;

        // Arrow
        this.pts.start.x = this.pts.end.x =
            this.nodeSize.width * this.cell.col + this.nodeSize.width * .5;
        this.pts.start.y = this.nodeSize.height * -.5 - offsetY;
        this.pts.end.y = this.nodeSize.height * this.cell.row +
            this.nodeSize.height * .5 - offsetY;

        // Tooltip
        if (this.newLabel) {
            this.label.pts.left = this.pts.start.x + this.bgRect.left -
                this.label.rect.width / 2;
            this.label.pts.top = this.bgRect.top - this.label.rect.height - 2;
        }
    }
}

/**
 * Draw arrow on the bottom half of N2 matrix going left from the
 * onscreen source cell to the offscreen target.
 */
class N2OffGridLeftArrow extends N2OffGridArrow {
    constructor(attribs, n2Groups, nodeSize) {
        super(Object.assign(attribs, {
            'start': { 'id': attribs.cell.srcId },
            'end': { 'id': attribs.offscreenId },
            'direction': 'left',
            'color': attribs.color ? attribs.color : N2Style.color.outputArrow
        }), n2Groups, nodeSize);

        this.label.ref = d3.select("div#bottom.offgrid");

        this._addToLabel();
        this._computePts();
        this.draw();
    }

    /**
     * Use the start coordinates and the direction of the arrow to determine
     * the starting coordinates and location of the arrowhead.
     */
    _computePts() {
        let offsetX = this.offsetAbsX;

        // Arrow
        this.pts.start.y = this.pts.end.y =
            this.nodeSize.height * this.cell.row + this.nodeSize.height * .5;
        this.pts.start.x = this.nodeSize.width * this.cell.col +
            this.nodeSize.width * .5 - offsetX;
        this.pts.end.x = 0;

        // Tooltip
        if (this.newLabel) {
            this.label.pts.left = this.pts.end.x + this.bgRect.left -
                this.label.rect.width * 0.667;
            this.label.pts.top = this.pts.end.y + this.bgRect.top +
                this.label.rect.height / 2;
        }
    }
}

/**
 * Draw arrow on the top half of the N2 matrix going right away from
 * the onscreen source cell to the offscreen target.
 */
class N2OffGridRightArrow extends N2OffGridArrow {
    constructor(attribs, n2Groups, nodeSize) {
        super(Object.assign(attribs, {
            'start': { 'id': attribs.cell.srcId },
            'end': { 'id': attribs.offscreenId },
            'direction': 'right',
            'color': attribs.color ? attribs.color : N2Style.color.outputArrow
        }), n2Groups, nodeSize);

        this.label.ref = d3.select("div#right.offgrid");

        this._addToLabel();
        this._computePts();
        this.draw();
    }

    /**
     * Use the start coordinates and the direction of the arrow to determine
     * the starting coordinates and location of the arrowhead.
     */
    _computePts() {
        let offsetX = this.offsetAbsX;

        // Arrow
        this.pts.start.y = this.pts.end.y =
            this.nodeSize.height * this.cell.row + this.nodeSize.height * .5;
        this.pts.start.x = this.nodeSize.width * this.cell.col +
            this.nodeSize.width * .5 + offsetX;
        this.pts.end.x = this.attribs.matrixSize * this.nodeSize.width;

        // Tooltip
        if (this.newLabel) {
            this.label.pts.left = this.pts.end.x + this.bgRect.left -
                this.label.rect.width / 3;
            this.label.pts.top = this.pts.end.y + this.bgRect.top +
                this.label.rect.height / 2;
        }
    }
}

/**
 * Maintain a set of arrows, usually for pinning/unpinning.
 * @typedef N2ArrowCache
 * @property {Object} arrows Every individual arrow keyed by it's ID
 * @property {Object} eventCells Cells that triggered the arrows, with a Set
 *   of associated arrow IDs.
 */
class N2ArrowCache {
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
     * @param {N2Arrow} arrow The arrow object to cache.
     * @param {Boolean} [allowReplace = true] Replacing an existing arrow is OK.
     */
    add(cellId, arrow, allowReplace = true) {
        if (this.hasArrow(arrow.id) && !allowReplace) {
            console.warn(`N2ArrowCache.add(): Not adding arrow ${arrow.id} to cache
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
     * @param {N2ArrowCache} oldCache The cache to move the arrow from.
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
     * @param {N2ArrowCache} oldCache The cache to move the arrows from.
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

/**
 * Manage all connection arrow operations. Create new arrows, maintain
 * caches of hovered/pinned arrows, move arrows between them, and 
 * transition on updates.
 * @typedef N2ArrowManager
 * @prop {Object} n2Groups DOM elements referenced by N2Diagram
 * @prop {N2ArrowCache} hoverArrows Arrows that disappear if the mouse moves away from the cell
 * @prop {N2ArrowCache} pinnedArrows Arrows that persist and are redrawn during updates
 * @prop {Object} nodeSize Matrix cell width and height
 * @prop {Number} lineWidth Width of the arrows, calculated from nodeSize.
 * @prop {Object} arrowDirClasses The various offscreen N2Arrow derived classes.
 */
class N2ArrowManager {
    constructor(n2Groups) {
        this.n2Groups = n2Groups;
        this.hoverArrows = new N2ArrowCache();
        this.pinnedArrows = new N2ArrowCache();
        this.nodeSize = { 'width': -1, 'height': -1 };
        this._lineWidth = -1;

        this.arrowDirClasses = {
            'top': {
                'incoming': N2OffGridDownArrow,
                'outgoing': N2OffGridRightArrow
            },
            'bottom': {
                'incoming': N2OffGridUpArrow,
                'outgoing': N2OffGridLeftArrow
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
     * Create a new N2BentArrow object. This may replace existing elements
     * on the screen with new dimensions and colors. However, the arrow may
     * already exist in one of the caches, in which case it's replaced.
     * @param {String} cellId The ID of the cell that triggered the event.
     * @param {Object} attribs Values to pass to the N2Arrow constructor.
     * @returns {N2BentArrow} The newly created arrow object.
     */
    addFullArrow(cellId, attribs) {
        attribs.width = this.lineWidth;
        const newArrow = new N2BentArrow(attribs, this.n2Groups, this.nodeSize);

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
     * Create a new N2OffGridArrow-derived object. This may replace existing
     * elements on the screen with new dimensions and colors. However, the
     * arrow may already exist in one of the caches, in which case it's not
     * added again.
     * @param {String} cellId The ID of the cell that triggered the event.
     * @param {String} side Whether the arrow is in the top or bottom.
     * @param {String} dir Whether the arrow is incoming or outgoing.
     * @param {Object} attribs Values to pass to the N2Arrow constructor.
     * @returns {N2BentArrow} The newly created arrow object.
     */
    addOffGridArrow(cellId, side, dir, attribs) {
        attribs.width = this.lineWidth;
        debugInfo("addOffGridArrow(): ", side, dir, attribs)
        const newArrow = new (this.arrowDirClasses[side][dir])(attribs,
            this.n2Groups, this.nodeSize);

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
     * Both endpoints are visible, so draw a full arrow between them.
     * @param {N2Arrow} arrow The arrow object to transition.
     * @param {N2MatrixCell} startCell Cell at the beginning of the arrow.
     * @param {N2MatrixCell} endCell Cell at the end of the arrow.
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
            new N2BentArrow(attribs, this.n2Groups, this.nodeSize);
    }

    /**
     * Only the starting cell is visible, so draw an arrow from that
     * heading offscreen in the direction the end would be.
     * @param {N2Arrow} arrow The arrow object to transition.
     * @param {N2MatrixCell} startCell Cell at the beginning of the arrow.
     * @param {N2Matrix} matrix Reference to the matrix object.
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
            'matrixSize': matrix.diagNodes.length,
            'offscreenId': arrow.attribs.end.id,
            'label': matrix.model.nodeIds[arrow.attribs.end.id].absPathName,
            'color': arrow.attribs.color
        }
        this.pinnedArrows.arrows[arrow.id] =
            new (this.arrowDirClasses[side]['outgoing'])(attribs,
                    this.n2Groups, this.nodeSize);
    }

    /**
     * Only the ending cell is visible, so draw an arrow to that
     * from offscreen in the direction the starting cell would be.
     * @param {N2Arrow} arrow The arrow object to transition.
     * @param {N2MatrixCell} endCell Cell at the end of the arrow.
     * @param {N2Matrix} matrix Reference to the matrix object.
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
            'matrixSize': matrix.diagNodes.length,
            'offscreenId': arrow.attribs.start.id,
            'label': matrix.model.nodeIds[arrow.attribs.start.id].absPathName,
            'color': arrow.attribs.color
        }
        this.pinnedArrows.arrows[arrow.id] =
            new (this.arrowDirClasses[side]['incoming'])(attribs,
                this.n2Groups, this.nodeSize);       
    }

    /**
     * Handle nodes that were uncollapsed with pinned arrows by pinning arrows
     * to their visible child nodes. This is done after the rest of the arrow
     * transitions because new arrows are added to the cache.
     * @param {Array} uncollapsedNodeIds List of nodeIds that were uncollapsed.
     * @param {N2Matrix} matrix Reference to the matrix object.
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
     * @param {N2Matrix} matrix The matrix to operate with.
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
     * @param {String} cellId The ID of the N2MatrixCell to operate on.
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
        else {
            console.warn(`togglePin(): No known arrows for cell ${cellId}.`);
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
}
