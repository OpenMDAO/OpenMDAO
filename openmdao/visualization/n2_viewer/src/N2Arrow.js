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

        const existingArrow = d3.select('g#' + this.id);
        if (existingArrow.empty()) {
            debugInfo(`N2Arrow(): Creating new ${this.id}`);
            this.doTransition = false;
            this.group = n2Groups.arrows.append('g')
                .attr('id', this.id)
                .attr('class', 'n2_hover_elements')
        }
        else {
            debugInfo(`N2Arrow(): Using existing ${this.id}`);
            this.doTransition = true;
            this.group = existingArrow;
        }
    }

    get offsetAbsX() {
        return this.nodeSize.width * N2Arrow.cellOverlap;
    }

    get offsetAbsY() {
        return this.nodeSize.height * N2Arrow.cellOverlap + 3;  // + to account for the arrow size
    }

    /**
     * Generate a CSS id for the arrow. This is based on the ids of the
     * N2TreeNodes it points to rather than the cells, so if it's pinned
     * and the matrix is redrawn, we know whether to transition it. The color
     * is included because there can be identical arrows of different colors
     * stacked on each other. Derived arrow classes may need to redefine
     * this based on how they find node ids.
     */
    get id() {
        return 'arrow-' + this.attribs.start.id + '-to-'
            + this.attribs.end.id + '-' + this.attribs.color.replace(/#/, '');
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
            .attr("class", "n2_hover_elements");

        this.topCircle = this.group.append("circle")
            .attr('id', 'top-circle')
            .attr("class", "n2_hover_elements")
    }

    /** Use SVG to draw the line segments and an arrow at the end-point. */
    draw() {
        if (this.doTransition) {
            // Arrow already exists, size and/or shape needs updated
            this.path = this.group.select('path#arrow-path').transition(sharedTransition);

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
            this.path = this.group.insert("path")
                .attr('id', 'arrow-path')
                .attr("class", "n2_hover_elements");

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
    constructor(attribs, n2Groups, nodeSize, markerSize = null) {
        super(attribs, n2Groups, nodeSize, markerSize);

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
            return false;
        }

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

            this.path = this.group.select('path');
        }
        else {
            this.path = this.group.insert('path')
                .attr('class', 'n2_hover_elements');
        }

        this.group.classed('off-grid-arrow', true);

        this.path
            .attr('marker-end', 'url(#arrow)')
            .attr('stroke-dasharray', '5,5')
            .attr('d', 'M' + this.pts.start.x + ' ' + this.pts.start.y +
                ' L' + this.pts.end.x + ' ' + this.pts.end.y)
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
    constructor(attribs, n2Groups, nodeSize, markerSize = null) {
        super(attribs, n2Groups, nodeSize, markerSize);
        this.label.ref = d3.select("div#left.offgrid");
        this.attribs.direction = 'up';
        this.color = N2Style.color.redArrow;

        if (this._addToLabel()) {
            this._computePts();
            this.draw();
        }
    }

    /** Override id to handle an off-screen node */
    get id() {
        return 'arrow-' + this.attribs.offscreenId + '-to-' +
            this.attribs.cell.tgtId + '-' + this.attribs.color.replace(/#/, '')
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
        this.label.pts.left = this.pts.start.x + this.bgRect.left -
            this.label.rect.width / 2;
        this.label.pts.top = this.bgRect.bottom + 2;
    }
}

/**
 * Draw arrow on the top half of the N2 matrix coming down into the
 * onscreen target cell from the offscreen source.
 */
class N2OffGridDownArrow extends N2OffGridArrow {
    constructor(attribs, n2Groups, nodeSize, markerSize = null) {
        super(attribs, n2Groups, nodeSize, markerSize);

        this.label.ref = d3.select("div#bottom.offgrid");
        this.attribs.direction = 'down';
        this.color = N2Style.color.redArrow;

        if (this._addToLabel()) {
            this._computePts();
            this.draw();
        }
    }

    /** Override id to handle an off-screen node */
    get id() {
        return 'arrow-' + this.attribs.offscreenId + '-to-' +
            this.attribs.cell.tgtId + '-' + this.attribs.color.replace(/#/, '');
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
        this.label.pts.left = this.pts.start.x + this.bgRect.left -
            this.label.rect.width / 2;
        this.label.pts.top = this.bgRect.top - this.label.rect.height - 2;
    }
}

/**
 * Draw arrow on the bottom half of N2 matrix going left from the
 * onscreen source cell to the offscreen target.
 */
class N2OffGridLeftArrow extends N2OffGridArrow {
    constructor(attribs, n2Groups, nodeSize, markerSize = null) {
        super(attribs, n2Groups, nodeSize, markerSize);

        this.label.ref = d3.select("div#bottom.offgrid");
        this.attribs.direction = 'left';
        this.color = N2Style.color.greenArrow;

        if (this._addToLabel()) {
            this._computePts();
            this.draw();
        }
    }

    /** Override id to handle an off-screen node */
    get id() {
        return 'arrow-' + this.attribs.cell.srcId + '-to-' +
            this.attribs.offscreenId + '-' + this.attribs.color.replace(/#/, '');
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
        this.label.pts.left = this.pts.end.x + this.bgRect.left -
            this.label.rect.width * 0.667;
        this.label.pts.top = this.pts.end.y + this.bgRect.top +
            this.label.rect.height / 2;
    }
}

/**
 * Draw arrow on the top half of the N2 matrix going right away from
 * the onscreen source cell to the offscreen target.
 */
class N2OffGridRightArrow extends N2OffGridArrow {
    constructor(attribs, n2Groups, nodeSize, markerSize = null) {
        super(attribs, n2Groups, nodeSize, markerSize);

        this.label.ref = d3.select("div#right.offgrid");
        this.attribs.direction = 'right';
        this.color = N2Style.color.greenArrow;

        if (this._addToLabel()) {
            this._computePts();
            this.draw();
        }
    }

    /** Override id to handle an off-screen node */
    get id() {
        return 'arrow-' + this.attribs.cell.srcId + '-to-' +
            this.attribs.offscreenId + '-' + this.attribs.color.replace(/#/, '');
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
        this.label.pts.left = this.pts.end.x + this.bgRect.left -
            this.label.rect.width / 3;
        this.label.pts.top = this.pts.end.y + this.bgRect.top +
            this.label.rect.height / 2;
    }
}

/**
 * Maintain a set of arrows, usually for pinning/unpinning.
 * @typedef N2ArrowCache
 * @property {Object} arrows Every individual arrow keyed by it's ID
 * @property {Object} cells  Arrows associated with cells, keyed by IDs
 */
class N2ArrowCache {
    /**
     * Initialize with empty caches.
     */
    constructor() {
        // All arrows will be referenced somewhere in both objects.
        this.arrows = {};
        this.cells = {};
    }

    hasArrow(arrowId) {
        return exists(this.arrows[arrowId]);
    }

    hasCell(cellId) {
        return exists(this.cells[cellId]);
    }

    /**
     * Add an individual arrow to the cache.
     * @param {String} cellId The ID of the cell associated with the arrow.
     * @param {N2Arrow} arrow The arrow object to cache.
     */
    add(cellId, arrow) {
        if (this.hasArrow(arrow.id)) {
            console.warn(`Not adding arrow ${arrow.id} to cache since it already exists.`)
        }
        else {
            this.arrows[arrow.id] = arrow;
            if (!this.cells[cellId]) this.cells[cellId] = {};
            this.cells[cellId][arrow.id] = arrow;
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
     * @param {String} cellId The D of the cell associated with the arrow.
     * @param {String} arrowId The ID of the single arrow to remove.
     */
    removeArrowFromCache(cellId, arrowId) {
        if (!this.hasArrow(arrowId)) {
            console.warn(`removeArrowFromCache: Arrow ${arrowId} doesn't exist.`)
        }
        else {
            delete this.cells[cellId][arrowId];
            delete this.arrows[arrowId];
        }
    }

    /**
     * Remove all arrow elements associated with the cell from the screen.
     * @param {String} cellId The id of the cell to remove.
     */
    removeCellFromScreen(cellId) {
        if (!this.hasCell(cellId)) {
            console.warn(`removeCellFromScreen: Cell ${cellId} doesn't exist in cache.`)
        }
        else {
            for (const arrowId in this.cells[cellId]) {
                this.removeArrowFromScreen(arrowId);
            }
        }
    }

    /**
     * Remove all arrows associated with the cell and the cell itself
     * from the cache.
     * @param {String} cellId The id of the cell to remove.
     */
    removeCellFromCache(cellId) {
        if (!this.hasCell(cellId)) {
            console.warn(`removeCellFromCache: Cell ${cellId} doesn't exist in cache.`)
        }
        else {
            for (const arrowId in this.cells[cellId]) {
                this.removeArrowFromCache(cellId, arrowId);
            }
            delete this.cells[cellId];
        }
    }

    /**
     * Remove everything we're tracking from screen and cache.
     */
    removeAll() {
        for (const cellId in this.cells) {
            this.removeCellFromScreen(cellId);
            this.removeCellFromCache(cellId);
        }
    }

    /**
     * Move an arrow from one cache to another and change its CSS classes.
     * @param {String} cellId The ID of the associated cell.
     * @param {String} arrowId The ID of the single arrow.
     * @param {N2ArrowCache} oldCache The cache to move the arrow from.
     * @param {String} newClassName The CSS class to add to HTML elements.
     * @param {String} oldClassName The CSS class to remove from HTML elements.
     */
    migrateArrow(cellId, arrowId, oldCache, newClassName, oldClassName) {
        const arrow = oldCache.arrows[arrowId];
        this.add(cellId, arrow);
        oldCache.removeArrowFromCache(cellId, arrowId);

        debugInfo(`migrateArrow(): Moving ${arrowId} from class ${oldClassName} to ${newClassName}`);

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
        if (!oldCache.hasCell(cellId)) {
            console.warn(`Cell ${cellId} doesn't exist in old cache, can't migrate arrows.`)
        }
        else {
            for (const arrowId in oldCache.cells[cellId]) {
                this.migrateArrow(cellId, arrowId, oldCache, newClassName, oldClassName);
            }
            delete oldCache.cells[cellId];
        }
    }

    /**
     * Iterate through all tracked cells. If the cell is displayed in
     * the matrix, draw its arrows, otherwise hide them.
     * @param {N2Matrix} matrix The matrix to operate with.
     */
    transitionArrows(matrix) {
        for (const cellId in this.cells) {
            const cell = matrix.findCellById(cellId);
            if (cell) {
                matrix.drawConnectionArrows(cell, true);
            }
            else {
                this.removeCellFromScreen(cellId);
            }
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
     * already exist in one of the caches, in which case it's not added again.
     * @param {String} cellId The ID of the associated N2MatrixCell.
     * @param {Object} attribs Values to pass to the N2Arrow constructor.
     * @returns {N2BentArrow} The newly created arrow object.
     */
    addFullArrow(cellId, attribs) {
        attribs.width = this.lineWidth;
        const newArrow = new N2BentArrow(attribs, this.n2Groups, this.nodeSize);
        if (!this.arrowExists(newArrow.id)) {
            this.hoverArrows.add(cellId, newArrow);
        }

        return newArrow;
    }

    /**
     * Create a new N2OffGridArrow-derived object. This may replace existing
     * elements on the screen with new dimensions and colors. However, the
     * arrow may already exist in one of the caches, in which case it's not
     * added again.
     * @param {String} cellId The ID of the associated N2MatrixCell.
     * @param {String} side Wether the arrow is in the top or bottom.
     * @param {String} dir Wether the arrow is incoming or outgoing.
     * @param {Object} attribs Values to pass to the N2Arrow constructor.
     * @returns {N2BentArrow} The newly created arrow object.
     */
    addOffGridArrow(cellId, side, dir, attribs) {
        attribs.width = this.lineWidth;
        const newArrow = new (this.arrowDirClasses[side][dir])(attribs,
            this.n2Groups, this.nodeSize);

        if (!this.hoverArrows.hasArrow(newArrow.id)) {
            this.hoverArrows.add(cellId, newArrow);
        }

        return newArrow;
    }

    /**
     * Draw all the visible arrows in the pinned arrow cache.
     * @param {N2Matrix} matrix The matrix to operate with.
     */
    transition(matrix) {
        this.pinnedArrows.transitionArrows(matrix);
    }

    /**
     * If arrows are hovering, then pin them, and vice versa.
     * @param {String} cellId The ID of the N2MatrixCell to operate on.
     */
    togglePin(cellId) {
        const cellClassName = "n2_hover_elements_" + cellId;
        if (this.pinnedArrows.hasCell(cellId)) { // Arrows already pinned
            this.hoverArrows.migrateCell(cellId, this.pinnedArrows,
                'n2_hover_elements', cellClassName);
        }
        else if (this.hoverArrows.hasCell(cellId)) { // Arrows just "hovered"
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
        debugInfo(`removeAllHovered(): Removed: ${removedArrowIds.length} arrows`)
        for (const arrowId of removedArrowIds) {
            if (this.pinnedArrows.hasArrow(arrowId)) {
                debugInfo(`removeAllHovered(): Redrawing pinned arrow ${arrowId}.`)
                this.pinnedArrows.arrows[arrowId].draw();
                //TODO: The pinned N2Arrow now is pointing to a non-existant g
            }
        }
    }
}