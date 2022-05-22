/**
 * Base class for all types of arrows.
 * @typedef Arrow
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
class Arrow {
    /**
     * Determine whether arrow elements already exist, and create them if not.
     * @param {Object} attribs All of the specific arrow properties.
     * @param {Object} diagGroups References to <g> SVG elements.
     * @param {Object} nodeSize The dimensions of each cell in the matrix.
     */
    constructor(attribs, diagGroups, nodeSize) {
        this.color = attribs.color;
        this.width = attribs.width;
        this.nodeSize = nodeSize;
        this.attribs = attribs;
        this._genPath = this._angledPath;

        /*
         * Generate a CSS id for the arrow. This is based on the ids of the
         * TreeNodes it points to rather than the cells, so if it's pinned
         * and the matrix is redrawn, we know whether to transition it. The color
         * is included because there can be identical arrows of different colors
         * stacked on each other.
         */
        this.id = 'arrow-' + attribs.start.id + '-to-'
            + attribs.end.id + '-' + this.color.replace(/#/, '');

        const existingArrow = d3.select('g#' + this.id);
        if (existingArrow.empty()) {
            debugInfo(`Arrow(): Creating new ${this.id}`);
            this.doTransition = false;
            this.cssClass = 'n2_hover_elements';
            this.group = diagGroups.arrows.append('g')
                .attr('id', this.id)
                .attr('class', this.cssClass)
        }
        else {
            this.doTransition = true;
            this.group = existingArrow;
            const cssClasses = String(this.group.attr('class')).split(' ');
            this.cssClass = cssClasses.find(o => o.match(/n2_hover_elements.*/));
            debugInfo(`Arrow(): Using existing ${this.id} with class ${this.cssClass}`);
        }
    }

    connectsToCell(cellId) {
        return (this.attribs.start.id == cellId ||
            this.attribs.end.id == cellId);
    }

    get offsetAbsX() {
        return this.nodeSize.width * Arrow.cellOverlap;
    }

    get offsetAbsY() {
        return this.nodeSize.height * Arrow.cellOverlap + 3;  // + to account for the arrow size
    }
}

Arrow.cellOverlap = .125;

/**
 * Draws a two-segment arrow, using provided beginning and end node locations.
 */
class BentArrow extends Arrow {
    constructor(attribs, diagGroups, nodeSize) {
        super(attribs, diagGroups, nodeSize);

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
            this.path = this.group.select('path').transition(getTransition());

            if (this.group.classed('off-grid-arrow')) {
                // The arrow was previously going offscreen but now is fully onscreen
                this._createDots();
            }
            else {
                this.bottomCircle = this.group.select('circle#bottom-circle').transition(getTransition());
                this.topCircle = this.group.select('circle#top-circle').transition(getTransition());
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
            .style("fill", Style.color.connection);

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
class OffGridArrow extends Arrow {
    constructor(attribs, diagGroups, nodeSize) {
        super(attribs, diagGroups, nodeSize);

        this.cell = attribs.cell;
        this.direction = attribs.direction;
        this.cellId = attribs.cellId;

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

        /* Get the bounding rect for the matrix background rect (use rect
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

            this.path = this.group.select('path').transition(getTransition());
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
 * Draw an arrow on the bottom half of the matrix coming up into
 * the onscreen target cell from the offscreen source.
 */
class OffGridUpArrow extends OffGridArrow {
    constructor(attribs, diagGroups, nodeSize) {
        super(Object.assign(attribs, {
            'start': { 'id': attribs.offscreenId },
            'end': { 'id': attribs.cell.tgtId },
            'direction': 'up',
            'cellId': attribs.cellId,
            'color': attribs.color ? attribs.color : Style.color.inputArrow
        }), diagGroups, nodeSize);

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
 * Draw arrow on the top half of the matrix coming down into the
 * onscreen target cell from the offscreen source.
 */
class OffGridDownArrow extends OffGridArrow {
    constructor(attribs, diagGroups, nodeSize) {
        super(Object.assign(attribs, {
            'start': { 'id': attribs.offscreenId },
            'end': { 'id': attribs.cell.tgtId },
            'direction': 'down',
            'cellId': attribs.cellId,
            'color': attribs.color ? attribs.color : Style.color.inputArrow
        }), diagGroups, nodeSize);

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
 * Draw arrow on the bottom half of matrix going left from the
 * onscreen source cell to the offscreen target.
 */
class OffGridLeftArrow extends OffGridArrow {
    constructor(attribs, diagGroups, nodeSize) {
        super(Object.assign(attribs, {
            'start': { 'id': attribs.cell.srcId },
            'end': { 'id': attribs.offscreenId },
            'direction': 'left',
            'cellId': attribs.cellId,
            'color': attribs.color ? attribs.color : Style.color.outputArrow
        }), diagGroups, nodeSize);

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
 * Draw arrow on the top half of the matrix going right away from
 * the onscreen source cell to the offscreen target.
 */
class OffGridRightArrow extends OffGridArrow {
    constructor(attribs, diagGroups, nodeSize) {
        super(Object.assign(attribs, {
            'start': { 'id': attribs.cell.srcId },
            'end': { 'id': attribs.offscreenId },
            'direction': 'right',
            'cellId': attribs.cellId,
            'color': attribs.color ? attribs.color : Style.color.outputArrow
        }), diagGroups, nodeSize);

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
