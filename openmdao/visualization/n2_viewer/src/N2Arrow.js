/**
 * Base class for all types of N2 arrows.
 */
class N2Arrow {
    /**
     * Setup and draw the arrow.
     * @param {Object} attribs
     * @param {Object} attribs.start Coordinates of the starting node
     * @param {number} attribs.start.row
     * @param {number} attribs.start.col
     * @param {Object} attribs.end Coordinates of the ending node, which the arrow points to
     * @param {number} attribs.end.row
     * @param {number} attribs.end.col
     * @param {string} attribs.color Color of the line/circle (arrow is black)
     * @param {number} attribs.width Width of the line
     * @param {Object} n2Groups References to <g> SVG elements.
     * @param {Object} nodeSize The dimensions of each cell in the matrix.
     */
    constructor(attribs, n2Groups, nodeSize) {
        this.color = attribs.color;
        this.width = attribs.width;
        this.arrowsGrp = n2Groups.arrows;
        this.dotsGrp = n2Groups.dots;
        this.elementsGrp = n2Groups.elements;
        this.nodeSize = nodeSize;
        this.attribs = attribs;
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

        let offsetX = (this.start.col < this.end.col) ? offsetAbsX : -offsetAbsX; // Left-to-Right : Right-to-Left
        this.pts.start.x = this.nodeSize.width * this.start.col + this.nodeSize.width * .5 + offsetX;
        this.pts.mid.x = this.nodeSize.width * this.middle.col + this.nodeSize.width * .5;
        this.pts.end.x = this.nodeSize.width * this.end.col + this.nodeSize.width * .5;

        let offsetY = (this.start.row < this.end.row) ? -offsetAbsY : offsetAbsY; // Down : Up
        this.pts.start.y = this.nodeSize.height * this.start.row + this.nodeSize.height * .5;
        this.pts.mid.y = this.nodeSize.height * this.middle.row + this.nodeSize.height * .5;
        this.pts.end.y = this.nodeSize.height * this.end.row + this.nodeSize.height * .5 + offsetY;
    }

    /**
     * Use SVG to draw the line segments, add a circle at the "middle",
     * and an arrow at the end-point.
     */
    draw() {
        this.path = this.arrowsGrp.insert("path")
            .attr("class", "n2_hover_elements")
            .attr("d", "M" + this.pts.start.x + " " + this.pts.start.y +
                " L" + this.pts.mid.x + " " + this.pts.mid.y +
                " L" + this.pts.end.x + " " + this.pts.end.y)
            .attr("fill", "none")
            .style("stroke-width", this.width)
            .style("stroke", this.color);

        this.dotsGrp.append("circle")
            .attr("class", "n2_hover_elements")
            .attr("cx", this.pts.mid.x)
            .attr("cy", this.pts.mid.y)
            .attr("r", this.width * 1.0)
            .style("stroke-width", 0)
            .style("fill-opacity", 1)
            .style("fill", "black");

        this.dotsGrp.append("circle")
            .attr("class", "n2_hover_elements")
            .attr("cx", this.pts.mid.x)
            .attr("cy", this.pts.mid.y)
            .attr("r", this.width * 1.0)
            .style("stroke-width", 0)
            .style("fill-opacity", .75)
            .style("fill", this.color);

        this.path.attr("marker-end", "url(#arrow)");
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

    /**
     * Put the SVG arrow on the screen and position the tooltip.
     */
    draw() {
        debugInfo('Adding offscreen ' + this.attribs.direction +
            ' arrow connected to ' + this.label.text);

        this.path = this.arrowsGrp.insert('path')
            .attr('class', 'n2_hover_elements')
            .attr('stroke-dasharray', '5,5')
            .attr('d', 'M' + this.pts.start.x + ' ' + this.pts.start.y +
                ' L' + this.pts.end.x + ' ' + this.pts.end.y)
            .attr('fill', 'none')
            .style('stroke-width', this.width)
            .style('stroke', this.color);

        this.path.attr('marker-end', 'url(#arrow)');

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

N2OffGridArrow.arrowDir = {
    'top': {
        'incoming': N2OffGridDownArrow,
        'outgoing': N2OffGridRightArrow
    },
    'bottom': {
        'incoming': N2OffGridUpArrow,
        'outgoing': N2OffGridLeftArrow
    }
};