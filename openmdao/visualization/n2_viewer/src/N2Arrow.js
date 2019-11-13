
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

class N2OffGridArrow extends N2Arrow {
    constructor(attribs, n2Groups, nodeSize, markerSize = null) {
        super(attribs, n2Groups, nodeSize, markerSize);

        this.cell = attribs.cell;

        this.pts = {
            'start': { 'x': -1, 'y': -1 },
            'end': { 'x': -1, 'y': -1 }
        }

        this._computePts();
        this.draw();
    }

    /**
     * Use the start coordinates and the direction of the arrow to determine the starting
     * coordinates and location of the arrowhead.
     */
    _computePts() {
        let offsetX = this.offsetAbsX;
        let offsetY = this.offsetAbsY;

        switch (this.attribs.direction) {
            case 'up':
                this.pts.start.x = this.pts.end.x =
                    this.nodeSize.width * this.cell.col + this.nodeSize.width * .5;
                this.pts.start.y = this.attribs.matrixSize * this.nodeSize.height +
                    this.nodeSize.height * .5 + offsetY;
                this.pts.end.y = this.nodeSize.height * this.cell.row +
                    this.nodeSize.height;
                break;
            case 'down':
                this.pts.start.x = this.pts.end.x =
                    this.nodeSize.width * this.cell.col + this.nodeSize.width * .5;
                this.pts.start.y = this.nodeSize.height * -.5 - offsetY;
                this.pts.end.y = this.nodeSize.height * this.cell.row;
                break;
            case 'left':
                this.pts.start.y = this.pts.end.y =
                    this.nodeSize.height * this.cell.row + this.nodeSize.height * .5;

                this.pts.start.x = this.cell.col * this.nodeSize.width +
                    this.nodeSize.width * .5 - offsetX;
                this.pts.end.x = offsetX;
                break;
            case 'right':
                this.pts.start.y = this.pts.end.y =
                    this.nodeSize.height * this.cell.row + this.nodeSize.height * .5;
                this.pts.start.x = this.nodeSize.width * this.cell.col +
                    this.nodeSize.width * .5 + offsetX;
                this.pts.end.x = this.attribs.matrixSize * this.nodeSize.width - offsetX;
                break;
            default:
                throw ('N2OffGridArrow._computePts(): Unrecognized direction "' +
                    this.attribs.direction + '" used.')

        }
    }

    draw() {
        this.path = this.arrowsGrp.insert("path")
            .attr("class", "n2_hover_elements")
            .attr("d", "M" + this.pts.start.x + " " + this.pts.start.y +
                " L" + this.pts.end.x + " " + this.pts.end.y)
            .attr("fill", "none")
            .style("stroke-width", this.width)
            .style("stroke", this.color);

        this.path.attr("marker-end", "url(#offgridArrow)");
    }
}