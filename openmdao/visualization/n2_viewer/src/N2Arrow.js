/**
 * Draws a two-segment arrow, using provided beginning and end node locations.
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
     */
    constructor(attribs) {
        this.start = attribs.start;
        this.end = attribs.end;
        this.color = attribs.color;
        this.width = attribs.width;

        this.middle = { row: -1, col: -1 };

        this.startPt = { x: -1, y: -1 };
        this.midPt = { x: -1, y: -1 };
        this.endPt = { x: -1, y: -1 };

        this.computePts();
        this.draw();
    }

    /**
     * Determine the location of the "middle" node, where the bend is. Calculate
     * the points corresponding to the associated col, row in the matrix.
     */
    computePts() {
        this.middle.row = this.start.row;
        this.middle.col = this.end.col;
        let offsetAbsX = n2Dx * .125;
        let offsetAbsY = n2Dy * .125 + 3; // + to account for the arrow size

        let offsetX = (this.start.col < this.end.col) ? offsetAbsX : -offsetAbsX; // Left-to-Right : Right-to-Left
        this.startPt.x = n2Dx * this.start.col + n2Dx * .5 + offsetX;
        this.midPt.x = n2Dx * this.middle.col + n2Dx * .5;
        this.endPt.x = n2Dx * this.end.col + n2Dx * .5;

        let offsetY = (this.start.row < this.end.row) ? -offsetAbsY : offsetAbsY; // Down : Up
        this.startPt.y = n2Dy * this.start.row + n2Dy * .5;
        this.midPt.y = n2Dy * this.middle.row + n2Dy * .5;
        this.endPt.y = n2Dy * this.end.row + n2Dy * .5 + offsetY;
    }

    /**
     * Use SVG to draw the line segments, add a circle at the "middle",
     * and an arrow at the end-point.
     */
    draw() {
        this.path = n2ArrowsGroup.insert("path")
            .attr("class", "n2_hover_elements")
            .attr("d", "M" + this.startPt.x + " " + this.startPt.y +
                " L" + this.midPt.x + " " + this.midPt.y +
                " L" + this.endPt.x + " " + this.endPt.y)
            .attr("fill", "none")
            .style("stroke-width", this.width)
            .style("stroke", this.color);

        n2DotsGroup.append("circle")
            .attr("class", "n2_hover_elements")
            .attr("cx", this.midPt.x)
            .attr("cy", this.midPt.y)
            .attr("r", this.width * 1.0)
            .style("stroke-width", 0)
            .style("fill-opacity", 1)
            .style("fill", "black");

        n2DotsGroup.append("circle")
            .attr("class", "n2_hover_elements")
            .attr("cx", this.midPt.x)
            .attr("cy", this.midPt.y)
            .attr("r", this.width * 1.0)
            .style("stroke-width", 0)
            .style("fill-opacity", .75)
            .style("fill", this.color);

        this.path.attr("marker-end", "url(#arrow)");
    }
}