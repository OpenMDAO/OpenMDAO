class N2Arrow {
    constructor(attribs) {
        /* attribs:
            start: {col, row}
            end: {col, row}
            color
            width
            */
        this.start = attribs.start;
        this.end = attribs.end;
        this.color = attribs.color;
        this.width = attribs.width;

        this.middle = { };
        this.startPt = { };
        this.startPtOffset = {x: 0, y: 0};
        this.midPt = { };
        this.endPt = { };
        this.endPtOffset = {x: 0, y: 0};
        this.computePts();
        this.draw();
    }

    computePts() {
        // Two segments:
        if (this.start.col != this.end.col && this.start.row != this.end.row) {
            if (this.start.col < this.end.col) { // Left-to-Right
                this.middle.col = this.end.col;
                this.startPt.x = n2Dx * this.start.col + n2Dx * .5 + 6;
                this.midPt.x = n2Dx * this.middle.col + n2Dx * .5;
                this.endPt.x = n2Dx * this.end.col + n2Dx * .5;
            } 
            else { // Right-to-Left
                this.middle.col = this.end.col;
                this.startPt.x = n2Dx * this.start.col + n2Dx * .5 - 6;
                this.midPt.x = n2Dx * this.middle.col + n2Dx * .5;
                this.endPt.x = n2Dx * this.end.col + n2Dx * .5;
            }

            if (this.start.row < this.end.row ) { // Down
                this.middle.row = this.start.row;
                this.startPt.y = n2Dy * this.start.row + n2Dy * .5;
                this.midPt.y = n2Dy * this.middle.row + n2Dy * .5;
                this.endPt.y = n2Dy * this.end.row + n2Dy * .5 - 6;
            }
            else { // Up
                this.middle.row = this.start.row;
                this.startPt.y = n2Dy * this.start.row + n2Dy * .5;
                this.midPt.y = n2Dy * this.middle.row + n2Dy * .5;
                this.endPt.y = n2Dy * this.end.row + n2Dy * .5 + 6;
            }
        }
        else {
            console.log("Only one segment detected!")
        }
    }

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