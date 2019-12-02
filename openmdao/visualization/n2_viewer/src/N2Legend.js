/**
 * Draw a box under the diagram describing each of the element types.
 * @typedef N2Legend
 * @property {String} title The label to put at the top of the legend.
 * @property {Boolean} shown Whether the legend is currently drawn or not.
 */
class N2Legend {
    /**
     * Initializes the legend object, but doesn't draw it yet.
     * @param {String} [title = "LEGEND"] The label at the top of the legend box.
     */
    constructor(title = "LEGEND") {
        this.title = title;
        this.shown = false; // Not shown until show() is called

        this._elementSize = 30; // The base width & height of each legend item
        this._xOffset = 10; // Left-margin width
        this._columnWidth = 250; // The width of each column in pixels

        // Overall legend sizes
        let numColumns = 3;
        this._width = this._columnWidth * numColumns;
        this._height = 500;

        // Individual element sizes
        this._elemDims = {
            'x': this._elementSize * .5,
            'y': this._elementSize * .5,
            'topLeft': { 'x': this._elementSize * -.5, 'y': this._elementSize * -.5 },
            'bottomRight': { 'x': this._elementSize * .5, 'y': this._elementSize * .5 },
            'size': { 'width': this._elementSize, 'height': this._elementSize }
        }

        // The HTML parent element to draw in.
        this._div = d3.select("div#legend")
            .style("width", this._width + "px")
            .style("height", this._height + "px");

        // Colors of each variable/system type
        this._colors = [
            { 'name': "Group", 'color': N2Style.color.group },
            { 'name': "Component", 'color': N2Style.color.component },
            { 'name': "Input", 'color': N2Style.color.param },
            { 'name': "Unconnected Input", 'color': N2Style.color.unconnectedParam },
            { 'name': "Output Explicit", 'color': N2Style.color.unknownExplicit },
            { 'name': "Output Implicit", 'color': N2Style.color.unknownImplicit },
            { 'name': "Collapsed", 'color': N2Style.color.collapsed },
            { 'name': "Connection", 'color': N2Style.color.connection }
        ];

        // Name and renderer object for each symbol type
        let symColor = N2Style.color.unknownExplicit;
        this._symbols = [
            { 'name': 'Scalar', 'cell': new N2ScalarCell(symColor) },
            { 'name': 'Vector', 'cell': new N2VectorCell(symColor) },
            { 'name': 'Collapsed variables', 'cell': new N2GroupCell(symColor) }
        ];
    }

    /**
     * Create a square, white border for a symbol.
     * @param {Array} svgGrp D3 selection of a previously-created SVG <g> element.
     */
    _createElementBorder(svgGrp) {
        svgGrp.append("rect")
            .attr("x", -this._elemDims.x)
            .attr("y", -this._elemDims.y)
            .attr("width", this._elemDims.size.width)
            .attr("height", this._elemDims.size.height)
            .style("stroke-width", 2)
            .style("stroke", "white")
            .style("fill", "none");
    }

    /**
     * Add SVG text label to the legend.
     * @param {Array} svgGrp D3 selection of a previously-created SVG <g> element.
     * @param {String} label The label to insert.
     */
    _createText(svgGrp, label) {
        svgGrp.append("svg:text")
            .attr("x", this._elemDims.x + 5)
            .attr("y", 0)
            .attr("dy", ".35em")
            .attr("font-size", 20)
            .text(label)
            .style("fill", "black");
    }

    /**
     * Add a colored rectangle to the legend.
     * @param {Array} svgGrp D3 selection of a previously-created SVG <g> element.
     * @param {String} color The value of the fill color.
     */
    _drawLegendColor(svgGrp, color) {
        let shape = svgGrp.append("rect")
            .attr("class", "colorMid")
            .style("fill", color);

        return shape.attr("x", -this._elemDims.x)
            .attr("y", -this._elemDims.y)
            .attr("width", this._elemDims.x * 2)
            .attr("height", this._elemDims.y * 2)
            .style("stroke-width", 0)
            .style("fill-opacity", 1);
    }

    /** Create an SVG group and text element containing the main title. */
    _addMainTitle() {
        let titleGrp = this.svg.append("g")
            .attr("transform", "translate(" + (this._width * .5) + ",15)");

        titleGrp.append("svg:text")
            .attr("text-anchor", "middle")
            .attr("dy", ".35em")
            .attr("font-size", 30)
            .attr("text-decoration", "underline")
            .text(this.title)
            .style("fill", "black");
    }

    /**
     * For each column, create an SVG group and text element with its title.
     * @param {Boolean} showLinearSolverNames Whether linear or non-linear solvers are viewed.
     */
    _addColumnTitles(showLinearSolverNames) {
        let titles = [
            "Systems & Variables",
            "N^2 Symbols",
            showLinearSolverNames ? " Linear Solvers" : "Nonlinear Solvers"
        ];

        for (let i = 0; i < titles.length; ++i) {
            let columnTitleGrp = this.svg.append("g")
                .attr("transform", "translate(" +
                    (this._columnWidth * i + this._xOffset) + ",60)");

            columnTitleGrp.append("svg:text")
                .attr("dy", ".35em")
                .attr("font-size", 24)
                .attr("text-decoration", "underline")
                .text(titles[i])
                .style("fill", "black");
        }
    }

    /** Draw a colored rectangle for each system/variable type in the first column. */
    _drawColors() {
        let i = 0;

        for (let color of this._colors) {
            let el = this.svg.append("g")
                .attr("transform", "translate(" +
                    (this._xOffset + this._elemDims.x) + "," +
                    (80 + i + this._elemDims.y) + ")");
            this._drawLegendColor(el, color.color, false);
            this._createText(el, color.name);
            i += 40;
        }
    }

    /** Render a shape for each symbol type in the second column. */
    _drawSymbols() {
        let yOffset = 0;

        for (let sym of this._symbols) {
            let svgGrp = this.svg.append("g")
                .attr("transform", "translate(" + (this._columnWidth + this._xOffset
                    + this._elemDims.x) + "," + (80 + yOffset + this._elemDims.y) + ")");

            sym.cell.render(svgGrp.node(), this._elemDims, true);

            this._createElementBorder(svgGrp);
            this._createText(svgGrp, sym.name);
            yOffset += 40;
        }
    }

    /**
     * Render a colored rectangle for each solver in the third column, depending
     * on whether linear or non-linear ones are selected.
     * @param {Boolean} showLinearSolverNames Determines solver name type displayed.
     * @param {Object} solverStyles Solver names, types, and styles including color.
     */
    _drawSolverColors(showLinearSolverNames, solverStyles) {
        let yOffset = 0;
        for (let solverName in solverStyles) {
            let solver = solverStyles[solverName];
            if ((solver.type == 'linear' && showLinearSolverNames) ||
                (solver.type == 'nonLinear' && !showLinearSolverNames)) {
                let el = this.svg.append("g")
                    .attr("transform", "translate(" + (this._columnWidth * 2 + this._xOffset +
                        this._elemDims.x) + "," + (80 + yOffset + this._elemDims.y) + ")");
                this._drawLegendColor(el, solver.style.fill, false);
                this._createText(el, solverName);
                yOffset += 40;
            }
        }
    }

    /** Remove all legend SVG elements. */
    _clear() {
        this._div.node().innerHTML = "";
    }

    /** Remove all legend SVG elements and mark as not shown. */
    hide() {
        this._clear();
        this.shown = false;
    }

    /**
     * Remove any existing SVG elements, create a new SVG element with
     * a background rect, and add all of the other pieces.
     * @param {Boolean} showLinearSolverNames Determines solver name type displayed.
     * @param {Object} solverStyles Solver names, types, and styles including color.
     */
    show(showLinearSolverNames, solverStyles) {
        // Clear anything that might already be in there
        this._clear();
        this.shown = true;

        this.svg = this._div.append("svg:svg")
            .attr("id", "legendSVG")
            .attr("width", this._width)
            .attr("height", this._height);

        this.svg.append("rect")
            .attr("class", "background")
            .attr("width", this._width)
            .attr("height", this._height);

        this._addMainTitle();
        this._addColumnTitles(showLinearSolverNames);
        this._drawColors();
        this._drawSymbols();
        this._drawSolverColors(showLinearSolverNames, solverStyles);
    }

    /**
     * If legend is shown, hide it; if it's hidden, show it.
     * @param {Boolean} showLinearSolverNames Determines solver name type displayed.
     * @param {Object} solverStyles Solver names, types, and styles including color.
     */
    toggle(showLinearSolverNames, solverStyles) {
        if (this.shown) this.hide();
        else this.show(showLinearSolverNames, solverStyles);
    }
}