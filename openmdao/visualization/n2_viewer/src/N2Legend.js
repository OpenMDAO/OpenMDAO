class N2Legend {
    constructor(title = "LEGEND") {
        this.numColumns = 3;
        this.elementSize = 30;
        this.xOffset = 10;
        this.columnWidth = 250;
        this.title = title;

        this.dims = {
            'width': this.columnWidth * this.numColumns,
            'height': 500,
            'x': this.elementSize * .5,
            'y': this.elementSize * .5
        }

        this.div = d3.select("div#legend")
            .style("width", this.dims.width + "px")
            .style("height", this.dims.height + "px");

        this.colors = [
            { 'name': "Group", 'color': N2Style.color.group },
            { 'name': "Component", 'color': N2Style.color.component },
            { 'name': "Input", 'color': N2Style.color.param },
            { 'name': "Unconnected Input", 'color': N2Style.color.unconnectedParam },
            { 'name': "Output Explicit", 'color': N2Style.color.unknownExplicit },
            { 'name': "Output Implicit", 'color': N2Style.color.unknownImplicit },
            { 'name': "Collapsed", 'color': N2Style.color.collapsed },
            { 'name': "Connection", 'color': N2Style.color.connection }
        ];
    }

    _createElementBorder(g) {
        g.append("rect")
            .attr("x", -this.dims.x)
            .attr("y", -this.dims.y)
            .attr("width", this.elementSize)
            .attr("height", this.elementSize)
            .style("stroke-width", 2)
            .style("stroke", "white")
            .style("fill", "none");
    }

    _createText(g, text) {
        g.append("svg:text")
            .attr("x", u + 5)
            .attr("y", 0)
            .attr("dy", ".35em")
            .attr("font-size", 20)
            .text(text)
            .style("fill", "black");
    }

    _drawLegendColor(g, color, justUpdate) {
        let shape = justUpdate ? g.select(".colorMid") :
            g.append("rect").attr("class", "colorMid").style("fill", color);

        return shape.attr("x", -this.dims.x)
            .attr("y", -this.dims.y)
            .attr("width", this.dims.x * 2)
            .attr("height", this.dims.y * 2)
            .style("stroke-width", 0)
            .style("fill-opacity", 1);
    }

    _addMainTitle() {
        let el = this.svg.append("g")
            .attr("transform", "translate(" + (legendWidth * .5) + ",15)");

        el.append("svg:text")
            .attr("text-anchor", "middle")
            .attr("dy", ".35em")
            .attr("font-size", 30)
            .attr("text-decoration", "underline")
            .text(this.title)
            .style("fill", "black");
    }

    _addColumnTitles(showLinearSolverNames) {
        let titles = [
            "Systems & Variables",
            "N^2 Symbols",
            showLinearSolverNames ? " Linear Solvers" : "Nonlinear Solvers"
        ];

        for (let i = 0; i < titles.length; ++i) {
            let el = this.svg.append("g")
                .attr("transform", "translate(" +
                    (this.columnWidth * i + this.xOffset) + ",60)");

            el.append("svg:text")
                .attr("dy", ".35em")
                .attr("font-size", 24)
                .attr("text-decoration", "underline")
                .text(titles[i])
                .style("fill", "black");
        }
    }

    _drawColors() {
        let i = 0;

        for (let color of this.colors) {
            let el = this.svg.append("g")
                .attr("transform", "translate(" +
                    (this.xOffset + this.dims.x) + "," +
                    (80 + i + this.dims.y) + ")");
            this.drawLegendColor(el, color.color, false);
            this.createText(el, color.name);
            i += 40;
        }
    }

    _drawSymbols() {
        var text = ["Scalar", "Vector", "Collapsed variables"];
        var colors = [N2Style.color.unknownExplicit, N2Style.color.unknownExplicit, N2Style.color.unknownExplicit];
        var shapeFunctions = [DrawScalar, DrawVector, DrawGroup];
        for (var i = 0; i < text.length; ++i) {
            var el = svg_legend.append("g").attr("transform", "translate(" + (columnWidth * 1 + xOffset + u) + "," + (80 + 40 * i + v) + ")");
            shapeFunctions[i](el, u, v, colors[i], false);
            CreateElementBorder(el);
            CreateText(el, text[i]);
        }
    }

    show(showLinearSolverNames) {
        // Clear anything that might already be in there
        this.div.node().innerHTML = "";

        this.svg = this.div.append("svg:svg")
            .attr("id", "legendSVG")
            .attr("width", this.dims.width)
            .attr("height", this.dims.height);

        this.svg.append("rect")
            .attr("class", "background")
            .attr("width", this.dims.width)
            .attr("height", this.dims.height);

        this._addMainTitle();
        this._addColumnTitles(showLinearSolverNames);
        this._drawColors();

    }
}