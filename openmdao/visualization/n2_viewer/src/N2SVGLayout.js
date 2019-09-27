let N2SVGLayout_statics = {
    'showLinearSolverNames': true,
    'rightTextMarginPx': 8,
    'heightPx': 600,
    'fontSizePx': 11,
    'minColumnWidthPx': 5,
    'parentNodeWidthPx': 40
};

class N2SVGLayout {
    constructor(model, zoomedElement) {
        this.model = model;
        this.zoomedElement = zoomedElement;

        this.outputNamingType = "Absolute";
        this.svg = d3.select("#svgId");

        this.setupTextRenderer();
        this.updateTextWidths();
        this.updateSolverTextWidths();
        this.computeLeaves();
        this.computeColumnWidths();
        this.computeSolverColumnWidths();
        this.setColumnLocations();
    }

    /** Switch back and forth between showing the linear or non-linear solver names. 
     * @return {Boolean} The new value.
     */
    static toggleSolverNameType() {
        N2SVGLayout.showLinearSolverNames = !N2SVGLayout.showLinearSolverNames;
        return N2SVGLayout.showLinearSolverNames;
    }

    /** Create an off-screen area to render text */
    setupTextRenderer() {
        let textGroup = this.svg.append("svg:g").attr("class", "partition_group");
        let textSVG = textGroup.append("svg:text")
            .text("")
            .attr("x", -50); // Put text off screen to the left.

        this.textRenderer = {
            'group': textGroup,
            'textSvg': textSVG,
            'node': textSVG.node(),
            'widthCache': {}
        };
    }

    /** Insert text into an off-screen SVG text object to determine the width.
     * Cache the result so repeat calls with the same text can just do a lookup.
     * @param {string} textToFind Text to render or find in cache.
     * @return {number} The SVG-computed width of the rendered string.
     */
    getTextWidth(textToFind) {
        // Check cache first
        if (exists(this.textRenderer.widthCache[textToFind]))
            return this.textRenderer.widthCache[textToFind];

        // Not found, render and return new width.
        this.textRenderer.textSvg.text(textToFind);
        let width = this.textRenderer.node.getBoundingClientRect().width;

        this.textRenderer.widthCache[textToFind] = width;
        return width;
    }

    /** Determine the text associated with the element. Normally its name,
     * but can be changed if promoted or originally contained a colon.
     * @param {Object} element The item to operate on.
     * @return {string} The selected text.
     */
    getText(element) {
        let retVal = element.name;

        if (this.outputNamingType == "Promoted" &&
            element.type.match(/^(unknown|param|unconnected_param)$/) &&
            this.zoomedElement.propExists('promotions') &&
            this.zoomedElement.promotions.propExists(element.absPathName)) {
            retVal = this.zoomedElement.promotions[element.absPathName];
        }

        if (element.splitByColon && Array.isArray(element.children) &&
            element.children.length > 0) {
            retVal += ":";
        }

        return retVal;
    }

    /** Return a the name of the linear or non-linear solver depending
     * on the current setting.
     * @param {Object} element The item to get the solver text from.
     */
    getSolverText(element) {
        return N2SVGLayout.showLinearSolverNames ?
            element.linear_solver : element.nonlinear_solver;
    }

    /** Determine text widths for all descendents of the specified element.
     * @param {Object} [element = this.zoomedElement] Item to begin looking from.
     */
    updateTextWidths(element = this.zoomedElement) {
        if (element.varIsHidden) return;

        element.nameWidthPx = this.getTextWidth(this.GetText(element)) + 2 *
            N2SVGLayout.rightTextMarginPx;

        if (Array.isArray(element.children)) {
            for (var i = 0; i < element.children.length; ++i) {
                this.updateTextWidths(element.children[i]);
            }
        }
    }

    /** Determine text width for this and all decendents if element is a solver.
     * @param {Object} [element = this.zoomedElement] Item to begin looking from.
     */
    updateSolverTextWidths(element = this.zoomedElement) {
        if (element.type.match(/^(param|unconnected_param)$/) || element.varIsHidden) {
            return;
        }

        element.nameSolverWidthPx = this.getTextWidth(this.getSolverText(element)) + 2 *
            N2SVGLayout.rightTextMarginPx;

        if (Array.isArray(element.children)) {
            for (let i = 0; i < element.children.length; ++i) {
                this.updateSolverTextWidths(element.children[i]);
            }
        }
    }

    /** Recurse through the tree and add up the number of leaves that each
     * node has, based on their array of children.
     * @param {Object} [element = this.model.root] The starting node.
     */
    computeLeaves(element = this.model.root) {
        let self = this;

        if (element.varIsHidden) {
            element.numLeaves = 0;
        }
        else if (Array.isArray(element.children) && !element.isMinimized) {
            element.numLeaves = 0;
            element.children.forEach(function (child) {
                self.computeLeaves(child);
                element.numLeaves += child.numLeaves;
            })
        }
        else {
            element.numLeaves = 1;
        }

        element.numSolverLeaves = element.numLeaves;
    }

    /** For visible elements with children, choose a column width
     * large enough to accomodate the widest label in their column.
     * @param {Object} element The item to operate on.
     * @param {string} childrenProp Either 'children' or 'subsystem_children'.
     * @param {Object[]} colArr The array of column info.
     * @param {number[]} leafArr The array of leaf width info.
     * @param {string} widthProp Either 'nameWidthPx' or 'nameSolverWidthPx'.
     */
    setColumnWidthsFromWidestText(element, childrenProp, colArr, leafArr, widthProp) {
        if (element.varIsHidden) return;

        let height = N2SVGLayout.heightPx * element.numLeaves / this.zoomedElement.numLeaves;
        element.textOpacity0 = element.propExists('textOpacity') ? element.textOpacity : 0;
        element.textOpacity = (height > N2SVGLayout.fontSizePx) ? 1 : 0;
        let hasVisibleDetail = (height >= 2.0);
        let width = (hasVisibleDetail) ? N2SVGLayout.minColumnWidthPx : 1e-3;
        if (element.textOpacity > 0.5) width = element[widthProp];

        this.greatestDepth = Math.max(this.greatestDepth, element.depth);

        if (Array.isArray(element[childrenProp]) && !element.isMinimized) { //not leaf
            colArr[element.depth].width = Math.max(colArr[element.depth].width, width)
            for (let i = 0; i < element[childrenProp].length; ++i) {
                this.setColumnWidthsFromWidestText(element[childrenProp][i],
                    childrenProp, colArr, leafArr, widthProp);
            }
        }
        else { //leaf
            leafArr[element.depth] = Math.max(leafArr[element.depth], width);
        }
    }

    /** Compute column widths across the model, then adjust ends as needed.
     * @param {Object} [element = this.zoomedElement] Item to operate on.
     */
    computeColumnWidths(element = this.zoomedElement) {
        this.greatestDepth = 0;
        this.leafWidthsPx = new Array(this.model.maxDepth + 1).fill(0.0);
        this.cols = Array.from({ length: this.model.maxDepth + 1 }, () =>
            ({ 'width': 0.0, 'location': 0.0 }));

        this.setColumnWidthsFromWidestText(element, 'children', this.cols,
            this.leafWidthsPx, 'nameWidthPx');

        let sum = 0;
        let lastColumnWidth = 0;
        for (let i = this.leafWidthsPx.length - 1; i >= this.zoomedElement.depth; --i) {
            sum += this.cols[i].width;
            let lastWidthNeeded = this.leafWidthsPx[i] - sum;
            lastColumnWidth = Math.max(lastWidthNeeded, lastColumnWidth);
        }

        this.cols[this.zoomedElement.depth - 1].width = N2SVGLayout.parentNodeWidthPx;
        this.cols[this.greatestDepth].width = lastColumnWidth;
    }

    /** Compute solver column widths across the model, then adjust ends as needed.
     * @param {Object} [element = this.zoomedElement] Item to operate on.
     */
    computeSolverColumnWidths(element = this.zoomedElement) {
        this.greatestDepth = 0;
        this.leafSolverWidthsPx = new Array(this.model.maxDepth + 1).fill(0.0);
        this.solverCols = Array.from({ length: this.model.maxDepth + 1 }, () =>
            ({ 'width': 0.0, 'location': 0.0 }));

        this.setColumnWidthsFromWidestText(element, 'subsystem_children', this.solverCols,
            this.leafSolverWidthsPx, 'nameSolverWidthPx');

        let sum = 0;
        let lastColumnWidth = 0;
        for (let i = this.leafSolverWidthsPx.length - 1; i >= this.zoomedElement.depth; --i) {
            sum += this.cols[i].width;
            let lastWidthNeeded = this.leafSolverWidthsPx[i] - sum;
            lastColumnWidth = Math.max(lastWidthNeeded, lastColumnWidth);
        }

        this.solverCols[this.zoomedElement.depth - 1].width = N2SVGLayout.parentNodeWidthPx;
        this.solverCols[this.greatestDepth].width = lastColumnWidth;
    }

    /** Set the location of the columns based on the width of the columns
     * to the left.
     */
    setColumnLocations() {
        this.widthPTreePx = 0;
        this.widthPSolverTreePx = 0;

        for (let depth = 1; depth <= this.model.maxDepth; ++depth) {
            this.cols[depth].location = this.widthPTreePx;
            this.widthPTreePx += this.cols[depth].width;

            this.solverCols[depth].location = this.widthPSolverTreePx;
            this.widthPSolverTreePx += this.solverCols[depth].width;
        }
    }
}

Object.assign(N2SVGLayout, N2SVGLayout_statics);