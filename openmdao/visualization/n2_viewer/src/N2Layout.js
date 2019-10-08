let N2Layout_statics = {
    'showLinearSolverNames': true,
    'rightTextMarginPx': 8,
    'heightPx': HEIGHT_PX,
    'fontSizePx': 11,
    'minColumnWidthPx': 5,
    'parentNodeWidthPx': 40
};

/**
 * Calculates and stores the size and positions of visible elements.
 * @typedef N2Layout
 * @property {ModelData} model Reference to the preprocessed model.
 * @property {Object} zoomedElement Reference to zoomedElement managed by N2Diagram.
 * @property {Object[]} zoomedNodes  Child nodes of the current zoomed element.
 * @property {Object[]} visibleNodes Zoomed nodes that are actually drawn.
 * @property {Object[]} zoomedSolverNodes Child solver nodes of the current zoomed element.
 * @property {Boolean} updateRecomputesAutoComplete Zoomed solver nodes that are actually drawn.
 * @property {Object} svg Reference to the top-level SVG element in the document.
 */
class N2Layout {

    /**
     * Compute the new layout based on the model data and the zoomed element.
     * @param {ModelData} model The pre-processed model object.
     * @param {Object} newZoomedElement The element the new layout is based around.
     */
    constructor(model, newZoomedElement) {
        this.model = model;

        // TODO: Remove global zoomedElement
        this.zoomedElement = newZoomedElement;
        
        this.updateRecomputesAutoComplete = true;
        this.updateAutoCompleteIfNecessary();

        this.outputNamingType = "Absolute";
        this.zoomedNodes = [];
        this.visibleNodes = [];

        this.zoomedSolverNodes = [];
        this.visibleSolverNodes = [];

        this.svg = d3.select("#svgId");

        this.setupTextRenderer();
        let startTime = Date.now();
        this.updateTextWidths();
        console.log("N2Layout.updateTextWidths: ", Date.now() - startTime, "ms");

        startTime = Date.now();
        this.updateSolverTextWidths();
        console.log("N2Layout.updateSolverTextWidths: ", Date.now() - startTime, "ms");
        delete (this.textRenderer);

        startTime = Date.now();
        this.computeLeaves();
        console.log("N2Layout.computeLeaves: ", Date.now() - startTime, "ms");

        startTime = Date.now();
        this.computeColumnWidths();
        console.log("N2Layout.computeColumnWidths: ", Date.now() - startTime, "ms");

        startTime = Date.now();
        this.computeSolverColumnWidths();
        console.log("N2Layout.computeSolverColumnWidths: ", Date.now() - startTime, "ms");

        startTime = Date.now();
        this.setColumnLocations();
        console.log("N2Layout.setColumnLocations: ", Date.now() - startTime, "ms");

        startTime = Date.now();
        this.computeNormalizedPositions(this.model.root, 0, false, null);
        console.log("N2Layout.computeNormalizedPositions: ", Date.now() - startTime, "ms");
        if (this.zoomedElement.parent)
            this.zoomedNodes.push(this.zoomedElement.parent);

        startTime = Date.now();
        this.computeSolverNormalizedPositions(this.model.root, 0, false, null);
        console.log("N2Layout.computeSolverNormalizedPositions: ", Date.now() - startTime, "ms");
        if (this.zoomedElement.parent)
            this.zoomedSolverNodes.push(this.zoomedElement.parent);

    }

    /**
     * Switch back and forth between showing the linear or non-linear solver names. 
     * @return {Boolean} The new value.
     */
    static toggleSolverNameType() {
        N2Layout.showLinearSolverNames = !N2Layout.showLinearSolverNames;
        return N2Layout.showLinearSolverNames;
    }

    /** Create an off-screen area to render text for getTextWidth() */
    setupTextRenderer() {
        let textGroup = this.svg.append("svg:g").attr("class", "partition_group");
        let textSVG = textGroup.append("svg:text")
            .text("")
            .attr("x", -100); // Put text off screen to the left.

        this.textRenderer = {
            'group': textGroup,
            'textSvg': textSVG,
            'node': textSVG.node(),
            'widthCache': {}
        };
    }

    /** Insert text into an off-screen SVG text object to determine the width.
     * Cache the result so repeat calls with the same text can just do a lookup.
     * @param {string} text Text to render or find in cache.
     * @return {number} The SVG-computed width of the rendered string.
     */
    getTextWidth(text) {
        let width = 0.0;

        // Check cache first
        if (this.textRenderer.widthCache.propExists(text)) {
            width = this.textRenderer.widthCache[text];
        }
        else {
            // Not found, render and return new width.
            this.textRenderer.textSvg.text(text);
            width = this.textRenderer.node.getBoundingClientRect().width;

            this.textRenderer.widthCache[text] = width;
        }

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
        return N2Layout.showLinearSolverNames ?
            element.linear_solver : element.nonlinear_solver;
    }

    /** Determine text widths for all descendents of the specified element.
     * @param {Object} [element = this.zoomedElement] Item to begin looking from.
     */
    updateTextWidths(element = this.zoomedElement) {
        if (element.varIsHidden) return;

        element.nameWidthPx = this.getTextWidth(this.getText(element)) + 2 *
            N2Layout.rightTextMarginPx;

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
            N2Layout.rightTextMarginPx;

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

        let height = N2Layout.heightPx * element.numLeaves / this.zoomedElement.numLeaves;
        element.textOpacity0 = element.propExists('textOpacity') ? element.textOpacity : 0;
        element.textOpacity = (height > N2Layout.fontSizePx) ? 1 : 0;
        let hasVisibleDetail = (height >= 2.0);
        let width = (hasVisibleDetail) ? N2Layout.minColumnWidthPx : 1e-3;
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

        this.cols[this.zoomedElement.depth - 1].width = N2Layout.parentNodeWidthPx;
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
            sum += this.solverCols[i].width;
            let lastWidthNeeded = this.leafSolverWidthsPx[i] - sum;
            lastColumnWidth = Math.max(lastWidthNeeded, lastColumnWidth);
        }

        this.solverCols[this.zoomedElement.depth - 1].width = N2Layout.parentNodeWidthPx;
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

    /**TODO: computeNormalizedPositions and computeSolverNormalizedPositions do almost
     * identical things, just storing info in different variables, so they should be
     * merged as much as possible.
     */

    /** TODO: Document what the *0 variables are for */

    /** Recurse over the model tree and determine the coordinates and
     * size of visible elements. If a parent is minimized, operations are
     * performed on it instead.
     * @param {Object} element The node to operate on.
     * @param {number} leafCounter Tally of leaves encountered so far.
     * @param {Boolean} isChildOfZoomed Whether element is a descendant of this.zoomedElement.
     * @param {Object} earliestMinimizedParent The minimized parent, if any, appearing
     *   highest in the tree hierarchy. Null if none exist.
     */
    computeNormalizedPositions(element, leafCounter,
        isChildOfZoomed, earliestMinimizedParent) {
        if (!isChildOfZoomed) {
            isChildOfZoomed = (element === this.zoomedElement);
        }

        if (earliestMinimizedParent == null && isChildOfZoomed) {
            if (!element.varIsHidden) this.zoomedNodes.push(element);
            if (!Array.isArray(element.children) || element.isMinimized) { //at a "leaf" node
                if (!element.varIsHidden) this.visibleNodes.push(element);
                earliestMinimizedParent = element;
            }
        }

        let node = (earliestMinimizedParent) ? earliestMinimizedParent : element;
        element.rootIndex0 = element.propExists('rootIndex') ? element.rootIndex : leafCounter;
        element.rootIndex = leafCounter;
        ['x', 'y', 'width', 'height'].forEach(function (val) {
            element[val + '0'] = element.hasOwnProperty(val) ? element[val] : 1e-6;
        })
        element.x = this.cols[node.depth].location / this.widthPTreePx;
        element.y = leafCounter / this.model.root.numLeaves;
        element.width = (Array.isArray(element.children) && !element.isMinimized) ?
            (this.cols[node.depth].width / this.widthPTreePx) : 1 - node.x;
        element.height = node.numLeaves / this.model.root.numLeaves;

        if (element.varIsHidden) { //param or hidden leaf leaving
            element.x = this.cols[element.parentComponent.depth + 1].location / this.widthPTreePx;
            element.y = element.parentComponent.y;
            element.width = 1e-6;
            element.height = 1e-6;
        }

        if (Array.isArray(element.children)) {
            for (let i = 0; i < element.children.length; ++i) {
                this.computeNormalizedPositions(element.children[i], leafCounter,
                    isChildOfZoomed, earliestMinimizedParent);
                if (earliestMinimizedParent == null) { //numleaves is only valid passed nonminimized nodes
                    leafCounter += element.children[i].numLeaves;
                }
            }
        }
    }

    /** Recurse over the model tree and determine the coordinates and
     * size of visible solver elements. If a parent is minimized, operations are
     * performed on it instead.
     * @param {Object} element The node to operate on.
     * @param {number} leafCounter Tally of leaves encountered so far.
     * @param {Boolean} isChildOfZoomed Whether element is a descendant of this.zoomedElement.
     * @param {Object} earliestMinimizedParent The minimized parent, if any, appearing
     *   highest in the tree hierarchy. Null if none exist.
     */
    computeSolverNormalizedPositions(element, leafCounter,
        isChildOfZoomed, earliestMinimizedParent) {
        if (!isChildOfZoomed) {
            isChildOfZoomed = (element === this.zoomedElement);
        }

        if (earliestMinimizedParent == null && isChildOfZoomed) {
            if (element.type.match(/^(subsystem|root)$/)) {
                this.zoomedSolverNodes.push(element);
            }
            if (!Array.isArray(element.children) || element.isMinimized) { //at a "leaf" node
                if (!element.type.match(/^(param|unconnected_param)$/) && !element.varIsHidden) {
                    this.visibleSolverNodes.push(element);
                }
                earliestMinimizedParent = element;
            }
        }

        let node = (earliestMinimizedParent) ? earliestMinimizedParent : element;
        element.rootIndex0 = element.hasOwnProperty('rootIndex') ? element.rootIndex : leafCounter;
        ['x', 'y', 'width', 'height'].forEach(function (val) {
            val += 'Solver';
            element[val + '0'] = element.hasOwnProperty(val) ? element[val] : 1e-6;
        })
        element.xSolver = this.solverCols[node.depth].location / this.widthPSolverTreePx;
        element.ySolver = leafCounter / this.model.root.numLeaves;
        element.widthSolver = (element.subsystem_children && !element.isMinimized) ?
            (this.solverCols[node.depth].width / this.widthPSolverTreePx) :
            1 - node.xSolver; //1-d.x;

        element.heightSolver = node.numLeaves / this.model.root.numLeaves;

        if (element.varIsHidden) { //param or hidden leaf leaving
            element.xSolver = this.cols[element.parentComponent.depth + 1].location / this.widthPTreePx;
            element.ySolver = element.parentComponent.y;
            element.widthSolver = 1e-6;
            element.heightSolver = 1e-6;
        }

        if (Array.isArray(element.children)) {
            for (let i = 0; i < element.children.length; ++i) {
                this.computeSolverNormalizedPositions(element.children[i],
                    leafCounter, isChildOfZoomed, earliestMinimizedParent);
                if (earliestMinimizedParent == null) { //numleaves is only valid passed nonminimized nodes
                    leafCounter += element.children[i].numLeaves;
                }
            }
        }
    }

    /**
     * Recurse through the children of the element and add their names to the
     * autocomplete list of names if they're not already in it.
     * @param {Object} element The element to search from.
     */
    populateAutoCompleteList(element) {
        // Depth first, don't go into minimized children
        if (Array.isPopulatedArray(element.children) && !element.isMinimized) {
            for (let i = 0; i < element.children.length; ++i) {
                this.populateAutoCompleteList(element.children[i]);
            }
        }

        if (element === this.zoomedElement) return;

        let curName = element.name;
        if (element.splitByColon && element.children &&
            element.children.length > 0) curName += ":";

        if (!element.type.match(paramOrUnknownRegex)) curName += ".";
        let namesToAdd = [curName];

        if (element.splitByColon)
            namesToAdd.push(element.colonName +
                ((Array.isPopulatedArray(element.children)) ? ":" : ""));

        namesToAdd.forEach(function (name) {
            if (!this.autoCompleteSetNames.hasOwnProperty(name)) {
                this.autoCompleteSetNames[name] = true;
                autoCompleteListNames.push(name);
            }
        }.bind(this));

        let localPathName = (this.zoomedElement === this.modelroot) ?
            element.absPathName :
            element.absPathName.slice(zoomedElement.absPathName.length + 1);

        if (!this.autoCompleteSetPathNames.hasOwnProperty(localPathName)) {
            this.autoCompleteSetPathNames[localPathName] = true;
            autoCompleteListPathNames.push(localPathName);
        }
    }

    /**
     * If this.updateRecomputesAutoComplete is true, update the autocomplete
     * list. If false, set it to true and return.
     */
    updateAutoCompleteIfNecessary() {
        if (!this.updateRecomputesAutoComplete) {
            this.updateRecomputesAutoComplete = true;
            return;
        }
        this.autoCompleteSetNames = {};
        this.autoCompleteSetPathNames = {};

        autoCompleteListNames = [];
        autoCompleteListPathNames = [];

        this.populateAutoCompleteList(this.zoomedElement);

        delete this.autoCompleteSetNames;
        delete this.autoCompleteSetPathNames;
    }
}

Object.assign(N2Layout, N2Layout_statics);