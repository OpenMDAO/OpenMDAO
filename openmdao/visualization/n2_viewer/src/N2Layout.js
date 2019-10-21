/**
 * Calculates and stores the size and positions of visible elements.
 * @typedef N2Layout
 * @property {ModelData} model Reference to the preprocessed model.
 * @property {N2TreeNode} zoomedElement Reference to zoomedElement managed by N2Diagram.
 * @property {N2TreeNode[]} zoomedNodes  Child workNodes of the current zoomed element.
 * @property {N2TreeNode[]} visibleNodes Zoomed workNodes that are actually drawn.
 * @property {N2TreeNode[]} zoomedSolverNodes Child solver workNodes of the current zoomed element.
 * @property {Boolean} updateRecomputesAutoComplete Zoomed solver workNodes that are actually drawn.
 * @property {Object} svg Reference to the top-level SVG element in the document.
 * @property {Object} size The dimensions of the model and solver trees.
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

        // Initial size values derived from read-only defaults
        this.size = JSON.parse(JSON.stringify(N2Layout.defaults.size));
        this.showLinearSolverNames = N2Layout.defaults.showLinearSolverNames;

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
    toggleSolverNameType() {
        this.showLinearSolverNames = !this.showLinearSolverNames;
        return this.showLinearSolverNames;
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
            'workNode': textSVG.node(),
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
            width = this.textRenderer.workNode.getBoundingClientRect().width;

            this.textRenderer.widthCache[text] = width;
        }

        return width;
    }

    /** Determine the text associated with the element. Normally its name,
     * but can be changed if promoted or originally contained a colon.
     * @param {N2TreeNode} element The item to operate on.
     * @return {string} The selected text.
     */
    getText(element) {
        let retVal = element.name;

        if (this.outputNamingType == "Promoted" &&
            element.isParamOrUnknown() &&
            this.zoomedElement.propExists('promotions') &&
            this.zoomedElement.promotions.propExists(element.absPathName)) {
            retVal = this.zoomedElement.promotions[element.absPathName];
        }

        if (element.splitByColon && element.hasChildren()) {
            retVal += ":";
        }

        return retVal;
    }

    /** Return a the name of the linear or non-linear solver depending
     * on the current setting.
     * @param {N2TreeNode} element The item to get the solver text from.
     */
    getSolverText(element) {
        return this.showLinearSolverNames ?
            element.linear_solver : element.nonlinear_solver;
    }

    /** Determine text widths for all descendents of the specified element.
     * @param {N2TreeNode} [element = this.zoomedElement] Item to begin looking from.
     */
    updateTextWidths(element = this.zoomedElement) {
        if (element.varIsHidden) return;

        element.nameWidthPx = this.getTextWidth(this.getText(element)) + 2 *
            this.size.rightTextMargin;

        if (element.hasChildren()) {
            for (let child of element.children) {
                this.updateTextWidths(child);
            }
        }
    }

    /** Determine text width for this and all decendents if element is a solver.
     * @param {N2TreeNode} [element = this.zoomedElement] Item to begin looking from.
     */
    updateSolverTextWidths(element = this.zoomedElement) {
        if (element.isParam() || element.varIsHidden) {
            return;
        }

        element.nameSolverWidthPx = this.getTextWidth(this.getSolverText(element)) + 2 *
            this.size.rightTextMargin;

        if (element.hasChildren()) {
            for (let child of element.children) {
                this.updateSolverTextWidths(child);
            }
        }
    }

    /** Recurse through the tree and add up the number of leaves that each
     * node has, based on their array of children.
     * @param {N2TreeNode} [element = this.model.root] The starting node.
     */
    computeLeaves(element = this.model.root) {
        if (element.varIsHidden) {
            element.numLeaves = 0;
        }
        else if (element.hasChildren() && !element.isMinimized) {
            element.numLeaves = 0;
            for (let child of element.children) {
                this.computeLeaves(child);
                element.numLeaves += child.numLeaves;
            }
        }
        else {
            element.numLeaves = 1;
        }
    }

    /** For visible elements with children, choose a column width
     * large enough to accomodate the widest label in their column.
     * @param {N2TreeNode} element The item to operate on.
     * @param {string} childrenProp Either 'children' or 'subsystem_children'.
     * @param {Object[]} colArr The array of column info.
     * @param {number[]} leafArr The array of leaf width info.
     * @param {string} widthProp Either 'nameWidthPx' or 'nameSolverWidthPx'.
     */
    setColumnWidthsFromWidestText(element, childrenProp, colArr, leafArr, widthProp) {
        if (element.varIsHidden) return;

        let height = this.size.diagram.height * element.numLeaves / this.zoomedElement.numLeaves;
        element.prevTextOpacity = element.propExists('textOpacity') ? element.textOpacity : 0;
        element.textOpacity = (height > this.size.font) ? 1 : 0;
        let hasVisibleDetail = (height >= 2.0);
        let width = (hasVisibleDetail) ? this.size.minColumnWidth : 1e-3;
        if (element.textOpacity > 0.5) width = element[widthProp];

        this.greatestDepth = Math.max(this.greatestDepth, element.depth);

        if (element.hasChildren(childrenProp) && !element.isMinimized) { //not leaf
            colArr[element.depth].width = Math.max(colArr[element.depth].width, width)
            for (let child of element[childrenProp]) {
                this.setColumnWidthsFromWidestText(child, childrenProp, colArr, leafArr, widthProp);
            }
        }
        else { //leaf
            leafArr[element.depth] = Math.max(leafArr[element.depth], width);
        }
    }

    /** Compute column widths across the model, then adjust ends as needed.
     * @param {N2TreeNode} [element = this.zoomedElement] Item to operate on.
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

        this.cols[this.zoomedElement.depth - 1].width = this.size.parentNodeWidth;
        this.cols[this.greatestDepth].width = lastColumnWidth;
    }

    /** Compute solver column widths across the model, then adjust ends as needed.
     * @param {N2TreeNode} [element = this.zoomedElement] Item to operate on.
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

        this.solverCols[this.zoomedElement.depth - 1].width = this.size.parentNodeWidth;
        this.solverCols[this.greatestDepth].width = lastColumnWidth;
    }

    /** Set the location of the columns based on the width of the columns
     * to the left.
     */
    setColumnLocations() {
        this.size.partitionTree.width = 0;
        this.size.solverTree.width = 0;

        for (let depth = 1; depth <= this.model.maxDepth; ++depth) {
            this.cols[depth].location = this.size.partitionTree.width;
            this.size.partitionTree.width += this.cols[depth].width;

            this.solverCols[depth].location = this.size.solverTree.width;
            this.size.solverTree.width += this.solverCols[depth].width;
        }
    }

    /**TODO: computeNormalizedPositions and computeSolverNormalizedPositions do almost
     * identical things, just storing info in different variables, so they should be
     * merged as much as possible.
     */

    /** TODO: Document what the *0 variables are for */

    /** Recurse over the model tree and determine the coordinates and
     * size of visible nodes. If a parent is minimized, operations are
     * performed on it instead.
     * @param {N2TreeNode} node The node to operate on.
     * @param {number} leafCounter Tally of leaves encountered so far.
     * @param {Boolean} isChildOfZoomed Whether node is a descendant of this.zoomedElement.
     * @param {Object} earliestMinimizedParent The minimized parent, if any, appearing
     *   highest in the tree hierarchy. Null if none exist.
     */
    computeNormalizedPositions(node, leafCounter,
        isChildOfZoomed, earliestMinimizedParent) {
        if (!isChildOfZoomed) {
            isChildOfZoomed = (node === this.zoomedElement);
        }

        if (earliestMinimizedParent == null && isChildOfZoomed) {
            if (!node.varIsHidden) this.zoomedNodes.push(node);
            if (!node.hasChildren() || node.isMinimized) { //at a "leaf" node
                if (!node.varIsHidden) this.visibleNodes.push(node);
                earliestMinimizedParent = node;
            }
        }

        let workNode = (earliestMinimizedParent) ? earliestMinimizedParent : node;
        node.preserveDims(false, leafCounter);

        node.dims.x = this.cols[workNode.depth].location / this.size.partitionTree.width;
        node.dims.y = leafCounter / this.model.root.numLeaves;
        node.dims.width = (node.hasChildren() && !node.isMinimized) ?
            (this.cols[workNode.depth].width / this.size.partitionTree.width) : 1 - workNode.dims.x;
        node.dims.height = workNode.numLeaves / this.model.root.numLeaves;

        if (node.varIsHidden) { // param or hidden leaf leaving
            node.dims.x = this.cols[node.parentComponent.depth + 1].location / this.size.partitionTree.width;
            node.dims.y = node.parentComponent.dims.y;
            node.dims.width = node.dims.height = 1e-6;
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                this.computeNormalizedPositions(child, leafCounter,
                    isChildOfZoomed, earliestMinimizedParent);
                if (earliestMinimizedParent == null) { //numleaves is only valid passed nonminimized nodes
                    leafCounter += child.numLeaves;
                }
            }
        }
    }

    /**
     * Recurse over the model tree and determine the coordinates and size of visible
     * solver nodes. If a parent is minimized, operations are performed on it instead.
     * @param {N2TreeNode} node The node to operate on.
     * @param {number} leafCounter Tally of leaves encountered so far.
     * @param {Boolean} isChildOfZoomed Whether node is a descendant of this.zoomedElement.
     * @param {Object} earliestMinimizedParent The minimized parent, if any, appearing
     *   highest in the tree hierarchy. Null if none exist.
     */
    computeSolverNormalizedPositions(node, leafCounter,
        isChildOfZoomed, earliestMinimizedParent) {
        if (!isChildOfZoomed) {
            isChildOfZoomed = (node === this.zoomedElement);
        }

        if (earliestMinimizedParent == null && isChildOfZoomed) {
            if (node.type.match(/^(subsystem|root)$/)) {
                this.zoomedSolverNodes.push(node);
            }
            if (!node.hasChildren() || node.isMinimized) { //at a "leaf" workNode
                if (!node.isParam() && !node.varIsHidden) {
                    this.visibleSolverNodes.push(node);
                }
                earliestMinimizedParent = node;
            }
        }

        let workNode = (earliestMinimizedParent) ? earliestMinimizedParent : node;
        node.preserveDims(true, leafCounter);

        node.solverDims.x = this.solverCols[workNode.depth].location / this.size.solverTree.width;
        node.solverDims.y = leafCounter / this.model.root.numLeaves;
        node.solverDims.width = (node.subsystem_children && !node.isMinimized) ?
            (this.solverCols[workNode.depth].width / this.size.solverTree.width) :
            1 - workNode.solverDims.x;

        node.solverDims.height = workNode.numLeaves / this.model.root.numLeaves;

        if (node.varIsHidden) { //param or hidden leaf leaving
            node.solverDims.x = this.cols[node.parentComponent.depth + 1].location /
                this.size.partitionTree.width;
            node.solverDims.y = node.parentComponent.dims.y;
            node.solverDims.width = node.solverDims.height = 1e-6;
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                this.computeSolverNormalizedPositions(child,
                    leafCounter, isChildOfZoomed, earliestMinimizedParent);
                if (earliestMinimizedParent == null) { //numleaves is only valid passed nonminimized nodes
                    leafCounter += child.numLeaves;
                }
            }
        }
    }

    /**
     * Recurse through the children of the element and add their names to the
     * autocomplete list of names if they're not already in it.
     * @param {N2TreeNode} element The element to search from.
     */
    populateAutoCompleteList(element) {
        // Depth first, don't go into minimized children
        if (element.hasChildren() && !element.isMinimized) {
            for (let child of element.children) {
                this.populateAutoCompleteList(child);
            }
        }

        if (element === this.zoomedElement) return;

        let curName = element.name;
        if (element.splitByColon && element.hasChildren()) curName += ":";

        if (!element.isParamOrUnknown()) curName += ".";
        let namesToAdd = [curName];

        if (element.splitByColon)
            namesToAdd.push(element.colonName +
                ((element.hasChildren()) ? ":" : ""));

        for (let name of namesToAdd) {
            if (!this.autoCompleteSetNames.hasOwnProperty(name)) {
                this.autoCompleteSetNames[name] = true;
                autoCompleteListNames.push(name);
            }
        };

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

    newSvgDivDimAttribs() {
        let width = (this.size.partitionTree.width +
            this.size.partitionTreeGap +
            this.size.diagram.width +
            this.size.solverTree.width +
            this.size.svgMargin * 2 +
            this.size.partitionTreeGap) +
            this.size.unit;

        let height = (this.size.partitionTree.height +
            this.size.svgMargin * 2) +
            this.size.unit;

        return ({ 'width': width, 'height': height });
    }

    newSvgElemDimAttribs() {
        let width = this.size.partitionTree.width +
            this.size.partitionTreeGap +
            this.size.diagram.width +
            this.size.solverTree.width +
            this.size.svgMargin * 2 +
            this.size.partitionTreeGap;

        let height = this.size.partitionTree.height +
            this.size.svgMargin * 2;

        return ({ 'width': width, 'height': height });
    }

}