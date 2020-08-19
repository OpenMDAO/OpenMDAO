/**
 * Calculates and stores the size and positions of visible elements.
 * @typedef N2Layout
 * @property {ModelData} model Reference to the preprocessed model.
 * @property {N2TreeNode} zoomedElement Reference to zoomedElement managed by N2Diagram.
 * @property {N2TreeNode[]} zoomedNodes  Child workNodes of the current zoomed element.
 * @property {N2TreeNode[]} visibleNodes Zoomed workNodes that are actually drawn.
 * @property {N2TreeNode[]} zoomedSolverNodes Child solver workNodes of the current zoomed element.
 * @property {Object} svg Reference to the top-level SVG element in the document.
 * @property {Object} size The dimensions of the model and solver trees.
 */
class N2Layout {

    /**
     * Compute the new layout based on the model data and the zoomed element.
     * @param {ModelData} model The pre-processed model object.
     * @param {Object} newZoomedElement The element the new layout is based around.
     * @param {boolean} showLinearSolverNames Whether to show linear or non-linear solver names.
     * @param {Object} dims The initial sizes for multiple tree elements.
     */
    constructor(model, newZoomedElement, showLinearSolverNames, dims) {
        this.model = model;

        this.zoomedElement = newZoomedElement;
        this.showLinearSolverNames = showLinearSolverNames;

        this.outputNamingType = "Absolute";
        this.zoomedNodes = [];
        this.visibleNodes = [];

        this.zoomedSolverNodes = [];
        this.visibleSolverNodes = [];
        this.curVisibleNodeCount = 0;
        this.prevVisibleNodeCount = 0;

        // Initial size values derived from read-only defaults
        this.size = dims.size;
        this.svg = d3.select("#svgId");

        startTimer('N2Layout._computeLeaves');
        this._computeLeaves();
        stopTimer('N2Layout._computeLeaves');

        this._setupTextRenderer();
        startTimer('N2Layout._updateTextWidths');
        this._updateTextWidths();
        stopTimer('N2Layout._updateTextWidths');

        startTimer('N2Layout._updateSolverTextWidths');
        this._updateSolverTextWidths();
        stopTimer('N2Layout._updateSolverTextWidths');
        delete (this.textRenderer);

        startTimer('N2Layout._computeColumnWidths');
        this._computeColumnWidths();
        stopTimer('N2Layout._computeColumnWidths');

        startTimer('N2Layout._computeSolverColumnWidths');
        this._computeSolverColumnWidths();
        stopTimer('N2Layout._computeSolverColumnWidths');

        startTimer('N2Layout._setColumnLocations');
        this._setColumnLocations();
        stopTimer('N2Layout._setColumnLocations');

        startTimer('N2Layout._computeNormalizedPositions');
        this._computeNormalizedPositions(this.model.root, 0, false, null);
        stopTimer('N2Layout._computeNormalizedPositions');
        if (this.zoomedElement.parent)
            this.zoomedNodes.push(this.zoomedElement.parent);

        startTimer('N2Layout._computeSolverNormalizedPositions');
        this._computeSolverNormalizedPositions(this.model.root, 0, false, null);
        stopTimer('N2Layout._computeSolverNormalizedPositions');
        if (this.zoomedElement.parent)
            this.zoomedSolverNodes.push(this.zoomedElement.parent);

        this.setTransitionPermission();

    }

    /**
     * If there are too many nodes, don't bother with transition animations
     * because it will cause lag and may timeout anyway. This is accomplished
     * by redefining a few D3 methods to return the same selection instead
     * of a transition object. When the number of nodes is low enough,
     * the original transition methods are restored.
     */
    setTransitionPermission() {
        this.prevVisibleNodeCount = this.visibleNodeCount;
        this.visibleNodeCount = this.visibleNodes.length;
        const highWaterMark = Math.max(this.prevVisibleNodeCount, this.visibleNodeCount);

        // Too many nodes, disable transitions.
        if (highWaterMark >= N2TransitionDefaults.maxNodes) {
            debugInfo("Denying transitions: ", this.visibleNodes.length,
                " visible nodes, max allowed: ", N2TransitionDefaults.maxNodes)

            // Return if already denied
            if (!d3.selection.prototype.transitionAllowed) return;
            d3.selection.prototype.transitionAllowed = false;

            d3.selection.prototype.transition = returnThis;
            d3.selection.prototype.duration = returnThis;
            d3.selection.prototype.delay = returnThis;
        }
        else { // OK, enable transitions.
            debugInfo("Allowing transitions: ", this.visibleNodes.length,
                " visible nodes, max allowed: ", N2TransitionDefaults.maxNodes)

            // Return if already allowed
            if (d3.selection.prototype.transitionAllowed) return;
            d3.selection.prototype.transitionAllowed = true;

            for (const func in d3.selection.prototype.originalFuncs) {
                d3.selection.prototype[func] =
                    d3.selection.prototype.originalFuncs[func];
            }
        }
    }

    /** Create an off-screen area to render text for _getTextWidth() */
    _setupTextRenderer() {
        const textGroup = this.svg.select('#text-width-renderer');
        const textSVG = textGroup.select('text');

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
    _getTextWidth(text) {
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

    /** Determine the text associated with the node. Normally its name,
     * but can be changed if promoted.
     * @param {N2TreeNode} node The item to operate on.
     * @return {string} The selected text.
     */
    getText(node) {
        testThis(this, 'N2Layout', 'getText');

        let retVal = node.name;

        if (this.outputNamingType == "Promoted" &&
            node.isInputOrOutput() &&
            this.zoomedElement.propExists('promotions') &&
            this.zoomedElement.promotions.propExists(node.absPathName)) {
            retVal = this.zoomedElement.promotions[node.absPathName];
        }

        return retVal;
    }

    /**
     * Return the name of the linear or non-linear solver depending
     * on the current setting.
     * @param {N2TreeNode} node The item to get the solver text from.
     */
    getSolverText(node) {
        testThis(this, 'N2Layout', 'getSolverText');

        let solver_name = this.showLinearSolverNames ? node.linear_solver : node.nonlinear_solver;

        if (!this.showLinearSolverNames && node.hasOwnProperty("solve_subsystems") && node.solve_subsystems) {
            return solver_name + " (sub_solve)";
        }
        else {
            return solver_name;
        }
    }

    /**
     * Determine text widths for all descendents of the specified node.
     * @param {N2TreeNode} [node = this.zoomedElement] Item to begin looking from.
     */
    _updateTextWidths(node = this.zoomedElement) {
        if (node.varIsHidden) return;

        node.nameWidthPx = this._getTextWidth(this.getText(node)) + 2 *
            this.size.rightTextMargin;

        if (node.hasChildren() && !node.isMinimized) {
            for (let child of node.children) {
                this._updateTextWidths(child);
            }
        }
    }

    /**
     * Determine text width for this and all decendents if node is a solver.
     * @param {N2TreeNode} [node = this.zoomedElement] Item to begin looking from.
     */
    _updateSolverTextWidths(node = this.zoomedElement) {
        if (node.isInput() || node.varIsHidden) {
            return;
        }

        node.nameSolverWidthPx = this._getTextWidth(this.getSolverText(node)) + 2 *
            this.size.rightTextMargin;

        if (node.hasChildren() && !node.isMinimized) {
            for (let child of node.children) {
                this._updateSolverTextWidths(child);
            }
        }
    }

    /** Recurse through the tree and add up the number of leaves that each
     * node has, based on their array of visible children.
     * @param {N2TreeNode} [node = this.model.root] The starting node.
     */
    _computeLeaves(node = this.model.root) {
        node.numLeaves = 0;

        if (! node.varIsHidden) {
            if (this.model.nodeIds.length > Precollapse.minimumNodes) {
                node.minimizeIfLarge(this.model.depthCount[node.depth]);
            }

            if (node.hasChildren() && !node.isMinimized) {
                for (let child of node.children) {
                    this._computeLeaves(child);
                    node.numLeaves += child.numLeaves;
                }
            }
            else {
                node.numLeaves = 1;
            }
        }
    }

    /** For visible nodes with children, choose a column width
     * large enough to accomodate the widest label in their column.
     * @param {N2TreeNode} node The item to operate on.
     * @param {string} childrenProp Either 'children' or 'subsystem_children'.
     * @param {Object[]} colArr The array of column info.
     * @param {number[]} leafArr The array of leaf width info.
     * @param {string} widthProp Either 'nameWidthPx' or 'nameSolverWidthPx'.
     */
    _setColumnWidthsFromWidestText(node, childrenProp, colArr, leafArr, widthProp) {
        if (node.varIsHidden) return;

        let height = this.size.n2matrix.height * node.numLeaves / this.zoomedElement.numLeaves;
        node.prevTextOpacity = node.propExists('textOpacity') ? node.textOpacity : 0;
        node.textOpacity = (height > this.size.font) ? 1 : 0;
        let hasVisibleDetail = (height >= 2.0);
        let width = (hasVisibleDetail) ? this.size.minColumnWidth : 1e-3;
        if (node.textOpacity > 0.5) width = node[widthProp];

        this.greatestDepth = Math.max(this.greatestDepth, node.depth);

        if (node.hasChildren(childrenProp) && !node.isMinimized) { //not leaf
            colArr[node.depth].width = Math.max(colArr[node.depth].width, width)
            for (let child of node[childrenProp]) {
                this._setColumnWidthsFromWidestText(child, childrenProp, colArr, leafArr, widthProp);
            }
        }
        else { //leaf
            leafArr[node.depth] = Math.max(leafArr[node.depth], width);
        }
    }

    /** Compute column widths across the model, then adjust ends as needed.
     * @param {N2TreeNode} [node = this.zoomedElement] Item to operate on.
     */
    _computeColumnWidths(node = this.zoomedElement) {
        this.greatestDepth = 0;
        this.leafWidthsPx = new Array(this.model.maxDepth + 1).fill(0.0);
        this.cols = Array.from({
            length: this.model.maxDepth + 1
        }, () =>
            ({
                'width': 0.0,
                'location': 0.0
            }));

        this._setColumnWidthsFromWidestText(node, 'children', this.cols,
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
     * @param {N2TreeNode} [node = this.zoomedElement] Item to operate on.
     */
    _computeSolverColumnWidths(node = this.zoomedElement) {
        this.greatestDepth = 0;
        this.leafSolverWidthsPx = new Array(this.model.maxDepth + 1).fill(0.0);
        this.solverCols = Array.from({
            length: this.model.maxDepth + 1
        }, () =>
            ({
                'width': 0.0,
                'location': 0.0
            }));

        this._setColumnWidthsFromWidestText(node, 'subsystem_children', this.solverCols,
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
    _setColumnLocations() {
        this.size.partitionTree.width = 0;
        this.size.solverTree.width = 0;

        for (let depth = 1; depth <= this.model.maxDepth; ++depth) {
            this.cols[depth].location = this.size.partitionTree.width;
            this.size.partitionTree.width += this.cols[depth].width;

            this.solverCols[depth].location = this.size.solverTree.width;
            this.size.solverTree.width += this.solverCols[depth].width;
        }

    }

    /**TODO: _computeNormalizedPositions and _computeSolverNormalizedPositions do almost
     * identical things, just storing info in different variables, so they should be
     * merged as much as possible.
     */

    /**
     * Recurse over the model tree and determine the coordinates and
     * size of visible nodes. If a parent is minimized, operations are
     * performed on it instead.
     * @param {N2TreeNode} node The node to operate on.
     * @param {number} leafCounter Tally of leaves encountered so far.
     * @param {Boolean} isChildOfZoomed Whether node is a descendant of this.zoomedElement.
     * @param {Object} earliestMinimizedParent The minimized parent, if any, appearing
     *   highest in the tree hierarchy. Null if none exist.
     */
    _computeNormalizedPositions(node, leafCounter,
        isChildOfZoomed, earliestMinimizedParent) {
        if (!isChildOfZoomed) {
            isChildOfZoomed = (node === this.zoomedElement);
        }

        if (earliestMinimizedParent == null && isChildOfZoomed) {
            if (!node.varIsHidden) this.zoomedNodes.push(node);
            if (!node.hasChildren() || node.isMinimized) { // at a "leaf" node
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

        if (node.varIsHidden) { // input or hidden leaf leaving
            node.dims.x = this.cols[node.parentComponent.depth + 1].location / this.size.partitionTree.width;
            node.dims.y = node.parentComponent.dims.y;
            node.dims.width = node.dims.height = 1e-6;
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                this._computeNormalizedPositions(child, leafCounter,
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
    _computeSolverNormalizedPositions(node, leafCounter,
        isChildOfZoomed, earliestMinimizedParent) {
        if (!isChildOfZoomed) {
            isChildOfZoomed = (node === this.zoomedElement);
        }

        if (earliestMinimizedParent == null && isChildOfZoomed) {
            if (node.type.match(/^(subsystem|root)$/)) {
                this.zoomedSolverNodes.push(node);
            }
            if (!node.hasChildren() || node.isMinimized) { //at a "leaf" workNode
                if (!node.isInput() && !node.varIsHidden) {
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

        if (node.varIsHidden) { //input or hidden leaf leaving
            node.solverDims.x = this.cols[node.parentComponent.depth + 1].location /
                this.size.partitionTree.width;
            node.solverDims.y = node.parentComponent.dims.y;
            node.solverDims.width = node.solverDims.height = 1e-6;
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                this._computeSolverNormalizedPositions(child,
                    leafCounter, isChildOfZoomed, earliestMinimizedParent);
                if (earliestMinimizedParent == null) { //numleaves is only valid passed nonminimized nodes
                    leafCounter += child.numLeaves;
                }
            }
        }
    }

    /**
     * Calculate new dimensions for the div element enclosing the main SVG element.
     * @returns {Object} Members width and height as strings with the unit appended.
     */
    newOuterDims() {
        let width = (this.size.partitionTree.width +
            this.size.n2matrix.margin +
            this.size.n2matrix.width +
            this.size.solverTree.width +
            this.size.n2matrix.margin);

        let height = (this.size.n2matrix.height +
            this.size.n2matrix.margin * 2);

        return ({
            'width': width,
            'height': height
        });
    }

    /**
     * Calculate new dimensions for the main SVG element.
     * @returns {Object} Members width and height as numbers.
     */
    newInnerDims() {
        let width = this.size.partitionTree.width +
            this.size.n2matrix.margin +
            this.size.n2matrix.width +
            this.size.solverTree.width +
            this.size.n2matrix.margin;

        let height = this.size.partitionTree.height;
        let margin = this.size.n2matrix.margin;

        return ({
            'width': width,
            'height': height,
            'margin': margin
        });
    }

    /**
     * Update container element dimensions when a new layout is calculated,
     * and set up transitions.
     * @param {Object} dom References to HTML elements.
     * @param {number} transitionStartDelay ms to wait before performing transition
     */
    updateTransitionInfo(dom, transitionStartDelay, manuallyResized) {

        sharedTransition = d3.transition()
            .duration(N2TransitionDefaults.duration)
            .delay(transitionStartDelay)
            // Hide the transition waiting animation when it ends:
            .on('end', function() { dom.waiter.attr('class', 'no-show'); });

        this.transitionStartDelay = N2TransitionDefaults.startDelay;

        let outerDims = this.newOuterDims();
        let innerDims = this.newInnerDims();

        this.ratio = (window.innerWidth - 200) / outerDims.width;
        if (this.ratio > 1 || manuallyResized) this.ratio = 1;
        else if (this.ratio < 1)
            debugInfo("Scaling diagram to " + Math.round(this.ratio * 100) + "%");

        dom.svgDiv
            .style("width", (outerDims.width * this.ratio) + this.size.unit)
            .style("height", (outerDims.height * this.ratio) + this.size.unit)

        dom.svg
            .transition(sharedTransition)
            .style("transform", "scale(" + this.ratio + ")")
            .attr("width", outerDims.width + this.size.unit)
            .attr("height", outerDims.height + this.size.unit);

        this.gapDist = (this.size.partitionTreeGap * this.ratio) - 3;
        this.gapSpace = this.gapDist + this.size.unit
        d3.select('#n2-resizer-box')
            .transition(sharedTransition)
            .style('bottom', this.gapSpace);

        dom.pTreeGroup
            .transition(sharedTransition)
            .attr("height", innerDims.height)
            .attr("width", this.size.partitionTree.width)
            .attr("transform", "translate(0 " + innerDims.margin + ")");

        dom.highlightBar
            .transition(sharedTransition)
            .attr("height", innerDims.height)
            .attr("width", "8")
            .attr("transform", "translate(" + this.size.partitionTree.width + 1 + " " + innerDims.margin + ")");

        // Move n2 outer group to right of partition tree, spaced by the margin.
        dom.n2OuterGroup
            .transition(sharedTransition)
            .attr("height", outerDims.height)
            .attr("width", outerDims.height)
            .attr("transform", "translate(" +
                (this.size.partitionTree.width) + " 0)");

        dom.n2InnerGroup.transition(sharedTransition)
            .attr("height", innerDims.height)
            .attr("width", innerDims.height)
            .attr("transform", "translate(" + innerDims.margin + " " + innerDims.margin + ")");

        dom.n2BackgroundRect.transition(sharedTransition)
            .attr("width", innerDims.height)
            .attr("height", innerDims.height)
            .attr("transform", "translate(0 0)");

        dom.pSolverTreeGroup.transition(sharedTransition)
            .attr("height", innerDims.height)
            .attr("transform", "translate(" + (this.size.partitionTree.width +
                innerDims.margin +
                innerDims.height +
                innerDims.margin) + " " +
                innerDims.margin + ")");
    }

    calcWidthBasedOnNewHeight(height) {
        return this.size.partitionTree.width + height + this.size.solverTree.width +
            this.size.n2matrix.margin * 2;
    }

    calcHeightBasedOnNewWidth(width) {
        return width - this.size.partitionTree.width - this.size.solverTree.width -
            this.size.n2matrix.margin * 2;
    }

    calcFitDims() {
        let height = window.innerHeight * 0.95;
        let width = this.calcWidthBasedOnNewHeight(height);

        if (width > window.innerWidth - 200) {
            width = window.innerWidth - 200;
            height = this.calcHeightBasedOnNewWidth(width);
        }

        return { 'width': width, 'height': height };
    }
}
