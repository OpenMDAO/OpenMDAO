// <<hpp_insert gen/Scale.js>>
// <<hpp_insert gen/Dimensions.js>>

/**
 * Calculates and stores the size and positions of visible elements.
 * @typedef Layout
 * @property {ModelData} model Reference to the preprocessed model.
 * @property {TreeNode} zoomedElement Reference to zoomedElement managed by Diagram.
 * @property {TreeNode[]} zoomedNodes  Child workNodes of the current zoomed element.
 * @property {TreeNode[]} visibleNodes Zoomed workNodes that are actually drawn.
 * @property {Object} svg Reference to the top-level SVG element in the document.
 * @property {Object} size The dimensions of the model tree.
 * @property {Object} scales Scalers in the X and Y directions to associate the relative
 *   position of an element to actual pixel coordinates.
 * @property {Object} treeSize
 */
class Layout {
    /**
     * Compute the new layout based on the model data and the zoomed element.
     * @param {ModelData} model The pre-processed model object.
     * @param {Object} newZoomedElement The element the new layout is based around.
     * @param {Object} dims The initial sizes for multiple tree elements.
     * @param {Boolean} callInit Whether to call _init() or have a subclass do it.
     */
    constructor(model, newZoomedElement, dims, callInit = true) {
        this.model = model;

        this.zoomedElement = newZoomedElement;

        this.zoomedNodes = [];
        this.visibleNodes = [];

        this.curVisibleNodeCount = 0;
        this.prevVisibleNodeCount = 0;

        // Initial size values derived from read-only defaults
        this.size = dims.size;
        this.svg = d3.select("#svgId");

        if (callInit) this._init();
    }

    /**
     * Separate these calls from the constructor so that subclasses can
     * set values before execution.
     */
    _init() {
        this._computeLeaves();
        this._setupTextRenderer();
        this._updateTextWidths();
        delete (this.textRenderer);
        this._computeColumnWidths();
        this._setColumnLocations(this.size.partitionTree, this.cols);

        this._computeNormalizedPositions(this.model.root, 0, false, null);
        if (this.zoomedElement.parent)
            this.zoomedNodes.push(this.zoomedElement.parent);

        this.setTransitionPermission();

        this.scales = {
            'model': new Scale(this.size.partitionTree),
            'firstRun': true
        }

        this.treeSize = {
            'model': new Dimensions({ 'width': 0, 'height': 0})
        }
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
        if (highWaterMark >= transitionDefaults.maxNodes) {
            debugInfo("Denying transitions: ", this.visibleNodes.length,
                " visible nodes, max allowed: ", transitionDefaults.maxNodes)

            // Return if already denied
            if (!d3.selection.prototype.transitionAllowed) return;
            d3.selection.prototype.transitionAllowed = false;

            d3.selection.prototype.transition = returnThis;
            d3.selection.prototype.duration = returnThis;
            d3.selection.prototype.delay = returnThis;
        }
        else { // OK, enable transitions.
            debugInfo("Allowing transitions: ", this.visibleNodes.length,
                " visible nodes, max allowed: ", transitionDefaults.maxNodes)

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
     * @param {String} text Text to render or find in cache.
     * @return {Number} The SVG-computed width of the rendered string.
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

    /**
     * Determine text widths for all descendents of the specified node.
     * @param {TreeNode} [node = this.zoomedElement] Item to begin looking from.
     */
    _updateTextWidths(node = this.zoomedElement) {
        if (node.draw.hidden) return;

        node.draw.nameWidthPx = this._getTextWidth(node.getTextName()) + 2 *
            this.size.rightTextMargin;

        if (node.hasChildren() && !node.draw.minimized) {
            for (const child of node.children) {
                this._updateTextWidths(child);
            }
        }
    }

    /**
     * Recurse through the tree and add up the number of leaves that each
     * node has, based on their array of visible children.
     * @param {TreeNode} [node = this.model.root] The starting node.
     */
    _computeLeaves(node = this.model.root) {
        node.draw.numLeaves = 0;

        if (!(node.draw.hidden || node.draw.filtered)) {
            if (node.name == '_auto_ivc' && !node.draw.manuallyExpanded) {
                node.minimize();
            }
            else if (this.model.nodeIds.length > Precollapse.minimumNodes) {
                node.minimizeIfLarge(this.model.depthCount[node.depth]);
            }

            if (node.hasChildren() && !node.draw.minimized) {
                for (const child of node.children) {
                    this._computeLeaves(child);
                    node.draw.numLeaves += child.draw.numLeaves;
                }
            }
            else {
                node.draw.numLeaves = 1; // Leaf node
            }
        }
    }

    /**
     * For visible nodes with children, choose a column width
     * large enough to accomodate the widest label in their column.
     * @param {TreeNode} node The item to operate on.
     * @param {String} childrenProp Usually 'children', subclasses may use a different property.
     * @param {Object[]} colArr The array of column info.
     * @param {Number[]} leafArr The array of leaf width info.
     * @param {String} [widthProp = 'nameWidthPx'] Can be modified by derived classes.
     */
    _setColumnWidthsFromWidestText(node, childrenProp, colArr, leafArr, widthProp = 'nameWidthPx') {
        if (node.draw.hidden) return;

        const height = this.size.matrix.height * node.draw.numLeaves / this.zoomedElement.draw.numLeaves;
        node.prevTextOpacity = node.propExists('textOpacity') ? node.textOpacity : 0;
        node.textOpacity = (height > this.size.font) ? 1 : 0;
        const hasVisibleDetail = (height >= 2.0);
        let width = (hasVisibleDetail) ? this.size.minColumnWidth : 1e-3;
        if (node.textOpacity > 0.5) width = node.draw[widthProp];

        this.greatestDepth = Math.max(this.greatestDepth, node.depth);

        if (node.hasChildren(childrenProp) && !node.draw.minimized) { //not leaf
            colArr[node.depth].width = Math.max(colArr[node.depth].width, width)
            for (const child of node[childrenProp]) {
                this._setColumnWidthsFromWidestText(child, childrenProp, colArr, leafArr, widthProp);
            }
        }
        else if (!node.draw.filtered) { // leaf
            leafArr[node.depth] = Math.max(leafArr[node.depth], width);
        }
    }

    /**
     * Compute column widths across the model, then adjust ends as needed.
     * @param {TreeNode} [node = this.zoomedElement] Item to operate on.
     */
    _computeColumnWidths(node = this.zoomedElement) {
        this.greatestDepth = 0;
        this.leafWidthsPx = new Array(this.model.maxDepth + 1).fill(0.0);
        this.cols = Array.from({length: this.model.maxDepth + 1},
            () => ({ 'width': 0.0, 'location': 0.0 }));

        this._setColumnWidthsFromWidestText(node, 'children', this.cols,
            this.leafWidthsPx, 'nameWidthPx');

        let sum = 0;
        let lastColumnWidth = 0;
        for (let i = this.leafWidthsPx.length - 1; i >= this.zoomedElement.depth; --i) {
            sum += this.cols[i].width;
            const lastWidthNeeded = this.leafWidthsPx[i] - sum;
            lastColumnWidth = Math.max(lastWidthNeeded, lastColumnWidth);
        }

        this.cols[this.zoomedElement.depth - 1].width = this.size.parentNodeWidth;
        this.cols[this.greatestDepth].width = lastColumnWidth;
    }

    /**
     * Set the location of the columns based on the width of the columns to the left.
     * @param {Dimensions} obj Object containing a width property.
     * @param {Object[]} cols Array of objects containing a width property.
     */
    _setColumnLocations(obj, cols) {
        obj.width = 0;

        for (let depth = 1; depth <= this.model.maxDepth; ++depth) {
            cols[depth].location = obj.width;
            obj.width += cols[depth].width;
        }
    }

    /**
     * Recurse over the model tree and determine the coordinates and
     * size of visible nodes. If a parent is minimized, operations are
     * performed on it instead.
     * @param {TreeNode} node The node to operate on.
     * @param {number} leafCounter Tally of leaves encountered so far.
     * @param {Boolean} isChildOfZoomed Whether node is a descendant of this.zoomedElement.
     * @param {Object} earliestMinimizedParent The minimized parent, if any, appearing
     *   highest in the tree hierarchy. Null if none exist.
     */
    _computeNormalizedPositions(node, leafCounter, isChildOfZoomed, earliestMinimizedParent) {
        if (!isChildOfZoomed) {
            isChildOfZoomed = (node === this.zoomedElement);
        }

        if (earliestMinimizedParent == null && isChildOfZoomed) {
            if (node.isVisible()) {
                this.zoomedNodes.push(node)
                if (node.isVisibleLeaf()) {
                    if (!node.draw.hidden) this.visibleNodes.push(node);
                    earliestMinimizedParent = node;
                }
            }
        }

        node.preserveDims(leafCounter);
        const workNode = (earliestMinimizedParent) ? earliestMinimizedParent : node;
        const dims = node.draw.dims;

        if (! node.isVisible()) { // input or hidden leaf leaving
            dims.x = this.cols[node.parent.depth + 1].location / this.size.partitionTree.width;
            dims.y = node.parent.draw.dims.y;
            dims.width = 1e-6;
            dims.height = 1e-6;
        }
        else {
            dims.x = this.cols[workNode.depth].location / this.size.partitionTree.width;
            dims.y = leafCounter / this.model.root.draw.numLeaves;
            dims.width = (node.hasChildren() && !node.draw.minimized && !node.draw.filtered) ?
                (this.cols[workNode.depth].width / this.size.partitionTree.width) : 1 - workNode.draw.dims.x;
            dims.height = workNode.draw.numLeaves / this.model.root.draw.numLeaves;
        }

        if (node.hasChildren()) {
            for (const child of node.children) {
                if (!child.isInputOrOutput() || !child.draw.minimized) {
                    this._computeNormalizedPositions(child, leafCounter,
                        isChildOfZoomed, earliestMinimizedParent);
                    if (earliestMinimizedParent == null) { //numleaves is only valid passed nonminimized nodes
                        leafCounter += child.draw.numLeaves;
                    }
                }
            }
        }
    }

    /**
     * Calculate new dimensions for the div element enclosing the main SVG element.
     * @returns {Dimensions} Members width and height.
     */
    calcOuterDims() {
        const width = this.size.partitionTree.width + this.size.matrix.width +
            this.size.matrix.margin * 2;

        const height = this.size.matrix.height + this.size.matrix.margin * 2;

        return new Dimensions({'width': width, 'height': height});
    }

    /**
     * Calculate new dimensions for the main SVG element.
     * @returns {Dimensions} Members width, height, and margin as numbers.
     */
    calcInnerDims() {
        const width = this.size.partitionTree.width + this.size.matrix.margin +
            this.size.matrix.width + this.size.matrix.margin;

        const height = this.size.partitionTree.height;
        const margin = this.size.matrix.margin;

        return new Dimensions({'width': width, 'height': height, 'margin': margin});
    }

    /**
     * Set the geometry of the main diagram elements for the first time, without
     * using transition animations.
     * @param {Object} dom Collection of D3 selections associated with the HTML elements.
     */
    applyGeometryFirstRun(dom) {
        // Update svg dimensions before size changes
        const outerDims = this.calcOuterDims();
        const innerDims = this.calcInnerDims();
        const size = this.size;

        dom.svgDiv
            .style('width', outerDims.widthStyle)
            .style('height', outerDims.heightStyle);

        // NOTE: Apparently first setting these values with .attr and updating
        // them later with .style in .updateTransitionInfo() allows the diagram
        // transition animation to execute as intended. Setting them in both
        // places using the same method results in the transition not working.

        dom.svg
            .attr('width', outerDims.width)
            .attr('height', outerDims.height)
            .attr('transform', 'translate(0,0)');

        dom.pTreeGroup
            .attr('height', innerDims.height)
            .attr('width', size.partitionTree.width)
            .attr('transform', `translate(0,${innerDims.margin})`);

        dom.highlightBar
            .attr('height', innerDims.height)
            .attr('width', '8')
            .attr('transform', `translate(${size.partitionTree.width + 1},${innerDims.margin})`);

        dom.diagOuterGroup
            .attr('height', outerDims.height)
            .attr('width', outerDims.height)
            .attr('transform', `translate(${size.partitionTree.width},0)`);

        dom.diagInnerGroup
            .attr('height', innerDims.height)
            .attr('width', innerDims.height)
            .attr('transform', `translate(${innerDims.margin},${innerDims.margin})`);

        dom.diagBackgroundRect
            .attr('width', innerDims.height)
            .attr('height', innerDims.height)
            .attr('transform', 'translate(0,0)');

        const offgridHeight = size.font + 2;
        dom.diagGroups.offgrid.top
            .attr('transform', `translate(${innerDims.margin},0)`)
            .attr('width', innerDims.height)
            .attr('height', offgridHeight);

        dom.diagGroups.offgrid.bottom
            .attr('transform', `translate(0,${innerDims.height + offgridHeight})`)
            .attr('width', outerDims.height)
            .attr('height', offgridHeight);
    }

    /**
     * Recalculate the scale and apply new dimensions to diagram objects.
     * @param {Object} dom Collection of D3 selections associated with the HTML elements.
     * @return {Boolean} True if this was the first run.
     */
    updateGeometry(dom) {
        this.updateGeometryValues();

        if (this.scales.firstRun) { // first run, duplicate what we just calculated
            this.scales.firstRun = false;
            this.preservePreviousScaleValues();
            this.applyGeometryFirstRun(dom);

            return true;
        }

        return false;        
    }

    /**
     * Update container element dimensions when a new layout is calculated,
     * and set up transition animations.
     * @param {Object} dom References to HTML elements.
     * @param {Number} transitionStartDelay ms to wait before performing transition
     * @param {Boolean} manuallyResized Have the diagram dimensions have been changed through UI
     */
    updateTransitionInfo(dom, transitionStartDelay, manuallyResized) {
        sharedTransition = getTransition(transitionStartDelay);

        const outerDims = this.calcOuterDims();
        const u = outerDims.unit;
        const innerDims = this.calcInnerDims();

        this.ratio = (window.innerWidth - 200) / outerDims.width;
        if (this.ratio > 1 || manuallyResized) this.ratio = 1;
        else if (this.ratio < 1)
            debugInfo(`Scaling diagram to ${Math.round(this.ratio * 100)}%`);

        dom.svgDiv
            .style('width', `${outerDims.width * this.ratio}${u}`)
            .style('height',`${outerDims.height * this.ratio}${u}`)

        dom.svg.transition(sharedTransition)
            .style('transform', `scale(${this.ratio})`)
            .style('width', outerDims.widthStyle)
            .style('height', outerDims.heightStyle);

        // These two properties are used when dragging the resizer box.
        this.gapDist = (this.size.partitionTreeGap * this.ratio) - 3;
        this.gapSpace = `${this.gapDist}${u}`;
        d3.select('#n2-resizer-box').transition(sharedTransition)
            .style('bottom', this.gapSpace);

        dom.pTreeGroup.transition(sharedTransition)
            .style('height', innerDims.heightStyle)
            .style('width', `${this.size.partitionTree.width}${u}`)
            .style('transform', `translate(0,${innerDims.marginStyle})`);

        dom.highlightBar.transition(sharedTransition)
            .style('height', innerDims.heightStyle)
            .style('width', '8px')
            .style('transform',
                `translate(${this.size.partitionTree.width + 1}${u},${innerDims.marginStyle})`);

        // Move n2 outer group to right of partition tree, spaced by the margin.
        dom.diagOuterGroup.transition(sharedTransition)
            .style('height', outerDims.heightStyle)
            .style('width', outerDims.heightStyle)
            .style('transform', `translate(${this.size.partitionTree.width}${u},0)`);

        dom.diagInnerGroup.transition(sharedTransition)
            .style('height', innerDims.heightStyle)
            .style('width', innerDims.heightStyle)
            .style('transform', `translate(${innerDims.marginStyle},${innerDims.marginStyle})`);

        dom.diagBackgroundRect.transition(sharedTransition)
            .style('width', innerDims.heightStyle)
            .style('height', innerDims.heightStyle)
            .style('transform', 'translate(0,0)');
    }

    /**
     * Since the matrix portion of the diagram is square, calculate the width
     * based on the height of that, plus the model tree, plus the margins.
     * @param {Number} height Height of the diagram.
     * @returns {Number} Calculated width of the diagram.
     */
    calcWidthBasedOnNewHeight(height) {
        return this.size.partitionTree.width + height + this.size.matrix.margin * 2;
    }

    /**
     * Since the matrix portion of the diagram is square, calculate the height
     * based on the width of that, minus the model tree and margins.
     * @param {Number} width Width of the diagram.
     * @returns {Number} Calculated height of the diagram.
     */
    calcHeightBasedOnNewWidth(width) {
        return width - this.size.partitionTree.width - this.size.matrix.margin * 2;
    }

    /**
     * Calculate dimensions of the diagram based on what will fit in the window.
     * @returns {Dimensions} Object with computed width and height.
     */
    calcFitDims() {
        let height = window.innerHeight * 0.95;
        let width = this.calcWidthBasedOnNewHeight(height);

        if (width > window.innerWidth - 200) {
            width = window.innerWidth - 200;
            height = this.calcHeightBasedOnNewWidth(width);
        }
        return new Dimensions ({ 'width': width, 'height': height });
    }

    /** Make a copy of the previous dimensions and linear scalers before setting new ones. */
    preservePreviousScaleValues() {
        this.treeSize.model.preserve();
        this.scales.model.preserve();
    }

    /**
     * Calculate the dimensions of the diagram in pixels as well as updating the linear
     * scale. Preserve the previous values if there are any.
     */
    updateGeometryValues() {
        if (!this.scales.firstRun) this.preservePreviousScaleValues();

        const elemDims = this.zoomedElement.draw.dims;
        const initSize = this.size.partitionTree;

        this.treeSize.model.width = (elemDims.x ?
            initSize.width - this.size.parentNodeWidth : initSize.width) / (1 - elemDims.x);
        this.treeSize.model.height = initSize.height / elemDims.height;

        this.scales.model.x
            .domain([elemDims.x, 1])
            .range([elemDims.x ? this.size.parentNodeWidth : 0, initSize.width]);

        this.scales.model.y
            .domain([elemDims.y, elemDims.y + elemDims.height])
            .range([0, initSize.height]);
    }
}
