// <<hpp_insert gen/Layout.js>>
// <<hpp_insert gen/ClickHandler.js>>
// <<hpp_insert gen/UserInterface.js>>
// <<hpp_insert gen/ArrowManager.js>>
// <<hpp_insert gen/Search.js>>
// <<hpp_insert gen/Matrix.js>>
// <<hpp_insert gen/Style.js>>

/**
 * Manage all pieces of the application. The model data, the CSS styles, the
 * user interface, the layout of the matrix, and the matrix grid itself are
 * all member objects.
 * @typedef Diagram
 * @property {ModelData} model Processed model data received from Python.
 * @property {Style} style Manages diagram-related styles and functions.
 * @property {Layout} layout Sizes and positions of visible elements.
 * @property {Matrix} matrix Manages the grid of visible model parameters.
 * @property {TreeNode} zoomedElement The element the diagram is currently based on.
 * @property {TreeNode} zoomedElementPrev Reference to last zoomedElement.
 * @property {Object} dom Container for references to web page elements.
 * @property {Object} dom.svgDiv The div containing the SVG element.
 * @property {Object} dom.svg The SVG element.
 * @property {Object} dom.svgStyle Object where SVG style changes can be made.
 * @property {Object} dom.toolTip Div to display tooltips.
 * @property {Object} dom.diagOuterGroup The outermost div of the diagram itself.
 * @property {Object} dom.diagGroups References to <g> SVG elements.
 * @property {number} chosenCollapseDepth The selected depth from the drop-down.
 */
class Diagram {
    /**
     * Set initial values.
     * @param {Object} modelJSON The decompressed model structure.
     * @param {Boolean} [callInit = true] Whether to run _init() now.
     */
    constructor(modelJSON, callInit = true) {
        this.modelData = modelJSON;
        this._newModelData();
        this.zoomedElement = this.zoomedElementPrev = this.model.root;
        this.manuallyResized = false; // If the diagram has been sized by the user

        // Assign this way because defaultDims is read-only.
        this.dims = {...defaultDims};

        this._referenceD3Elements();
        this.transitionStartDelay = transitionDefaults.startDelay;
        this.chosenCollapseDepth = -1;

        this.search = new Search(this.zoomedElement, this.model.root);
        this.arrowMgr = new ArrowManager(this.dom.diagGroups);

        if (callInit) { this._init(); }
    }

    _newModelData() {
        this.model = new ModelData(this.modelData);
    }

    /** Create a Layout object. Can be overridden to create different types of Layouts */
    _newLayout() {
        return new Layout(this.model, this.zoomedElement, this.dims);
    }

    /** Create a Matrix object. Can be overridden by subclasses */
    _newMatrix(lastClickWasLeft, prevCellSize = null) {
        return new Matrix(this.model, this.layout, this.dom.diagGroups,
            this.arrowMgr, lastClickWasLeft, this.ui.findRootOfChangeFunction, prevCellSize);
    }

    /**
     * Separate these calls from the constructor so that subclasses can
     * set values before execution.
     */
    _init() {
        this.style = new Style(this.dom.svgStyle, this.dims.size.font);
        this.layout = this._newLayout();
        this.ui = new UserInterface(this);
        this.matrix = this._newMatrix(true);
    }

    /**
     * Setup internal references to D3 objects so we can avoid running
     * d3.select() over and over later.
     */
     _referenceD3Elements() {
        this.dom = {
            'svgDiv': d3.select("#svgDiv"),
            'svg': d3.select("#svgId"),
            'svgStyle': d3.select("#svgId style"),
            'toolTip': d3.select(".tool-tip"),
            'arrowMarker': d3.select("#arrow"),
            'diagOuterGroup': d3.select('g#n2outer'),
            'diagInnerGroup': d3.select('g#n2inner'),
            'pTreeGroup': d3.select('g#tree'),
            'highlightBar': d3.select('g#highlight-bar'),
            'diagBackgroundRect': d3.select('g#n2inner rect'),
            'waiter': d3.select('#waiting-container'),
            'clips': {
                'partitionTree': d3.select("#partitionTreeClip > rect"),
                'n2Matrix': d3.select("#n2MatrixClip > rect"),
            }
        };

        const diagGroups = {};
        this.dom.diagInnerGroup.selectAll('g').each(function () {
            const d3elem = d3.select(this);
            const name = new String(d3elem.attr('id')).replace(/n2/, '');
            diagGroups[name] = d3elem;
        })
        this.dom.diagGroups = diagGroups;

        const offgrid = {};
        this.dom.diagOuterGroup.selectAll('g.offgridLabel').each(function () {
            const d3elem = d3.select(this);
            const name = new String(d3elem.attr('id')).replace(/n2/, '');
            offgrid[name] = d3elem;
        })
        this.dom.diagGroups.offgrid = offgrid;
    }

    /**
     * Save the SVG to a filename selected by the user.
     * TODO: Use a proper file dialog instead of a simple prompt.
     */
     saveSvg() {
        let svgData = this.dom.svg.node().outerHTML;

        // Add name spaces.
        if (!svgData.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) {
            svgData = svgData.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
        }
        if (!svgData.match(/^<svg[^>]+"http\:\/\/www\.w3\.org\/1999\/xlink"/)) {
            svgData = svgData.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
        }

        // Add XML declaration
        svgData = '<?xml version="1.0" standalone="no"?>\r\n' + svgData;

        svgData = vkbeautify.xml(svgData);
        const svgBlob = new Blob([svgData], {
            type: "image/svg+xml;charset=utf-8"
        });
        const svgUrl = URL.createObjectURL(svgBlob);
        const downloadLink = document.createElement("a");
        downloadLink.href = svgUrl;

        // To suggest a filename to save as, get the basename of the current HTML file,
        // remove the .html/.htm extension, and add ".svg".
        const svgFileName = basename() + ".svg";
        downloadLink.download = svgFileName;
        
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    }

    /**
     * Recurse and pull state info from model for saving.
     * @param {Array} dataList Array of objects with state info for each node.
     * @param {TreeNode} node The current node being examined.
     */
     getSubState(dataList, node = this.model.root) {
        if (node.isFilter()) return; // Ignore state for FilterNodes

        dataList.push(node.getStateForSave());

        if (node.hasChildren()) {
            for (const child of node.children) {
                this.getSubState(dataList, child);
            }
        }
    }

    /**
     * Recurse and set state info into model.
     * @param {Array} dataList Array of objects with state info for each node. 
     * @param {TreeNode} node The node currently being restored.
     */
    setSubState(dataList, node = this.model.root) {
        if (node.isFilter()) return; // Ignore state for FilterNodes

        node.setStateFromLoad(dataList.pop());

        // Get rid of any existing filters before processing children, as they'll
        // be populated while processing the state of each child node.
        if (node.hasFilters()) { node.wipeFilters(); }

        if (node.hasChildren()) {
            for (const child of node.children) {
                this.setSubState(dataList, child);
            }
        }
    }

    /**
     * Replace the current zoomedElement, but preserve its value.
     * @param {Object} newZoomedElement Replacement zoomed element.
     */
    updateZoomedElement(newZoomedElement) {
        this.zoomedElementPrev = this.zoomedElement;
        this.zoomedElement = newZoomedElement;
        this.layout.zoomedElement = this.zoomedElement;
    }


    /**
     * The mouse interface to the diagram can be in normal left-click mode, node info mode,
     * expand/collapse mode, or variable filter mode. This function performs the correct callback.
     * @param {Object} obj The HTML element that was clicked on.
     * @param {TreeNode} node The node associated with the element.
     */
    leftClickSelector(e, node) {
        switch (this.ui.click.clickEffect) {
            case ClickHandler.ClickEffect.NodeInfo:
                this.ui.pinInfoBox();
                break;
            case ClickHandler.ClickEffect.Collapse:
                this.ui.rightClick(e, node, e.currentTarget);
                break;
            case ClickHandler.ClickEffect.Filter:
                const color = d3.select(e.currentTarget).select('rect').style('fill');
                this.ui.altRightClick(e, node, color);
                break;
            default:
                this.ui.leftClick(e, node);
        }
    }

    /**
     * Add SVG groups & contents coupled to the visible nodes in the model tree.
     * Select all <g> elements that have class "model_tree_grp". If any already
     * exist, join to their associated nodes in the model tree. If no
     * existing <g> matches a displayable node, add it to the "enter"
     * selection so the <g> can be created. If a <g> exists but there is
     * no longer a displayable node for it, put it in the "exit" selection so
     * it can be removed.
     */
    _updateTreeCells() {
        const self = this;
        const scale = this.layout.scales.model;
        const treeSize = this.layout.treeSize.model;

        this.dom.pTreeGroup.selectAll("g.model_tree_grp")
            .data(this.layout.zoomedNodes, d => d.id)
            .join(
                enter => self._addNewTreeCells(enter, scale, treeSize),
                update => self._updateExistingTreeCells(update, scale, treeSize),
                exit => self._removeOldTreeCells(exit, scale, treeSize)
            )
    }

    /**
     * Using the visible nodes in the model tree as data points, create SVG objects to
     * represent each one. Dimensions are obtained from the precalculated layout.
     * @param {Selection} enter The selection to add <g> elements and children to.
     * @param {Scale} scale Linear scales of the diagram width and height.
     * @param {Dimensions} treeSize Actual width and height of the tree in pixels.
     */
     _addNewTreeCells(enter, scale, treeSize) {
        const self = this; // For callbacks that might change "this".
      
        // Create a <g> for each node in zoomedNodes that doesn't already have one. Dimensions
        // are obtained from the previous geometry so the new nodes can appear to transition
        // to the new size together with the existing nodes.
        const enterSelection = enter
            .append("g")
            .attr("class", d => `model_tree_grp ${self.style.getNodeClass(d)}`)
            .attr("transform", d =>
                `translate(${scale.prev.x(d.draw.dims.prev.x)},${scale.prev.y(d.draw.dims.prev.y)})`)
            .on("click", (e,d) => self.leftClickSelector(e, d))
            .on("contextmenu", function(e,d) {
                if (e.altKey) {
                    self.ui.altRightClick(e, d, d3.select(this).select('rect').style('fill'));
                }
                else {
                    self.ui.rightClick(e, d);
                }
            })
            .on("mouseover", (e,d) => self.ui.showInfoBox(e, d))
            .on("mouseleave", () => self.ui.removeInfoBox())
            .on("mousemove", e => self.ui.moveInfoBox(e));

        enterSelection
            .transition(sharedTransition)
            .attr("transform", d =>
                `translate(${scale.x(d.draw.dims.x)},${scale.y(d.draw.dims.y)})`)

        // Add the rectangle that is the visible shape.
        enterSelection
            .append("rect")
            .attr("id", d => TreeNode.pathToId(d.path))
            .attr('rx', 12)
            .attr('ry', 12)
            .transition(sharedTransition)
            .attr("width", d => d.draw.dims.width * treeSize.width)
            .attr("height", d => d.draw.dims.height * treeSize.height);

        // Add a label
        enterSelection
            .append("text")
            .text(d => d.getTextName())
            .style('visibility', 'hidden')
            .attr("dy", ".35em")
            .attr("transform", d => {
                const anchorX = d.draw.dims.width * treeSize.width -
                    self.layout.size.rightTextMargin;
                return `translate(${anchorX}, ${(d.draw.dims.height * treeSize.height / 2)})`;
            })
            .style("opacity", d => (d.depth < self.zoomedElement.depth)? 0 : d.textOpacity)
            .transition(sharedTransition)
            .on('end', function() { d3.select(this).style('visibility', 'visible'); } )

        return enterSelection;
    }

    /**
     * Update the geometry for existing <g> with a transition.
     * @param {Selection} update The selected group of existing model tree <g> elements.
     * @param {Scale} scale Linear scales of the diagram width and height.
     * @param {Dimensions} treeSize Actual width and height of the tree in pixels.
     */
     _updateExistingTreeCells(update, scale, treeSize) {
        const self = this; // For callbacks that change "this".

        this.dom.clips.partitionTree
            .transition(sharedTransition)
            .attr('height', this.dims.size.partitionTree.height);

        // New location for each group
        const mergedSelection = update
            .attr("class", d => `model_tree_grp ${self.style.getNodeClass(d)}`)
            .transition(sharedTransition)
            .attr("transform", d => 
                `translate(${scale.x(d.draw.dims.x)} ${scale.y(d.draw.dims.y)})`);

        // Resize each rectangle
        mergedSelection
            .select("rect")
            .attr("width", d => d.draw.dims.width * treeSize.width)
            .attr("height", d => d.draw.dims.height * treeSize.height);

        // Move the text label
        mergedSelection
            .select("text")
            .attr("transform", d => {
                const anchorX = d.draw.dims.width * treeSize.width -
                    self.layout.size.rightTextMargin;
                return `translate(${anchorX} ${(d.draw.dims.height * treeSize.height/2)})`;
            })
            .style("opacity", d => (d.depth < self.zoomedElement.depth)? 0 : d.textOpacity);

        return mergedSelection;
    }

    /**
     * Remove <g> that no longer have displayable nodes associated with them, and
     * transition them away.
     * @param {Selection} exit The selected group of model tree <g> elements to remove.
     * @param {Scale} scale Linear scales of the diagram width and height.
     * @param {Dimensions} treeSize Actual width and height of the tree in pixels.
     */
    _removeOldTreeCells(exit, scale, treeSize) {
        const self = this; // For callbacks that change "this". Alternative to using .bind().

        // Transition exiting nodes to the parent's new position.
        const exitSelection = exit.transition(sharedTransition)
            .attr("transform", d => 
                `translate(${scale.x(d.draw.dims.x)} ${scale.y(d.draw.dims.y)})`);

        exitSelection.select("rect")
            .attr("width", d => d.draw.dims.width * treeSize.width)
            .attr("height", d => d.draw.dims.height * treeSize.height)

        exitSelection.select("text")
            .attr("transform", d => {
                const anchorX = d.draw.dims.width * treeSize.width -
                    self.layout.size.rightTextMargin;
                return `translate(${anchorX}, ${(d.draw.dims.height * treeSize.height / 2)})`;
            })
            .style("opacity", 0)

        exitSelection.on('end', function() {d3.select(this).remove(); })

        return exitSelection;
    }

    /** Remove all rects in the highlight bar */
    clearHighlights() {
        const selection = this.dom.highlightBar.selectAll('rect');
        const size = selection.size();
        debugInfo(`clearHighlights: Removing ${size} highlights`);
        selection.remove();
    }

    /** Remove all pinned arrows */
    clearArrows() {
        this.arrowMgr.removeAllPinned();
        this.clearHighlights();
    }

    /** Display connection arrows for all visible inputs/outputs */
    showAllArrows() {
        for (const row in this.matrix.grid) {
            const cell = this.matrix.grid[row][row]; // Diagonal cells only
            this.matrix.drawOnDiagonalArrows(cell);
            this.arrowMgr.togglePin(cell.id, true);
        }
    }

    /**
     * Sleep for the specified number of milliseconds.
     * @param {Number} time Milliseconds to return after
     * @returns {Promise} Promise that can be awaited on until the timer expires.
     */
    delay(time) {
        return new Promise(function(resolve) {
            setTimeout(resolve, time)
        });
     }

    /** Display an animation while the transition is in progress */
    showWaiter() { this.dom.waiter.attr('class', 'show'); }

    /** Hide the animation after the transition completes */
    hideWaiter() { this.dom.waiter.attr('class', 'no-show'); }

    /**
     * Refresh the diagram when something has visually changed.
     * @param {Boolean} [computeNewTreeLayout = true] Whether to rebuild the layout and
     *  matrix objects.
     */
    async update(computeNewTreeLayout = true) {
        this.showWaiter();
        await this.delay(100);

        this.ui.update();
        this.search.update(this.zoomedElement, this.model.root);

        // Compute the new tree layout if necessary.
        if (computeNewTreeLayout) {
            this.layout = this._newLayout();
            this.ui.updateClickedIndices();
            this.matrix = this._newMatrix(this.ui.lastClickWasLeft, this.matrix.nodeSize);
        }

        this.layout.updateGeometry(this.dom);
        this.layout.updateTransitionInfo(this.dom, this.transitionStartDelay, this.manuallyResized);
        this._updateTreeCells();
        this.arrowMgr.transition(this.matrix);
        this.matrix.draw();

        if (!d3.selection.prototype.transitionAllowed) this.hideWaiter();
    }

    /**
     * Updates the intended dimensions of the diagrams and font, but does
     * not perform rendering itself.
     * @param {number} height The base height of the diagram without margins.
     * @param {number} fontSize The new size of the font.
     */
     updateSizes(height, fontSize) {
        let gapSize = fontSize + 4;

        this.dims.size.matrix.margin = gapSize;
        this.dims.size.partitionTreeGap = gapSize;

        this.dims.size.matrix.height =
            this.dims.size.matrix.width = // Match base height, keep it looking square
            this.dims.size.partitionTree.height = height;

        this.dims.size.font = fontSize;
    }

    /**
     * Adjust the height and corresponding width of the diagram based on user input.
     * @param {number} height The new height in pixels.
     */
     verticalResize(height) {
        // Don't resize if the height didn't actually change:
        if (this.dims.size.partitionTree.height == height) return;

        if (!this.manuallyResized) {
            height = this.layout.calcFitDims().height;
        }

        this.updateSizes(height, this.dims.size.font);

        transitionDefaults.duration = transitionDefaults.durationFast;
        this.update();
    }

    /**
     * Adjust the font size of all text in the diagram based on user input.
     * @param {number} fontSize The new font size in pixels.
     */
     fontSizeSelectChange(fontSize) {
        transitionDefaults.duration = transitionDefaults.durationFast;
        this.style.updateSvgStyle(fontSize);
        this.update();
    }

    /**
     * Since the matrix can be destroyed and recreated, use this to invoke the callback
     * rather than setting one up that points directly to a specific matrix.
     * @param {MatrixCell} cell The cell the event occured on.
     */
    mouseOverOnDiagonal(e, cell) {
        if (this.matrix.cellExists(cell)) {
            this.matrix.mouseOverOnDiagonal(cell);
            this.ui.showInfoBox(e, cell.obj, cell.color(), false);
        }
    }

    /**
     * Move the node info panel around if it's visible
     * @param {MatrixCell} cell The cell the event occured on.
     */
    mouseMoveOnDiagonal(e, cell) {
        if (this.matrix.cellExists(cell)) {
            this.ui.moveInfoBox(e);
        }
    }

    /**
     * Since the matrix can be destroyed and recreated, use this to invoke the callback
     * rather than setting one up that points directly to a specific matrix.
     */
    mouseOverOffDiagonal(e, cell) {
        if (this.matrix.cellExists(cell)) {
            this.matrix.mouseOverOffDiagonal(cell);
            this.ui.showCellInfoBox(e, cell, cell.color(), true);
        }
    }

    /** When the mouse leaves a cell, remove all temporary arrows and highlights. */
    mouseOut() {
        this.arrowMgr.removeAllHovered();
        this.clearHighlights();
        d3.selectAll("div.offgrid").style("visibility", "hidden").html('');

        this.ui.removeInfoBox();
    }

    /**
     * When the mouse is left-clicked on a cell, change their CSS class
     * so they're not removed when the mouse moves out. Or, if in info panel
     * mode, pin the info panel.
     * @param {MatrixCell} cell The cell the event occured on.
     */
    mouseClick(cell) {
        if (! this.ui.click.isNodeInfo) { // If not in info-panel mode, pin/unpin arrows
            this.arrowMgr.togglePin(cell.id);
        }
        else { // Make a persistent info panel
            this.ui.pinInfoBox();
        }
    }

    /**
     * Place member mouse callbacks in an object for easy reference.
     * @returns {Object} Object containing each of the functions.
     */
    getMouseFuncs() {
        const self = this;

        const mf = {
            'overOffDiag': self.mouseOverOffDiagonal.bind(self),
            'overOnDiag': self.mouseOverOnDiagonal.bind(self),
            'moveOnDiag': self.mouseMoveOnDiagonal.bind(self),
            'out': self.mouseOut.bind(self),
            'click': self.mouseClick.bind(self)
        }

        return mf;
    }

    /**
     * Set the new depth to collapse to and perform the operation.
     * @param {Number} depth If the node's depth is the same or more, collapse it.
     */
     minimizeToDepth(depth) {
        this.chosenCollapseDepth = depth;

        if (depth > this.zoomedElement.depth)
            this.model.minimizeToDepth(this.model.root, depth);
    }

    /** Unset all manually-selected node states and zoom to the root element */
    reset() {
        this.model.resetAllHidden([]);
        this.updateZoomedElement(this.model.root);
        transitionDefaults.duration = transitionDefaults.durationFast;
        this.update();
    }

    /**
     * Using an object populated by loading and validating a JSON file, set the model
     * to the saved view.
     * @param {Object} oldState The model view to restore.
     */
     restoreSavedState(oldState) {
        // Zoomed node
        this.zoomedElement = this.model.nodeIds[oldState.zoomedElement];

        // Expand/Collapse state of all group nodes in model.
        this.setSubState(oldState.expandCollapse.reverse());

        // Force an immediate display update.
        // Needed to do this so that the arrows don't slip in before the element zoom.
        this.layout = this._newLayout();
        this.ui.updateClickedIndices();
        this.matrix = this._newMatrix(this.ui.findRootOfChangeFunction, this.matrix.nodeSize);
        this.layout.updateGeometry(this.dom);
        this.layout.updateTransitionInfo(this.dom, this.transitionStartDelay, this.manuallyResized);

        // Arrow State
        this.arrowMgr.loadPinnedArrows(oldState.arrowState);
    }
}
 
