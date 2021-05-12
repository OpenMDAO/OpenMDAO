/**
 * Handle input events for the matrix and toolbar.
 * @typedef N2UserInterface
 * @property {N2Diagram} n2Diag Reference to the main diagram.
 * @property {N2TreeNode} leftClickedNode The last node that was left-clicked.
 * @property {N2TreeNode} rightClickedNode The last node that was right-clicked, if any.
 * @property {Boolean} lastClickWasLeft True is last mouse click was left, false if right.
 * @property {Boolean} leftClickIsForward True if the last node clicked has a greater depth
 *  than the current zoomed element.
 * @property {Array} backButtonHistory The stack of forward-navigation zoomed elements.
 * @property {Array} forwardButtonHistory The stack of backward-navigation zoomed elements.
 */

class N2UserInterface {
    /**
     * Initialize properties, set up the collapse-depth menu, and set up other
     * elements of the toolbar.
     * @param {N2Diagram} n2Diag A reference to the main diagram.
     */
    constructor(n2Diag) {
        this.n2Diag = n2Diag;

        this.leftClickedNode = document.getElementById('ptN2ContentDivId');
        this.rightClickedNode = null;
        this.lastClickWasLeft = true;
        this.leftClickIsForward = true;
        this.findRootOfChangeFunction = null;
        this.callSearchFromEnterKeyPressed = false;
        this.desVars = true;

        this.backButtonHistory = [];
        this.forwardButtonHistory = [];

        this._setupCollapseDepthSlider();
        this.updateClickedIndices();
        this._setupSearch();
        this._setupResizerDrag();
        this._setupWindowResizer();

        this.legend = new N2Legend(this.n2Diag.modelData);
        this.nodeInfoBox = new NodeInfo(this);
        this.toolbar = new N2Toolbar(this);

        // Add listener for reading in a saved view.
        let self = this;
        let n2diag = this.n2Diag;
        document.getElementById('state-file-input').addEventListener('change', function () {

            var fr = new FileReader();
            fr.onload = function () {
                let dataDict = false;

                try {
                    dataDict = JSON.parse(fr.result);
                }
                catch (error) {
                    alert("Cannot load view. The file does not appear to be a valid view file.");
                    return;
                }

                if (!dataDict.md5_hash) {
                    alert("Cannot load view. The file does not appear to be a valid view file.");
                    return;
                }

                // Make sure model didn't change.
                if (dataDict.md5_hash != n2diag.model.md5_hash) {
                    alert("Cannot load view. Current model structure is different than in saved view.")
                    return;
                }

                self.addBackButtonHistory();

                // Solver toggle state.
                n2diag.showLinearSolverNames = dataDict.showLinearSolverNames;
                n2diag.ui.setSolvers(dataDict.showLinearSolverNames);
                n2diag.showSolvers = dataDict.showSolvers;

                // Zoomed node (subsystem).
                n2diag.zoomedElement = n2diag.findNodeById(dataDict.zoomedElement);

                // Expand/Collapse state of all nodes (subsystems) in model.
                n2diag.setSubState(dataDict.expandCollapse.reverse());

                // Force an immediate display update.
                // Needed to do this so that the arrows don't slip in before the element zoom.
                n2diag.layout = new N2Layout(n2diag.model, n2diag.zoomedElement,
                    n2diag.showLinearSolverNames, n2diag.showSolvers, n2diag.dims);
                n2diag.ui.updateClickedIndices();
                n2diag.matrix = new N2Matrix(n2diag.model, n2diag.layout,
                    n2diag.dom.n2Groups, n2diag.arrowMgr, n2diag.ui.lastClickWasLeft,
                    n2diag.ui.findRootOfChangeFunction, n2diag.matrix.nodeSize);
                n2diag._updateScale();
                n2diag.layout.updateTransitionInfo(n2diag.dom, n2diag.transitionStartDelay, n2diag.manuallyResized);

                // Arrow State
                n2diag.arrowMgr.loadPinnedArrows(dataDict.arrowState);
            }
            fr.readAsText(this.files[0]);
        })
    }

    /**
     * Set the range and current value of the collapse depth slider.
     * @param {Number} opts.min The shallowest elements to allow to collapse via slider.
     * @param {Number} opts.max The deepest elements to allow to collapse via slider.
     * @param {Number} opts.val The current slider collapse depth value.
     */
    setCollapseDepthSlider(opts = {}) {
        const min = opts.min ? opts.min : 2;
        const max = opts.max ? opts.max : this.n2Diag.model.maxDepth;
        const val = opts.val ? opts.val : max;

        this.collapseDepthSlider
            .property('min', min)
            .property('max', max)
            .property('value', val);

        this.collapseDepthLabel.text(val);
    }

    get collapseDepthSliderVal() { return Number(this.collapseDepthSlider.property('value')); }

    /** Set up the menu for selecting an arbitrary depth to collapse to. */
    _setupCollapseDepthSlider() {
        const self = this;

        this.collapseDepthSlider = d3.select('input#depth-slider'),
            this.collapseDepthLabel = d3.select('p#depth-slider-label');

        this.setCollapseDepthSlider();

        this.collapseDepthSlider
            .on('mouseup', e => {
                const val = self.collapseDepthSlider.property('value');
                self.collapseToDepth(val);
            })
            .on('input', e => {
                self.collapseDepthLabel.text(self.collapseDepthSliderVal);
            });
    }

    /** Set up event handlers for grabbing the bottom corner and dragging */
    _setupResizerDrag() {
        const handle = d3.select('#n2-resizer-handle');
        const box = d3.select('#n2-resizer-box');
        const body = d3.select('body');

        handle.on('mousedown', e => {
            box
                .style('top', n2Diag.layout.gapSpace)
                .style('bottom', n2Diag.layout.gapSpace);

            handle.attr('class', 'active-resizer-handle');
            box.attr('class', 'active-resizer-box');

            const startPos = {
                'x': d3.event.clientX,
                'y': d3.event.clientY
            };
            const startDims = {
                'width': parseInt(box.style('width')),
                'height': parseInt(box.style('height'))
            };
            const offset = {
                'x': startPos.x - startDims.width,
                'y': startPos.y - startDims.height
            };
            let newDims = {
                'x': startDims.width,
                'y': startDims.height
            };

            handle.html(Math.round(newDims.x) + ' x ' + newDims.y);

            body.style('cursor', 'nwse-resize')
                .on('mouseup', e => {
                    n2Diag.manuallyResized = true;

                    // Update the slider value and display
                    const defaultHeight = window.innerHeight * .95;
                    const newPercent = Math.round((newDims.y / defaultHeight) * 100);
                    d3.select('#model-slider').node().value = newPercent;
                    d3.select('#model-slider-label').html(newPercent + "%");

                    // Perform the actual resize
                    n2Diag.verticalResize(newDims.y);

                    box.style('width', null).style('height', null);

                    // Turn off the resizing box border and handle
                    if (n2Diag.showSolvers) {
                        handle.attr('class', 'inactive-resizer-handle');
                    } else {
                        handle.attr('class', 'inactive-resizer-handle-without-solvers');
                    }
                    box.attr('class', 'inactive-resizer-box');

                    // Get rid of the drag event handlers
                    body.style('cursor', 'default')
                        .on('mousemove', null)
                        .on('mouseup', null);
                })
                .on('input', e => {
                    const newHeight = d3.event.clientY - offset.y;
                    if (newHeight + n2Diag.layout.gapDist * 2 >= window.innerHeight * .5) {
                        newDims = {
                            'x': d3.event.clientX - offset.x,
                            'y': newHeight
                        };

                        // Maintain the ratio by only resizing in the least moved direction
                        // and resizing the other direction by a fraction of that
                        if (newDims.x < newDims.y) {
                            newDims.y = n2Diag.layout.calcHeightBasedOnNewWidth(newDims.x);
                        }
                        else {
                            newDims.x = n2Diag.layout.calcWidthBasedOnNewHeight(newDims.y);
                        }

                        box
                            .style('width', newDims.x + 'px')
                            .style('height', newDims.y + 'px');

                        handle.html(Math.round(newDims.x) + ' x ' + newDims.y);
                    }
                });

            d3.event.preventDefault();
        });

    }

    /** Respond to window resize events if the diagram hasn't been manually sized */
    _setupWindowResizer() {
        const self = this;
        const n2Diag = self.n2Diag;
        this.pixelRatio = window.devicePixelRatio;

        self.resizeTimeout = null;
        d3.select(window).on('resize', function () {
            const newPixelRatio = window.devicePixelRatio;

            // If the browser window itself is zoomed, don't do anything
            if (newPixelRatio != self.pixelRatio) {
                self.pixelRatio = newPixelRatio;
                return;
            }

            if (!n2Diag.manuallyResized) {
                clearTimeout(self.resizeTimeout);
                self.resizeTimeout =
                    setTimeout(function () {
                        n2Diag.verticalResize(window.innerHeight * .95);
                    }, 200);
            }
        })
    }

    /**
     * Make sure the clicked node is deeper than the zoomed node, that
     * it's not the root node, and that it actually has children.
     * @param {N2TreeNode} node The right-clicked node to check.
     */
    isCollapsible(node) {
        return (node.depth > this.n2Diag.zoomedElement.depth &&
            node.type !== 'root' && node.hasChildren());
    }

    /**
     * When a node is right-clicked or otherwise targeted for collapse, make sure it
     * it's allowed, then set the node as minimized and update the diagram drawing.
     */
    collapse() {
        testThis(this, 'N2UserInterface', 'collapse');

        let node = this.rightClickedNode;

        if (this.isCollapsible(node)) {

            if (this.collapsedRightClickNode !== undefined) {
                this.rightClickedNode = this.collapsedRightClickNode;
                this.collapsedRightClickNode = undefined;
            }

            this.findRootOfChangeFunction =
                this.findRootOfChangeForRightClick.bind(this);

            N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
            this.lastClickWasLeft = false;
            node.minimize();
            this.n2Diag.update();
        }
    }

    /**
     * When a node is right-clicked, collapse it if it's allowed.
     * @param {N2TreeNode} node The node that was right-clicked.
     */
    rightClick(node) {
        testThis(this, 'N2UserInterface', 'rightClick');

        d3.event.preventDefault();
        d3.event.stopPropagation();

        if (node.isMinimized) {
            this.rightClickedNode = node;
            this.addBackButtonHistory();
            node.manuallyExpanded = true;
            this._uncollapse(node);
            this.n2Diag.update();
        }
        else if (this.isCollapsible(node)) {
            this.rightClickedNode = node;
            node.collapsable = true;

            this.addBackButtonHistory();
            node.manuallyExpanded = false;
            this.collapse();
        }
    }

    /**
     * Update states as if a left-click was performed, which may or may not have
     * actually happened.
     * @param {N2TreeNode} node The node that was targetted.
     */
    _setupLeftClick(node) {
        this.leftClickedNode = node;
        this.lastClickWasLeft = true;
        if (this.leftClickedNode.depth > this.n2Diag.zoomedElement.depth) {
            this.leftClickIsForward = true; // forward
        }
        else if (this.leftClickedNode.depth < this.n2Diag.zoomedElement.depth) {
            this.leftClickIsForward = false; // backwards
        }
        this.n2Diag.updateZoomedElement(node);
        N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
    }

    /**
     * React to a left-clicked node by zooming in on it.
     * @param {N2TreeNode} node The targetted node.
     */
    leftClick(node) {
        // Don't do it if the node is already zoomed
        if (node === this.n2Diag.zoomedElement) return;

        testThis(this, 'N2UserInterface', 'leftClick');
        d3.event.preventDefault();
        d3.event.stopPropagation();

        if (!node.hasChildren() || node.isInput()) return;
        if (d3.event.button != 0) return;
        this.addBackButtonHistory();
        node.expand();
        node.manuallyExpanded = true;
        this._setupLeftClick(node);

        this.n2Diag.update();
    }

    /**
     * Set up for an animated transition by setting and remembering where things were.
     */
    updateClickedIndices() {
        enterIndex = exitIndex = 0;

        if (this.lastClickWasLeft) {
            let lcRootIndex = (!this.leftClickedNode || !this.leftClickedNode.rootIndex) ? 0 :
                this.leftClickedNode.rootIndex;

            if (this.leftClickIsForward) {
                exitIndex = lcRootIndex - this.n2Diag.zoomedElementPrev.rootIndex;
            }
            else {
                enterIndex = this.n2Diag.zoomedElementPrev.rootIndex - lcRootIndex;
            }
        }
    }

    /**
     * Preserve the current zoomed element and state of all hidden elements.
     * @param {Boolean} clearForward If true, erase the forward history.
     */
    addBackButtonHistory(clearForward = true) {
        let formerHidden = [];
        this.n2Diag.findAllHidden(formerHidden, false);

        this.backButtonHistory.push({
            'node': this.n2Diag.zoomedElement,
            'hidden': formerHidden,
            'search': this.toolbar.getSearchState(),
            'collapseDepth': this.currentCollapseDepth
        });

        if (clearForward) this.forwardButtonHistory = [];
    }

    /**
     * Preserve the specified node as the zoomed element,
     * and remember the state of all hidden elements.
     * @param {N2TreeNode} node The node to preserve as the zoomed element.
     */
    addForwardButtonHistory(node) {
        let formerHidden = [];
        this.n2Diag.findAllHidden(formerHidden, true);

        this.forwardButtonHistory.push({
            'node': node,
            'hidden': formerHidden,
            'search': this.toolbar.getSearchState(),
            'collapseDepth': this.currentCollapseDepth
        });
    }

    /**
     * When the back history button is clicked, pop the top node from that
     * history stack, and disable the button if the stack is empty. Find the
     * neared un-minimized node (if not the node itself) and zoom to that.
     * Add the previous zoomed node to the forward history stack.
     */
    backButtonPressed() {
        testThis(this, 'N2UserInterface', 'backButtonPressed');

        if (this.backButtonHistory.length == 0) {
            debugInfo("backButtonPressed(): no items in history");
            return;
        }

        debugInfo("backButtonPressed(): " +
            this.backButtonHistory.length + " items in history");

        const history = this.backButtonHistory.pop();
        const node = history.node;

        this.toolbar.setSearchState(history.search);
        this.setCollapseDepthSlider({ 'val': history.collapseDepth });

        // Check to see if the node is a collapsed node or not
        if (node.collapsable) {
            this.leftClickedNode = node;
            this.addForwardButtonHistory(node);
            this.collapse();
        }
        else {
            for (let obj = node; obj != null; obj = obj.parent) {
                //make sure history item is not minimized
                if (obj.isMinimized) return;
            }

            this.addForwardButtonHistory(this.n2Diag.zoomedElement);
            this._setupLeftClick(node);
        }

        this.n2Diag.resetAllHidden(history.hidden);
        this.n2Diag.update();
    }

    /**
     * When the forward history button is clicked, pop the top node from that
     * history stack, and disable the button if the stack is empty. Find the
     * neared un-minimized node (if not the node itself) and zoom to that.
     * Add the previous zoomed node to the back history stack.
     */
    forwardButtonPressed() {
        testThis(this, 'N2UserInterface', 'forwardButtonPressed');

        if (this.forwardButtonHistory.length == 0) {
            debugInfo("forwardButtonPressed(): no items in history");
            return;
        }

        debugInfo("forwardButtonPressed(): " +
            this.forwardButtonHistory.length + " items in history");

        const history = this.forwardButtonHistory.pop();
        const node = history.node;

        this.toolbar.setSearchState(history.search);
        this.setCollapseDepthSlider({ 'val': history.collapseDepth });

        d3.select('#redo-graph').classed('disabled-button',
            (this.forwardButtonHistory.length == 0));

        for (let obj = node; obj != null; obj = obj.parent) {
            // make sure history item is not minimized
            if (obj.isMinimized) return;
        }

        this.addBackButtonHistory(false);
        this._setupLeftClick(node);

        this.n2Diag.resetAllHidden(history.hidden);
        this.n2Diag.update();
    }

    /**
     * When the last event to change the zoom level was a right-click,
     * return the targetted node. Called during drawing/transition.
     * @returns The last right-clicked node.
     */
    findRootOfChangeForRightClick() {
        return this.rightClickedNode;
    }

    /**
     * When the last event to change the zoom level was the selection
     * from the collapse depth menu, return the node with the
     * appropriate depth.
     * @returns The node that has the selected depth if it exists.
     */
    findRootOfChangeForCollapseDepth(node) {
        for (let obj = node; obj != null; obj = obj.parent) {
            //make sure history item is not minimized
            if (obj.depth == this.n2Diag.chosenCollapseDepth) return obj;
        }
        return node;
    }

    /**
     * When either of the collapse or uncollapse toolbar buttons are
     * pressed, return the parent component of the targetted node if
     * it has one, or the node itself if not.
     * @returns Parent component of output node or node itself.
     */
    findRootOfChangeForCollapseUncollapseOutputs(node) {
        return node.hasOwnProperty('parentComponent') ?
            node.parentComponent :
            node;
    }

    /**
     * When the home button (aka return-to-root) button is clicked, zoom
     * to the root node.
     */
    homeButtonClick() {
        testThis(this, 'N2UserInterface', 'homeButtonClick');

        this.leftClickedNode = this.n2Diag.model.root;
        this.lastClickWasLeft = true;
        this.leftClickIsForward = false;
        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        this.addBackButtonHistory();
        this.setCollapseDepthSlider();

        this.n2Diag.reset();
    }

    /**
     * Minimize the specified node and recursively minimize its children.
     * @param {N2TreeNode} node The current node to operate on.
     */
    _collapseOutputs(node) {
        if (node.subsystem_type && node.subsystem_type == 'component') {
            node.isMinimized = true;
        }
        if (node.hasChildren()) {
            for (let child of node.children) {
                this._collapseOutputs(child);
            }
        }
    }

    /**
     * React to a button click and collapse all outputs of the specified node.
     * @param {N2TreeNode} node The initial node, usually the currently zoomed element.
     */
    collapseOutputsButtonClick(startNode) {
        testThis(this, 'N2UserInterface', 'collapseOutputsButtonClick');

        this.addBackButtonHistory();
        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this._collapseOutputs(startNode);
        this.n2Diag.update();
    }

    /**
     * Mark this node and all of its children as unminimized/unhidden
     * @param {N2TreeNode} node The node to operate on.
     */
    _uncollapse(node) {
        node.expand();
        node.varIsHidden = false;

        if (node.hasChildren()) {
            for (let child of node.children) {
                this._uncollapse(child);
            }
        }
    }

    /**
     * React to a button click and uncollapse the specified node.
     * @param {N2TreeNode} startNode The initial node.
     */
    uncollapseButtonClick(startNode) {
        testThis(this, 'N2UserInterface', 'uncollapseButtonClick');

        this.addBackButtonHistory();
        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this._uncollapse(startNode);
        startNode.manuallyExpanded = true;
        this.n2Diag.update();
    }

    /** Any collapsed nodes are expanded, starting with the specified node. */
    expandAll(startNode) {
        testThis(this, 'N2UserInterface', 'expandAll');

        this.n2Diag.showWaiter();

        this.addBackButtonHistory();
        this.n2Diag.manuallyExpandAll(startNode);

        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this.n2Diag.update();
    }

    /** All nodes are collapsed, starting with the specified node. */
    collapseAll(startNode) {
        testThis(this, 'N2UserInterface', 'collapseAll');

        this.addBackButtonHistory();
        this.n2Diag.minimizeAll(startNode);

        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this.n2Diag.update();
    }

    /**
     * React to a new selection in the collapse-to-depth toolbar slider.
     * @param {Number} depth Selected depth to collapse to.
     */
    collapseToDepth(depth) {
        testThis(this, 'N2UserInterface', 'collapseToDepth');

        this.addBackButtonHistory();
        this.n2Diag.minimizeToDepth(depth);
        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseDepth.bind(
            this
        );
        N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this.n2Diag.update();
    }

    /**
     * Wipe the current solvers legend area and populate with the other type.
     * @param {Boolean} linear True to use linear solvers, false for non-linear.
     */
    setSolvers(linear) {

        // Update the diagram
        this.n2Diag.showLinearSolverNames = linear;

        // update the legend
        this.legend.toggleSolvers(this.n2Diag.showLinearSolverNames);

        if (this.legend.shown)
            this.legend.show(
                this.n2Diag.showLinearSolverNames,
                this.n2Diag.style.solvers
            );
        this.n2Diag.update();
    }

    /**
     * React to the toggle-solver-name button press and show non-linear if linear
     * is currently shown, and vice-versa.
     */
    showSolvers() {
        // d3.select('#solver_tree').style('display','block');
        n2Diag.showSolvers = true;
        this.n2Diag.update();
        d3.select('#n2-resizer-handle').attr('class', 'inactive-resizer-handle')
    }
    hideSolvers() {
        // d3.select('#solver_tree').style('display','none');
        // d3.select('#solver_tree').attr('width',0);
        n2Diag.showSolvers = false;
        this.n2Diag.update();
        // const handle = d3.select('#n2-resizer-handle');
        d3.select('#n2-resizer-handle').attr('class', 'inactive-resizer-handle-without-solvers')
        // n2-resizer-handle
    }

    /** React to the toggle legend button, and show or hide the legend below the N2. */
    toggleLegend() {
        testThis(this, 'N2UserInterface', 'toggleLegend');
        this.legend.toggle();

        d3.select('#legend-button').attr('class',
            this.legend.hidden ? 'fas icon-key' : 'fas icon-key active-tab-icon');
    }

    toggleDesVars() {
        testThis(this, 'N2UserInterface', 'toggleDesVars');

        if (this.desVars) {
            this.n2Diag.showDesignVars();
            this.desVars = false;
        } else {
            this.n2Diag.hideDesignVars();
            this.desVars = true;
        }

        d3.select('#desvars-button').attr('class',
            this.desVars ? 'fas icon-fx-2' : 'fas icon-fx-2 active-tab-icon');
    }

    /** Show or hide the node info panel button */
    toggleNodeData() {
        testThis(this, 'N2UserInterface', 'toggleNodeData');

        const infoButton = d3.select('#info-button');
        const nodeData = d3.select('#node-info-table');

        if (nodeData.classed('info-hidden')) {
            nodeData.attr('class', 'info-visible');
            infoButton.attr('class', 'fas icon-info-circle active-tab-icon');
        }
        else {
            nodeData.attr('class', 'info-hidden');
            infoButton.attr('class', 'fas icon-info-circle');
        }
    }

    _setupSearch() {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        // Keyup so it will be after the input and awesomplete-selectcomplete event listeners
        window.addEventListener(
            'keyup',
            self.searchEnterKeyUpEventListener.bind(self),
            true
        );

        // Keydown so it will be before the input and awesomplete-selectcomplete event listeners
        window.addEventListener(
            'keydown',
            self.searchEnterKeyDownEventListener.bind(self),
            true
        );
    }

    /** Make sure UI controls reflect history and current reality. */
    update() {
        testThis(this, 'N2UserInterface', 'update');

        this.currentCollapseDepth = this.collapseDepthSliderVal;

        d3.select('#undo-graph').classed('disabled-button',
            (this.backButtonHistory.length == 0));
        d3.select('#redo-graph').classed('disabled-button',
            (this.forwardButtonHistory.length == 0));
    }

    /** Called when the search button is actually or effectively clicked to start a search. */
    searchButtonClicked() {
        testThis(this, 'N2UserInterface', 'searchButtonClicked');
        this.addBackButtonHistory();
        this.n2Diag.search.performSearch();

        this.findRootOfChangeFunction = this.n2Diag.search.findRootOfChangeForSearch;
        N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this.n2Diag.search.updateRecomputesAutoComplete = false;
        this.n2Diag.update();
    }

    /**
     * Called when the enter key is pressed in the search input box.
     * @param {Event} e Object with information about the event.
     */
    searchEnterKeyDownEventListener(e) {
        testThis(this, 'N2UserInterface', 'searchEnterKeyDownEventListener');

        let target = e.target;
        if (target.id == 'awesompleteId') {
            let key = e.which || e.keyCode;
            if (key === 13) {
                // 13 is enter
                this.callSearchFromEnterKeyPressed = true;
            }
        }
    }

    searchEnterKeyUpEventListener(e) {
        testThis(this, 'N2UserInterface', 'searchEnterKeyUpEventListener');

        let target = e.target;
        if (target.id == 'awesompleteId') {
            let key = e.which || e.keyCode;
            if (key == 13) {
                // 13 is enter
                if (this.callSearchFromEnterKeyPressed) {
                    this.searchButtonClicked();
                }
            }
        }
    }

    /** Save the model state to a file. */
    saveState() {
        const stateFileName = prompt("Filename to save view state as", 'saved.n2view');

        // Solver toggle state.
        let showLinearSolverNames = this.n2Diag.showLinearSolverNames;
        let showSolvers = this.n2Diag.showSolvers;

        // Zoomed node (subsystem).
        let zoomedElement = this.n2Diag.zoomedElement.id;

        // Expand/Collapse state of all nodes (subsystems) in model.
        let expandCollapse = Array()
        this.n2Diag.getSubState(expandCollapse);

        // Arrow State
        let arrowState = this.n2Diag.arrowMgr.savePinnedArrows();

        let dataDict = {
            'showLinearSolverNames': showLinearSolverNames,
            'showSolvers': showSolvers,
            'zoomedElement': zoomedElement,
            'expandCollapse': expandCollapse,
            'arrowState': arrowState,
            'md5_hash': this.n2Diag.model.md5_hash,
        };

        let link = document.createElement('a');
        link.setAttribute('download', stateFileName);
        let data_blob = new Blob([JSON.stringify(dataDict)],
            { type: 'text/plain' });

        // If we are replacing a previously generated file we need to
        // manually revoke the object URL to avoid memory leaks.
        if (stateFileName !== null) {
            window.URL.revokeObjectURL(stateFileName);
        }

        link.href = window.URL.createObjectURL(data_blob);
        document.body.appendChild(link);

        // wait for the link to be added to the document
        window.requestAnimationFrame(function () {
            var event = new MouseEvent('click');
            link.dispatchEvent(event);
            document.body.removeChild(link);
        })
    }

    /** Load the model state to a file. */
    loadState() {
        document.getElementById('state-file-input').click();
    }


}
