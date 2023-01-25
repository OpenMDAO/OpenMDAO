// <<hpp_insert gen/ClickHandler.js>>
// <<hpp_insert gen/Toolbar.js>>
// <<hpp_insert gen/ChildSelectDialog.js>>
// <<hpp_insert gen/NodeInfo.js>>
// <<hpp_insert gen/Legend.js>>

/**
 * Handle input events for the matrix and toolbar.
 * @typedef UserInterface
 * @property {Diagram} diag Reference to the main diagram.
 * @property {TreeNode} leftClickedNode The last node that was left-clicked.
 * @property {TreeNode} rightClickedNode The last node that was right-clicked, if any.
 * @property {Boolean} lastClickWasLeft True is last mouse click was left, false if right.
 * @property {Boolean} leftClickIsForward True if the last node clicked has a greater depth
 *  than the current zoomed element.
 * @property {Array} backButtonHistory The stack of forward-navigation zoomed elements.
 * @property {Array} forwardButtonHistory The stack of backward-navigation zoomed elements.
 */

class UserInterface {
    /**
     * Initialize properties, set up the collapse-depth menu, and set up other
     * elements of the toolbar.
     * @param {Diagram} diag A reference to the main diagram.
     */
    constructor(diag) {
        this.diag = diag;

        this.leftClickedNode = document.getElementById('diagram-content');
        this.rightClickedNode = null;
        this.lastClickWasLeft = true;
        this.leftClickIsForward = true;
        this.findRootOfChangeFunction = null;
        this.callSearchFromEnterKeyPressed = false;
        this.nodeInfoBox = null;

        this.backButtonHistory = [];
        this.forwardButtonHistory = [];

        this._init();
        this._setupCollapseDepthSlider();
        this.updateClickedIndices();
        this._setupSearch();
        this._setupResizerDrag();
        this._setupWindowResizer();
        this.click = new ClickHandler();


        // Add listener for reading in a saved view.
        const self = this;
        d3.select('#state-file-input').on('change', function(e) { self.loadView(e, self); });
    }

    /**
     * Separate these calls from the constructor so that subclasses can
     * set values before execution.
     */
    _init() {
        this.legend = new Legend(this.diag.modelData);
        this.toolbar = new Toolbar(this);
    }

    /**
     * Create a new node info window.
     * @param {Selection} svgNodeGroup Placeholder for subclasses.
     */
    _newInfoBox(svgNodeGroup) { 
        this.nodeInfoBox = new NodeInfo(this);
    }

    /**
     * Create a new node connection info window.
     * @param {Selection} svgNodeGroup Placeholder for subclasses.
     */
    _newConnInfoBox(svgNodeGroup) { 
        this.nodeInfoBox = new NodeConnectionInfo(this);
    }

    /**
     * If node info mode is active, create a new node info window, populate, and display it.
     * @param {Event} event The Event object created by the trigger.
     * @param {TreeNode} node The node associated with the HTML object.
     * @param {String} [color = null] The color of the titlebard. Autoselected if null.
     * @param {Boolean} [isConnection = false] If true, make a connection info window.
     */
    showInfoBox(event, node, color = null, isConnection = false) {
        if (this.click.isNodeInfo) {
            const svgNodeGroup = d3.select(event.currentTarget);
            if (!color) color = svgNodeGroup.select('rect').style('fill');

            if (isConnection) {
                this._newConnInfoBox(svgNodeGroup)
            }
            else this._newInfoBox(svgNodeGroup);

            this.nodeInfoBox.activate();
            this.click.update(this.nodeInfoBox);
            this.nodeInfoBox.update(event, node, color);
        }
    }

    showCellInfoBox(event, cell, color = null) {
        if (this.click.isNodeInfo) {
            const svgNodeGroup = d3.select(event.currentTarget);
            if (!color) color = svgNodeGroup.select('rect').style('fill');

            this._newConnInfoBox(svgNodeGroup);
            this.nodeInfoBox.activate();
            this.click.update(this.nodeInfoBox);
            this.nodeInfoBox.update(event, cell, color);
        }
    }

    /** Hide and destroy the node info window */
    removeInfoBox() {
        if (this.nodeInfoBox) {
            this.nodeInfoBox.clear();
            delete this.nodeInfoBox;
            this.nodeInfoBox = null;
        }
    }

    /** Move the node info window near the mouse if node info mode is active */
    moveInfoBox(e) {
        if (this.nodeInfoBox) { this.nodeInfoBox.moveNearMouse(e); }        
    }

    /** Create a persistent node info window if node info mode is active */
    pinInfoBox() {
        if (this.nodeInfoBox) { this.nodeInfoBox.pin(); }
    }

    /**
     * Set the range and current value of the collapse depth slider.
     * @param {Number} opts.min The shallowest elements to allow to collapse via slider.
     * @param {Number} opts.max The deepest elements to allow to collapse via slider.
     * @param {Number} opts.val The current slider collapse depth value.
     */
    setCollapseDepthSlider(opts = {}) {
        const min = opts.min ? opts.min : 2;
        const max = opts.max ? opts.max : this.diag.model.maxDepth;
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

        this.collapseDepthSlider = d3.select('input#depth-slider');
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

    /** Determine the style of the inactive handle (override with subclass) */
    _inactiveResizerHandlerStyle() {
        return 'inactive-resizer-handle';
    }

    /** Set up event handlers for grabbing the bottom corner and dragging */
    _setupResizerDrag() {
        const handle = d3.select('#n2-resizer-handle');
        const box = d3.select('#n2-resizer-box');
        const body = d3.select('body');
        const self = this;

        handle.on('mousedown', e => {
            box
                .style('top', self.diag.layout.gapSpace)
                .style('bottom', self.diag.layout.gapSpace);

            handle.attr('class', 'active-resizer-handle');
            box.attr('class', 'active-resizer-box');

            const startPos = new Dimensions({ x: e.clientX, y: e.clientY });
            const startDims = new Dimensions({
                width: parseInt(box.style('width')),
                height: parseInt(box.style('height'))
            });
            const offset = new Dimensions({
                x: startPos.x - startDims.width,
                y: startPos.y - startDims.height
            });

            // Reassigned by mousemove:
            let newDims = new Dimensions({
                x: startDims.width,
                y: startDims.height
            });

            handle.html(Math.round(newDims.x) + ' x ' + newDims.y);

            body.style('cursor', 'nwse-resize')
                .on('mouseup', () => {
                    self.diag.manuallyResized = true;

                    // Update the slider value and display
                    const defaultHeight = window.innerHeight * .95;
                    const newPercent = Math.round((newDims.y / defaultHeight) * 100);
                    d3.select('#model-slider').node().value = newPercent;
                    d3.select('#model-slider-label').html(newPercent + "%");

                    // Perform the actual resize
                    self.diag.verticalResize(newDims.y);

                    box.style('width', null).style('height', null);

                    // Turn off the resizing box border and handle
                    handle.attr('class', self._inactiveResizerHandlerStyle())
                    box.attr('class', 'inactive-resizer-box');

                    // Get rid of the drag event handlers
                    body.style('cursor', 'default')
                        .on('mousemove', null)
                        .on('mouseup', null);
                })
                .on('mousemove', e => {
                    const newHeight = e.clientY - offset.y;
                    if (newHeight + self.diag.layout.gapDist * 2 >= window.innerHeight * .5) {
                        newDims = new Dimensions({ x: e.clientX - offset.x, y: newHeight});

                        // Maintain the ratio by only resizing in the least moved direction
                        // and resizing the other direction by a fraction of that
                        if (newDims.x < newDims.y) {
                            newDims.y = self.diag.layout.calcHeightBasedOnNewWidth(newDims.x);
                        }
                        else {
                            newDims.x = self.diag.layout.calcWidthBasedOnNewHeight(newDims.y);
                        }

                        box.style('width', newDims.x + 'px').style('height', newDims.y + 'px');
                        handle.html(Math.round(newDims.x) + ' x ' + newDims.y);
                    }
                });

            e.preventDefault();
        });

    }

    /** Respond to window resize events if the diagram hasn't been manually sized */
    _setupWindowResizer() {
        const self = this;
        const diag = self.diag;
        this.pixelRatio = window.devicePixelRatio;

        self.resizeTimeout = null;
        d3.select(window).on('resize', function () {
            const newPixelRatio = window.devicePixelRatio;

            // If the browser window itself is zoomed, don't do anything
            if (newPixelRatio != self.pixelRatio) {
                self.pixelRatio = newPixelRatio;
                return;
            }

            if (!self.diag.manuallyResized) {
                clearTimeout(self.resizeTimeout);
                self.resizeTimeout =
                    setTimeout(function () {
                        diag.verticalResize(window.innerHeight * .95);
                    }, 200);
            }
        })
    }

    /**
     * Make sure the clicked node is deeper than the zoomed node, that
     * it's not the root node, and that it actually has children.
     * @param {TreeNode} node The right-clicked node to check.
     */
    isCollapsible(node) {
        return (node.depth > this.diag.zoomedElement.depth &&
            node.type !== 'root' && node.hasChildren());
    }

    /**
     * When a node is right-clicked or otherwise targeted for collapse, make sure it
     * it's allowed, then set the node as minimized and update the diagram drawing.
     */
    collapse() {
        const node = this.rightClickedNode;

        if (this.isCollapsible(node)) {

            if (this.collapsedRightClickNode !== undefined) {
                this.rightClickedNode = this.collapsedRightClickNode;
                this.collapsedRightClickNode = undefined;
            }

            this.findRootOfChangeFunction =
                this.findRootOfChangeForRightClick.bind(this);

            transitionDefaults.duration = transitionDefaults.durationFast;
            this.lastClickWasLeft = false;
            node.minimize();
            this.diag.update();
        }
    }

    /**
     * When a node is right-clicked, collapse it if it's allowed.
     * @param {TreeNode} node The node that was right-clicked.
     */
    rightClick(e, node) {
        e.preventDefault();
        e.stopPropagation();

        if (node.draw.minimized) {
            this.rightClickedNode = node;
            this.addBackButtonHistory();
            node.draw.manuallyExpanded = true;
            this._uncollapse(node);
            this.diag.update();
        }
        else if (this.isCollapsible(node)) {
            this.rightClickedNode = node;
            node.collapsable = true;

            this.addBackButtonHistory();
            node.draw.manuallyExpanded = false;
            this.collapse();
        }
    }

    /**
     * When a node with variables is alt-right-clicked, present a dialog with the
     * list of variables and allow the user to select which ones should be displayed.
     * @param {TreeNode} node The node that was alt-right-clicked.
     * @param {String} color The color of the clicked node, to use for the dialog ribbons.
     */
    altRightClick(e, node, color) {
        e.preventDefault();
        e.stopPropagation();
        window.getSelection().empty();

        // Make sure node is collapsible and window doesn't exist yet.
        if (this.isCollapsible(node) && !node.isFilter() && 
            d3.select('#childSelect-' + node.toId()).empty()) {
            new ChildSelectDialog(e, node, color, this.diag); // Create the modal dialog
        }
    }

    /**
     * Update states as if a left-click was performed, which may or may not have
     * actually happened.
     * @param {TreeNode} node The node that was targetted.
     */
    _setupLeftClick(node) {
        this.leftClickedNode = node;
        this.lastClickWasLeft = true;
        if (this.leftClickedNode.depth > this.diag.zoomedElement.depth) {
            this.leftClickIsForward = true; // forward
        }
        else if (this.leftClickedNode.depth < this.diag.zoomedElement.depth) {
            this.leftClickIsForward = false; // backwards
        }
        this.diag.updateZoomedElement(node);
        transitionDefaults.duration = transitionDefaults.durationFast;
    }

    /**
     * React to a left-clicked node by zooming in on it.
     * @param {TreeNode} node The targetted node.
     */
    leftClick(e, node) {
        // Don't do it if the node is already zoomed
        if (node === this.diag.zoomedElement) return;

        e.preventDefault();
        e.stopPropagation();

        if (!node.hasChildren() || node.isInput()) return;
        if (e.button != 0) return;
        this.addBackButtonHistory();
        node.expand();
        node.draw.manuallyExpanded = true;
        this._setupLeftClick(node);

        this.diag.update();
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
                exitIndex = lcRootIndex - this.diag.zoomedElementPrev.rootIndex;
            }
            else {
                enterIndex = this.diag.zoomedElementPrev.rootIndex - lcRootIndex;
            }
        }
    }

    /**
     * Preserve the current zoomed element and state of all hidden elements.
     * @param {Boolean} clearForward If true, erase the forward history.
     */
    addBackButtonHistory(clearForward = true) {
        let formerHidden = [];
        this.diag.model.findAllHidden(formerHidden, false);

        this.backButtonHistory.push({
            'node': this.diag.zoomedElement,
            'hidden': formerHidden,
            'search': this.toolbar.getSearchState(),
            'collapseDepth': this.currentCollapseDepth
        });

        if (clearForward) this.forwardButtonHistory = [];
    }

    /**
     * Preserve the specified node as the zoomed element,
     * and remember the state of all hidden elements.
     * @param {TreeNode} node The node to preserve as the zoomed element.
     */
    addForwardButtonHistory(node) {
        let formerHidden = [];
        this.diag.model.findAllHidden(formerHidden, true);

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
        if (this.backButtonHistory.length == 0) {
            debugInfo("backButtonPressed(): no items in history");
            return;
        }

        debugInfo("backButtonPressed(): " +
            this.backButtonHistory.length + " items in history");

        const history = this.backButtonHistory.pop();
        const oldZoomedElement = history.node;

        this.toolbar.setSearchState(history.search);
        this.setCollapseDepthSlider({ 'val': history.collapseDepth });

        // Check to see if the node is a collapsed node or not
        if (oldZoomedElement.collapsable) {
            this.leftClickedNode = oldZoomedElement;
            this.addForwardButtonHistory(oldZoomedElement);
            this.collapse();
        }
        else {
            for (let obj = oldZoomedElement; obj != null; obj = obj.parent) {
                //make sure history item is not minimized
                if (obj.draw.minimized) return;
            }

            this.addForwardButtonHistory(this.diag.zoomedElement);
            this._setupLeftClick(oldZoomedElement);
        }

        this.diag.model.resetAllHidden(history.hidden);
        this.diag.update();
    }

    /**
     * When the forward history button is clicked, pop the top node from that
     * history stack, and disable the button if the stack is empty. Find the
     * neared un-minimized node (if not the node itself) and zoom to that.
     * Add the previous zoomed node to the back history stack.
     */
    forwardButtonPressed() {
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
            if (obj.draw.minimized) return;
        }

        this.addBackButtonHistory(false);
        this._setupLeftClick(node);

        this.diag.model.resetAllHidden(history.hidden);
        this.diag.update();
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
            if (obj.depth == this.diag.chosenCollapseDepth) return obj;
        }
        return node;
    }

    /**
     * When either of the collapse or uncollapse toolbar buttons are
     * pressed, return the parent of the targetted node if it's a variable,
     * or the node itself if not.
     * @returns Parent of output node or node itself.
     */
    findRootOfChangeForCollapseUncollapseOutputs(node) {
        return node.isInputOrOutput()? node.parent : node;
    }

    /**
     * When the home button (aka return-to-root) button is clicked, zoom
     * to the root node.
     */
    homeButtonClick() {
        this.leftClickedNode = this.diag.model.root;
        this.lastClickWasLeft = true;
        this.leftClickIsForward = false;
        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        this.addBackButtonHistory();
        this.setCollapseDepthSlider();

        this.diag.reset();
    }

    /**
     * Minimize the specified node and recursively minimize its children.
     * @param {TreeNode} node The current node to operate on.
     */
    _collapseOutputs(node) {
        if (node.isGroup()) node.minimize();

        if (node.hasChildren()) {
            for (const child of node.children) {
                this._collapseOutputs(child);
            }
        }
    }

    /**
     * React to a button click and collapse all outputs of the specified node.
     * @param {TreeNode} node The initial node, usually the currently zoomed element.
     */
    collapseOutputsButtonClick(startNode) {
        this.addBackButtonHistory();
        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        transitionDefaults.duration = transitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this._collapseOutputs(startNode);
        this.diag.update();
    }

    /**
     * Mark this node and all of its children as unminimized/unhidden
     * @param {TreeNode} node The node to operate on.
     */
    _uncollapse(node) {
        node.expand().show();
        if (node.isFilteredVariable()) node.draw.filtered = false;

        if (node.hasChildren()) {
            for (const child of node.children) {
                this._uncollapse(child);
            }
        }

        if (node.isFilter()) { node.wipe(); } // Clear the contents if it's a filter node
    }

    /**
     * React to a button click and uncollapse the specified node.
     * @param {TreeNode} startNode The initial node.
     */
    uncollapseButtonClick(startNode) {
        this.addBackButtonHistory();
        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        transitionDefaults.duration = transitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this._uncollapse(startNode);
        startNode.draw.manuallyExpanded = true;
        this.diag.update();
    }

    /** Any collapsed nodes are expanded, starting with the specified node. */
    expandAll(startNode) {
        this.diag.showWaiter();

        this.addBackButtonHistory();
        this.diag.model.manuallyExpandAll(startNode);

        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        transitionDefaults.duration = transitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this.diag.update();
    }

    /** All nodes are collapsed, starting with the specified node. */
    collapseAll(startNode) {
        this.addBackButtonHistory();
        this.diag.model.minimizeAll(startNode);

        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        transitionDefaults.duration = transitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this.diag.update();
    }

    /**
     * React to a new selection in the collapse-to-depth toolbar slider.
     * @param {Number} depth Selected depth to collapse to.
     */
    collapseToDepth(depth) {
        this.addBackButtonHistory();
        this.diag.minimizeToDepth(depth);
        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseDepth.bind(
            this
        );
        transitionDefaults.duration = transitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this.diag.update();
    }

    /** React to the toggle legend button, and show or hide the legend. */
    toggleLegend() {
        this.legend.toggle();

        d3.selectAll('i.icon-key').attr('class',
            this.legend.hidden ? 'fas icon-key' : 'fas icon-key active-tab-icon');
    }

    /** Show or hide the node info panel button */
    toggleNodeData() {
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
        const self = this; // For callbacks that change "this".

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
        this.currentCollapseDepth = this.collapseDepthSliderVal;

        d3.select('#undo-graph').classed('disabled-button',
            (this.backButtonHistory.length == 0));
        d3.select('#redo-graph').classed('disabled-button',
            (this.forwardButtonHistory.length == 0));
    }

    /** Called when the search button is actually or effectively clicked to start a search. */
    searchButtonClicked() {
        this.addBackButtonHistory();
        this.diag.search.performSearch();

        this.findRootOfChangeFunction = this.diag.search.findRootOfChangeForSearch;
        transitionDefaults.duration = transitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this.diag.search.updateRecomputesAutoComplete = false;
        this.diag.update();
    }

    /**
     * Called when the enter key is pressed down in the search input box.
     * @param {Event} e Object with information about the event.
     */
    searchEnterKeyDownEventListener(e) {

        const target = e.target;
        if (target.id == 'awesompleteId') {
            const key = e.which || e.keyCode;
            if (key == 13) {
                // 13 is enter
                this.callSearchFromEnterKeyPressed = true;
            }
        }
    }

    /**
     * Called when the enter key is released in the search input box.
     * @param {Event} e Object with information about the event.
     */
    searchEnterKeyUpEventListener(e) {
        const target = e.target;
        if (target.id == 'awesompleteId') {
            const key = e.which || e.keyCode;
            if (key == 13) {
                // 13 is enter
                if (this.callSearchFromEnterKeyPressed) {
                    this.searchButtonClicked();
                }
            }
        }
    }

    /**
     * Save the model state to a file.
     * @param {Object} [extraData={}] Additional items to save.
     */
    saveState(extraData = {}) {
        const stateFileName = basename() + '.n2view';

        // Zoomed node
        const zoomedElement = this.diag.zoomedElement.id;

        // Expand/Collapse state of all nodes in model.
        const expandCollapse = Array();
        this.diag.getSubState(expandCollapse);

        // Arrow State
        const arrowState = this.diag.arrowMgr.savePinnedArrows();

        const dataDict = {
            ...extraData,
            'zoomedElement': zoomedElement,
            'expandCollapse': expandCollapse,
            'arrowState': arrowState,
            'md5_hash': this.diag.model.md5_hash,
        };

        const link = document.createElement('a');
        link.setAttribute('download', stateFileName);
        const data_blob = new Blob([JSON.stringify(dataDict)], { type: 'text/plain' });

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

    /**
     * Preset a file open dialog, read in the selected file, validate the contents,
     * and update the diagram to the saved state.
     * @param {Event} e The event that initiated the dialog.
     * @param {UserInterface} ui Reference to the UserInterface object.
     */
    loadView(e, ui) {
        const fr = new FileReader();
        const fileInput = e.currentTarget;

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
            if (dataDict.md5_hash != ui.diag.model.md5_hash) {
                alert("Cannot load view. Current model structure is different than in saved view.")
                return;
            }

            ui.addBackButtonHistory();
            ui.diag.restoreSavedState(dataDict);
        }

        fr.readAsText(fileInput.files[0]);
    }

    /** Load the model state to a file. */
    loadState() {
        document.getElementById('state-file-input').click();
    }
}
