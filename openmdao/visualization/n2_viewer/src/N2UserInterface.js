/**
 * Manage info for each node metadata property
 * @typedef InfoPropDefault
 * @property {String} key The identifier of the property.
 * @property {String} desc The description (label) to display.
 * @property {Boolean} capitalize Whether to capitialize every word in the desc.
 */
class InfoPropDefault {
    constructor(key, desc, capitalize = false) {
        this.key = key;
        this.desc = desc;
        this.capitalize = capitalize;
    }

    /** Return the same message since this is the base class */
    output(msg) { return msg; }
}

/**
 * Outputs a Yes or No to display. 
 * @typedef InfoPropYesNo
 */
class InfoPropYesNo extends InfoPropDefault {
    constructor(key, desc, capitalize = false) {
        super(key, desc, capitalize);
    }

    /** Return Yes or No when given True or False */
    output(boolVal) { return boolVal ? 'Yes' : 'No'; }
}

/**
 * Rename params to inputs and unknowns to outputs. 
 * @typedef InfoPropYesNo
 */
class InfoUpdateType extends InfoPropDefault {
    constructor(key, desc, capitalize = false) {
        super(key, desc, capitalize);
    }

    /** Replace the old terms with new ones */
    output(msg) {
        return msg
            .replace(/param/, 'input')
            .replace(/unknown/, 'output')
            .replace('_', ' ');
    }
}

/**
 * Manage a table containing all available metadata properties for
 * the currently active node, as well as whether the table is
 * visible or not.
 * @typedef NodeInfo
 */
class NodeInfo {
    /**
     * Build a list of the properties we care about and set up
     * references to the HTML elements.
     * @param {Object} abs2prom Object containing promoted variable names.
     */
    constructor(abs2prom) {
        this.propList = [
            new InfoPropDefault('absPathName', 'Absolute Name'),
            new InfoPropDefault('class', 'Class'),
            new InfoUpdateType('type', 'Type', true),
            new InfoPropDefault('dtype', 'DType'),
            new InfoPropDefault('subsystem_type', 'Subsystem Type', true),
            new InfoPropDefault('component_type', 'Component Type', true),
            new InfoPropYesNo('implicit', 'Implicit'),
            new InfoPropYesNo('is_parallel', 'Parallel'),
            new InfoPropDefault('linear_solver', 'Linear Solver'),
            new InfoPropDefault('nonlinear_solver', 'Non-Linear Solver')
        ];

        this.abs2prom = abs2prom;
        this.table = d3.select('#node-info-container');
        this.thead = this.table.select('thead');
        this.tbody = this.table.select('tbody');
        this.toolbarButton = d3.select('#info-button');
        this.hidden = true;
    }

    /** Make the info box visible if it's hidden */
    show() {
        this.toolbarButton.attr('class', 'fas icon-info-circle active-tab-icon');
        this.hidden = false;
        d3.select('#all_pt_n2_content_div').classed('node-data-cursor', true);
    }

    /** Make the info box hidden if it's visible */
    hide() {
        this.toolbarButton.attr('class', 'fas icon-info-circle');
        this.hidden = true;
        d3.select('#all_pt_n2_content_div').classed('node-data-cursor', false);
    }

    /** Toggle the visibility setting */
    toggle() {
        if (this.hidden) this.show();
        else this.hide();
    }

    _addPropertyRow(label, val, capitalize = false) {
        const newRow = this.tbody.append('tr');

        newRow.append('th')
            .attr('scope', 'row')
            .text(label);

        const td = newRow.append('td')
            .text(val);

        if (capitalize) td.attr('class', 'caps');
    }

    /**
     * Iterate over the list of known properties and display them
     * if the specified object contains them.
     * @param {Object} event The related event so we can get position.
     * @param {N2TreeNode} obj The node to examine.
     * @param {N2TreeNode} color The color to make the title bar.
     */
    update(event, obj, color = '#42926b') {
        if (this.hidden) return;
        // Put the name in the title
        this.table.select('thead th')
            .style('background-color', color)
            .text(obj.name);

        this.table.select('tfoot th')
            .style('background-color', color);

        if (this.abs2prom) {
            if (obj.isParam()) {
                this._addPropertyRow('Promoted Name', this.abs2prom.input[obj.absPathName]);
            }
            else if (obj.isUnknown()) {
                this._addPropertyRow('Promoted Name', this.abs2prom.output[obj.absPathName]);
            }
        }

        for (const prop of this.propList) {
            if (obj.propExists(prop.key) && obj[prop.key] != '') {
                this._addPropertyRow(prop.desc, prop.output(obj[prop.key]), prop.capitalize)
            }
        }

        // Solidify the size of the table after populating so that
        // it can be positioned reliably by move().
        this.table
            .style('width', this.table.node().scrollWidth + 'px')
            .style('height', this.table.node().scrollHeight + 'px')

        this.move(event);
        this.table.attr('class', 'info-visible');
    }

    /** Wipe the contents of the table body */
    clear() {
        if (this.hidden) return;
        this.table
            .attr('class', 'info-hidden')
            .style('width', 'auto')
            .style('height', 'auto')

        this.tbody.html('');
    }

    /**
     * Relocate the table to a position near the mouse
     * @param {Object} event The triggering event containing the position.
     */
    move(event) {
        if (this.hidden) return;
        const offset = 30;

        // Mouse is in left half of window, put box to right of mouse
        if (event.clientX < window.innerWidth / 2) {
            this.table.style('right', 'auto');
            this.table.style('left', (event.clientX + offset) + 'px')
        }
        // Mouse is in right half of window, put box to left of mouse
        else {
            this.table.style('left', 'auto');
            this.table.style('right', (window.innerWidth - event.clientX + offset) + 'px')
        }

        // Mouse is in top half of window, put box below mouse
        if (event.clientY < window.innerHeight / 2) {
            this.table.style('bottom', 'auto');
            this.table.style('top', (event.clientY - offset) + 'px')
        }
        // Mouse is in bottom half of window, put box above mouse
        else {
            this.table.style('top', 'auto');
            this.table.style('bottom', (window.innerHeight - event.clientY - offset) + 'px')
        }
    }
}

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

        this.backButtonHistory = [];
        this.forwardButtonHistory = [];

        this._setupCollapseDepthElement();
        this.updateClickedIndices();

        d3.select('#searchButtonId')
            .on('click', this.searchButtonClicked.bind(this));
        this._setupSearch();

        this.legend = new N2Legend(this.n2Diag.modelData);
        this.nodeInfoBox = new NodeInfo(this.n2Diag.model.abs2prom);
    }

    /** Set up the menu for selecting an arbitrary depth to collapse to. */
    _setupCollapseDepthElement() {
        let self = this;

        let collapseDepthElement = this.n2Diag.dom.parentDiv.querySelector(
            '#depth-slider'
        );

        collapseDepthElement.max = this.n2Diag.model.maxDepth - 1;
        collapseDepthElement.value = collapseDepthElement.max;

        collapseDepthElement.onmouseup = function (e) {
            const modelDepth = parseInt(e.target.value);
            self.collapseToDepthSelectChange(modelDepth);
        };
    }

    /**
     * When a node is right-clicked or otherwise targeted for collapse, make sure it
     * has children and isn't the root node. Set the node as minimized and update
     * the diagram drawing.
     */
    collapse() {
        testThis(this, 'N2UserInterface', 'collapse');

        let node = this.leftClickedNode;

        if (!node.hasChildren()) return;

        // Don't allow minimizing of root node
        if (node.depth > this.n2Diag.zoomedElement.depth || node.type !== 'root') {
            this.rightClickedNode = node;

            if (this.collapsedRightClickNode !== undefined) {
                this.rightClickedNode = this.collapsedRightClickNode;
                this.collapsedRightClickNode = undefined;
            }

            this.findRootOfChangeFunction =
                this.findRootOfChangeForRightClick.bind(this);

            N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
            this.lastClickWasLeft = false;
            node.toggleMinimize();
            this.n2Diag.update();
        }
    }

    /* When a node is right-clicked, collapse it. */
    rightClick(node1, node2) {
        testThis(this, 'N2UserInterface', 'rightClick');

        this.leftClickedNode = node1;
        this.rightClickedNode = node2;

        let node = this.leftClickedNode;
        node['collapsable'] = true;

        this.backButtonHistory.push({
            node: node,
        });

        d3.event.preventDefault();
        d3.event.stopPropagation();
        this.collapse();
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
        } else if (this.leftClickedNode.depth < this.n2Diag.zoomedElement.depth) {
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
        testThis(this, 'N2UserInterface', 'leftClick');

        if (!node.hasChildren() || node.isParam()) return;
        if (d3.event.button != 0) return;
        this.backButtonHistory.push({
            node: this.n2Diag.zoomedElement,
        });
        this.forwardButtonHistory = [];
        this._setupLeftClick(node);
        d3.event.preventDefault();
        d3.event.stopPropagation();
        this.n2Diag.update();
    }

    /**
     * Set up for an animated transition by setting and remembering where things were.
     */
    updateClickedIndices() {
        enterIndex = exitIndex = 0;
        if (this.lastClickWasLeft) {
            if (this.leftClickIsForward) {
                exitIndex =
                    this.leftClickedNode.rootIndex -
                    this.n2Diag.zoomedElementPrev.rootIndex;
            }
            else {
                enterIndex =
                    this.n2Diag.zoomedElementPrev.rootIndex -
                    this.leftClickedNode.rootIndex;
            }
        }
    }

    /**
     * When the back history button is clicked, pop the top node from that
     * history stack, and disable the button if the stack is empty. Find the
     * neared un-minimized node (if not the node itself) and zoom to that.
     * Add the previous zoomed node to the forward history stack.
     */
    backButtonPressed() {
        testThis(this, 'N2UserInterface', 'backButtonPressed');

        if (this.backButtonHistory.length == 0) return;

        let node = this.backButtonHistory[this.backButtonHistory.length - 1].node;

        // Check to see if the node is a collapsed node or not
        if (node.collapsable) {
            this.leftClickedNode = node;
            this.forwardButtonHistory.push({
                node: this.leftClickedNode,
            });
            this.collapse();
        }
        else {
            for (let obj = node; obj != null; obj = obj.parent) {
                //make sure history item is not minimized
                if (obj.isMinimized) return;
            }

            this.forwardButtonHistory.push({
                node: this.n2Diag.zoomedElement,
            });
            this._setupLeftClick(node);
        }

        this.backButtonHistory.pop();
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

        if (this.forwardButtonHistory.length == 0) return;
        let node = this.forwardButtonHistory.pop().node;
        d3.select('#forwardButtonId').attr('disabled',
            (this.forwardButtonHistory.length == 0));

        for (let obj = node; obj != null; obj = obj.parent) {
            // make sure history item is not minimized
            if (obj.isMinimized) return;
        }
        this.backButtonHistory.push({
            node: this.n2Diag.zoomedElement,
        });
        this._setupLeftClick(node);
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

        this.backButtonHistory.push({
            node: this.n2Diag.zoomedElement,
        });
        this.forwardButtonHistory = [];
        this._setupLeftClick(this.n2Diag.model.root);
        this.uncollapseButtonClick(this.n2Diag.model.root);
        this.n2Diag.update();
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

        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this._collapseOutputs(startNode);
        this.n2Diag.update();
    }

    /**
     * Mark this node and all of its children as unminimized
     * @param {N2TreeNode} node The node to operate on.
     */
    _uncollapse(node) {
        if (!node.isParam()) {
            node.isMinimized = false;
        }

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

        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseUncollapseOutputs;
        N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this._uncollapse(startNode);
        this.n2Diag.update();
    }

    /**
     * Recursively minimize non-parameter nodes to the specified depth.
     * @param {N2TreeNode} node The node to work on.
     * @param {Number} depth If the node's depth is the same or more, collapse it.
     */
    _collapseToDepth(node, depth) {
        if (node.isParamOrUnknown()) {
            return;
        }

        node.isMinimized = node.depth < depth ? false : true;

        if (node.hasChildren()) {
            for (let child of node.children) {
                this._collapseToDepth(child, depth);
            }
        }
    }

    /**
     * React to a new selection in the collapse-to-depth drop-down.
     * @param {Number} newChosenCollapseDepth Selected depth to collapse to.
     */
    collapseToDepthSelectChange(newChosenCollapseDepth) {
        testThis(this, 'N2UserInterface', 'collapseToDepthSelectChange');

        this.n2Diag.chosenCollapseDepth = newChosenCollapseDepth;
        if (this.n2Diag.chosenCollapseDepth > this.n2Diag.zoomedElement.depth) {
            this._collapseToDepth(
                this.n2Diag.model.root,
                this.n2Diag.chosenCollapseDepth
            );
        }
        this.findRootOfChangeFunction = this.findRootOfChangeForCollapseDepth.bind(
            this
        );
        N2TransitionDefaults.duration = N2TransitionDefaults.durationSlow;
        this.lastClickWasLeft = false;
        this.n2Diag.update();
    }

    /**
     * React to the toggle-solver-name button press and show non-linear if linear
     * is currently shown, and vice-versa.
     */
    toggleSolverNamesCheckboxChange() {
        testThis(this, 'N2UserInterface', 'toggleSolverNamesCheckboxChange');

        this.n2Diag.toggleSolverNameType();
        this.n2Diag.dom.parentDiv.querySelector(
            '#linear-solver-button'
        ).className = !this.n2Diag.showLinearSolverNames ?
                'fas icon-nonlinear-solver solver-button' :
                'fas icon-linear-solver solver-button';

        this.legend.toggleSolvers(this.n2Diag.showLinearSolverNames);

        if (this.legend.shown)
            this.legend.show(
                this.n2Diag.showLinearSolverNames,
                this.n2Diag.style.solvers
            );
        this.n2Diag.update();
    }

    /** React to the toggle legend button, and show or hide the legend below the N2. */
    toggleLegend() {
        testThis(this, 'N2UserInterface', 'toggleLegend');
        this.legend.toggle();

        d3.select('#legend-button').attr('class',
            this.legend.hidden ? 'fas icon-key' : 'fas icon-key active-tab-icon');
    }

    toggleNodeData() {
        testThis(this, 'N2UserInterface', 'toggleNodeData');

        const infoButton = d3.select('#info-button');
        const nodeData = d3.select('#node-info-container');

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

        d3.select('#undo-graph').attr('disabled',
            (this.backButtonHistory.length == 0));
        d3.select('#redo-graph').attr('disabled',
            (this.forwardButtonHistory.length == 0));
        d3.select('#reset-graph').attr('disabled',
            (this.n2Diag.zoomedElement === this.n2Diag.model.root));
    }

    /** Called when the search button is actually or effectively clicked to start a search. */
    searchButtonClicked() {
        testThis(this, 'N2UserInterface', 'searchButtonClicked');

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
}
