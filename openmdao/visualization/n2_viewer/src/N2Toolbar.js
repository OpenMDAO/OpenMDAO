/**
 * Base class for toolbar button events. Show, hide, or
 * move the tool tip box.
 * @typedef N2ToolbarButtonNoClick
 * @property {Object} tooltipBox A reference to the tool-tip element.
 * @property {Array} tooltips One or two tooltips to display.
 * @property {Object} toolbarButton A reference to the toolbar button.
 */
class N2ToolbarButtonNoClick {
    /**
     * Set up the event handlers.
     * @param {String} id A selector for the button element.
     * @param {Object} tooltipBox A reference to the tool-tip element.
     * @param {String} tooptipText Content to fill the tool-tip box with.
     */
    constructor(id, tooltipBox, tooltipText) {
        this.tooltips = [tooltipText];

        this.toolbarButton = d3.select(id);
        this.tooltipBox = tooltipBox;

        this.toolbarButton
            .on("mouseover", this.mouseOver.bind(this))
            .on("mouseleave", this.mouseLeave.bind(this))
            .on("mousemove", this.mouseMove.bind(this));
    }

    /** When the mouse enters the element, show the tool tip */
    mouseOver() {
        this.tooltipBox
            .text(this.tooltips[0])
            .style("visibility", "visible");
    }

    /** When the mouse leaves the element, hide the tool tip */
    mouseLeave() {
        this.tooltipBox.style("visibility", "hidden");
    }

    /** Keep the tool-tip near the mouse */
    mouseMove() {
        this.tooltipBox.style("top", (d3.event.pageY - 30) + "px")
            .style("left", (d3.event.pageX + 5) + "px");
    }
}

/**
 * Manage clickable toolbar buttons
 * @typedef N2ToolbarButtonClick
 * @property {Object} tooltipBox A reference to the tool-tip element.
 * @property {Array} tooltips One or two tooltips to display.
 * @property {Object} toolbarButton A reference to the toolbar button.
 * @property {Function} clickFn The function to call when clicked.
 */
class N2ToolbarButtonClick extends N2ToolbarButtonNoClick {
    /**
     * Set up the event handlers.
     * @param {String} id A selector for the button element.
     * @param {Object} tooltipBox A reference to the tool-tip element.
     * @param {String} tooptipText Content to fill the tool-tip box with.
     * @param {Function} clickFn The function to call when clicked.
     */ 
    constructor(id, tooltipBox, tooltipText, clickFn) {
        super(id, tooltipBox, tooltipText);
        this.clickFn = clickFn;

        let self = this;

        this.toolbarButton.on('click', function() { self.click(this); });
    }

    /**
     * Defined separately so the derived class can override
     * @param {Object} target Reference to the HTML element that was clicked
     */
    click(target) {
        this.clickFn(target);
    }
}

/**
 * Manage toolbar buttons that alternate states when clicked.
 * @typedef N2ToolbarButtonToggle
 * @property {Object} tooltipBox A reference to the tool-tip element.
 * @property {Array} tooltips One or two tooltips to display.
 * @property {Object} toolbarButton A reference to the toolbar button.
 * @property {Function} clickFn The function to call when clicked.
 * @property {Function} predicateFn Function returning a boolean representing the state.
 */
class N2ToolbarButtonToggle extends N2ToolbarButtonClick {
    /**
     * Set up the event handlers.
     * @param {String} id A selector for the button element.
     * @param {Object} tooltipBox A reference to the tool-tip element.
     * @param {String} tooptipTextArr A pair of tooltips for alternate states.
     * @param {Function} predicateFn Function returning a boolean representing the state.
     * @param {Function} clickFn The function to call when clicked.
     */ 
    constructor(id, tooltipBox, tooltipTextArr, predicateFn, clickFn) {
        super(id, tooltipBox, tooltipTextArr[0], clickFn);
        this.tooltips.push(tooltipTextArr[1]);
        this.predicateFn = predicateFn;
    }

    /**
     * When the mouse enters the element, show a tool tip based
     * on the result of the predicate function.
     */
    mouseOver() {
        this.tooltipBox
            .text(this.predicateFn() ? this.tooltips[0] : this.tooltips[1])
            .style("visibility", "visible");
    }

    /**
     * When clicked, perform the associated function, then change the tool tip
     * based on the result of the predicate function.
     * @param {Object} target Reference to the HTML element that was clicked
     */
    click(target) {
        this.clickFn(target);

        this.tooltipBox
            .text(this.predicateFn() ? this.tooltips[0] : this.tooltips[1])
            .style("visibility", "visible");

    }
}

/**
 * Manage the set of buttons and tools at the left of the diagram.
 * @typedef N2Toolbar
 * @property {Boolean} hidden Whether the toolbar is visible or not.
 */
class N2Toolbar {
    /**
     * Set up the event handlers for mouse hovering and clicking.
     * @param {N2UserInterface} n2ui Reference to the main interface object
     * @param {Number} sliderHeight The maximum height of the n2
     */
    constructor(n2ui, sliderHeight = window.innerHeight * .95) {
        const self = this;

        this.toolbarContainer = d3.select('#toolbarLoc');
        this.toolbar = d3.select('#true-toolbar');
        this.hideToolbarButton = d3.select('#hide-toolbar');
        this.hideToolbarIcon = this.hideToolbarButton.select('i');
        this.searchBar = d3.select('#awesompleteId');

        this.hidden = true;
        if (!EMBEDDED) this.show();

        d3.select('#model-slider')
            .attr('min', window.innerHeight * .5)
            .attr('max', window.innerHeight * 2)
            .attr('value', window.innerHeight * .95)

        this._setupExpandableButtons();
        this._setupButtonFunctions(n2ui);

        // Expand/contract the search bar
        d3.select('#searchbar-container')
            .on('mouseover', function () {
                self.searchBar.style('width', '200px');
                self.toolbarContainer.style('z-index', '5');
            })
            .on('mouseout', function () {
                self.searchBar.style('width', '0px');
                self.toolbarContainer.style('z-index', '1');
            })
    }

    /** Slide everything to the left offscreen 75px, rotate the button */
    hide() {
        this.toolbarContainer.style('left', '-75px');
        this.hideToolbarButton.style('left', '-30px');
        this.hideToolbarIcon.style('transform', 'rotate(-180deg)');
        d3.select('#d3_content_div').style('margin-left', '-75px');
        this.hidden = true;
    }

    /** Slide everything to the right and rotate the button */
    show() {
        this.hideToolbarIcon.style('transform', 'rotate(0deg)');
        this.toolbarContainer.style('left', '0px');
        this.hideToolbarButton.style('left', '45px');
        d3.select('#d3_content_div').style('margin-left', '0px');
        this.hidden = false;
    }

    toggle() {
        if (this.hidden) this.show();
        else this.hide();
    }

    _setupExpandableButtons() {
        const self = this;

        // Open expandable buttons when hovered over
        d3.selectAll('.expandable > div')
            .on('mouseover', function () {
                self.toolbarContainer.style('z-index', '5');
                d3.select(this).style('max-width', '200px');
            })
            .on('mouseout', function () {
                d3.select(this).style('max-width', '0');
                self.toolbarContainer.style('z-index', '1')
            })
    }

    /** When an expanded button is clicked, update the 'root' button to the same icon/function. */
    _setRootButton(clickedNode) {
        let container = d3.select(clickedNode.parentNode.parentNode);
        let button = d3.select(clickedNode);
        let rootButton = container.select('i');

        rootButton
            .attr('class', button.attr('class'))
            .attr('id', button.attr('id'))
            .node().onclick = button.node().onclick;
    }

    /**
     * Associate all of the buttons on the toolbar with a method in N2UserInterface.
     * @param {N2UserInterface} n2ui A reference to the UI object 
     */
    _setupButtonFunctions(n2ui) {
        const self = this; // For callbacks that change "this". Alternative to using .bind().
        const tooltipBox = d3.select(".tool-tip");

        new N2ToolbarButtonClick('#reset-graph', tooltipBox,
            "View entire model starting from root", e => { n2ui.homeButtonClick(); });

        new N2ToolbarButtonClick('#undo-graph', tooltipBox,
            "Move back in view history", e => { n2ui.backButtonPressed() });

        new N2ToolbarButtonClick('#redo-graph', tooltipBox,
            "Move forward in view history", e => { n2ui.forwardButtonPressed() });

        new N2ToolbarButtonClick('#collapse-element', tooltipBox,
            "Control variable collapsing",
            e => { n2ui.collapseOutputsButtonClick(n2ui.n2Diag.zoomedElement) });

        new N2ToolbarButtonClick('#collapse-element-2', tooltipBox,
            "Collapse only variables in current view",
            function (target) {
                n2ui.collapseOutputsButtonClick(n2ui.n2Diag.zoomedElement);
                self._setRootButton(target);
            });

        new N2ToolbarButtonClick('#collapse-all', tooltipBox,
            "Collapse all variables in entire model",
            function (target) {
                n2ui.collapseOutputsButtonClick(n2ui.n2Diag.model.root);
                self._setRootButton(target);
            });

        new N2ToolbarButtonClick('#expand-element', tooltipBox,
            "Expand only variables in current view",
            function (target) {
                n2ui.uncollapseButtonClick(n2ui.n2Diag.zoomedElement);
                self._setRootButton(target);
            });

        new N2ToolbarButtonClick('#expand-all', tooltipBox,
            "Expand all variables in entire model",
            function (target) {
                n2ui.uncollapseButtonClick(n2ui.n2Diag.model.root);
                self._setRootButton(target);
            });

        new N2ToolbarButtonClick('#show-connections', tooltipBox,
            "Set connections visibility",
            function (target) {
                n2ui.n2Diag.showArrows();
                self._setRootButton(target);
            });

        new N2ToolbarButtonClick('#hide-connections', tooltipBox,
            "Hide all connection arrows",
            function (target) {
                n2ui.n2Diag.clearArrows();
                self._setRootButton(target);
            });

        new N2ToolbarButtonClick('#show-connections-2', tooltipBox,
            "Show pinned connection arrows",
            function (target) {
                n2ui.n2Diag.showArrows();
                self._setRootButton(target);
            });

        new N2ToolbarButtonClick('#show-all-connections', tooltipBox,
            "Show all connections in model",
            function (target) {
                n2ui.n2Diag.showAllArrows();
                self._setRootButton(target);
            });

        new N2ToolbarButtonToggle('#linear-solver-button', tooltipBox,
            ["Show non-linear solvers", "Show linear solvers"],
            pred => { return !n2ui.n2Diag.showLinearSolverNames; },
            e => { n2ui.toggleSolverNamesCheckboxChange(); }
        );

        new N2ToolbarButtonToggle('#legend-button', tooltipBox,
            ["Show legend", "Hide legend"],
            pred => { return n2ui.legend.hidden; },
            e => { n2ui.toggleLegend(); }
        );

        new N2ToolbarButtonNoClick('#text-slider-button', tooltipBox, "Set text height");
        new N2ToolbarButtonNoClick('#depth-slider-button', tooltipBox, "Set collapse depth");
        new N2ToolbarButtonNoClick('#model-slider-button', tooltipBox, "Set model height");

        new N2ToolbarButtonClick('#save-button', tooltipBox,
            "Save to SVG", e => { n2ui.n2Diag.saveSvg() });

        new N2ToolbarButtonToggle('#info-button', tooltipBox,
            ["Show detailed node information", "Hide detailed node information"],
            pred => { return n2ui.nodeInfoBox.hidden; },
            e => { n2ui.nodeInfoBox.toggle(); }
        );

        new N2ToolbarButtonToggle('#question-button', tooltipBox,
            ["Hide N2 diagram help", "Show N2 diagram help"],
            pred => { return d3.select("#myModal").style('display') == "block"; },
            DisplayModal
        );

        new N2ToolbarButtonToggle('#hide-toolbar', tooltipBox,
            ["Show toolbar", "Hide toolbar"],
            pred => { return self.hidden },
            e => { self.toggle() }
        );

        // The font size slider is a range input
        this.toolbar.select('#text-slider').on('input', function () {
            const fontSize = this.value;
            n2ui.n2Diag.fontSizeSelectChange(fontSize);

            const fontSizeIndicator = self.toolbar.select('#font-size-indicator');
            fontSizeIndicator.html(fontSize + ' px');
        });

        // The model height slider is a range input
        this.toolbar.select('#model-slider').on('mouseup', function () {
            const modelHeight = parseInt(this.value);
            n2ui.n2Diag.verticalResize(modelHeight);
        });
    }
}