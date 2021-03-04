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

        this.id = id;
        this.toolbarButton = d3.select(id);
        this.tooltipBox = tooltipBox;
        this.help = null;

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

    /**
     * Use when the info displayed on the help screen is different than the tooltip.
     * @param {String} helpText The info to display on the help screen for this button.
     * @returns {N2ToolbarButtonNoClick} Reference to this.
     */
    setHelpInfo(helpText) {
        this.help = helpText;
        return this;
    }

    getHelpInfo() {
        const parent = d3.select(this.toolbarButton.node().parentNode);
        let primaryGrpBtnId = null;
        const expansionItem = parent.classed('toolbar-group-expandable');

        if (expansionItem) {
            const grandparent = d3.select(parent.node().parentNode);
            primaryGrpBtnId = grandparent.select(':first-child').attr('id');
        }

        return {
            'id': this.id.replace('#', ''),
            'desc': this.help ? this.help : this.tooltips[0],
            'bbox': this.toolbarButton.node().getBoundingClientRect(),
            'expansionItem': expansionItem,
            'primaryGrpBtnId': primaryGrpBtnId
        };
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

        this.toolbarButton.on('click', function () { self.click(this); });
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
        this.hideToolbarButton = d3.select('.toolbar-hide-container');
        this.hideToolbarIcon = this.hideToolbarButton.select('i');
        this.searchBar = d3.select('#awesompleteId');
        this.searchCount = d3.select('#searchCountId');
        this.buttons = [];

        this.hidden = true;
        this.helpInfo = null;

        // Display toolbar if not embedded, or if embedded doc location
        // href include the #toolbar anchor

        if (!EMBEDDED || (EMBEDDED && window.location.href.includes('#toolbar'))) {
            this.show();
        }

        this._setupButtonFunctions(n2ui);
    }

    /**
     * Generate the data structure that describes all of the toolbar buttons. Nothing
     * is actually rendered here (that is done in the N2Help constructor.)
     */
    async _setupHelp() {
        const self = this;

        // Take a "screenshot" of the toolbar and put it in a canvas
        await html2canvas(document.querySelector("#toolbarLoc"), {
            logging: false,
            /*
            onclone: function(cloneDoc) {
                const css = cloneDoc.createElement("style");
                css.type = 'text/css';
                const fontStyle = "@font-face { font-family: 'n2toolbar-icons'; src: url('data:application/font-woff;charset=utf-8;base64,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') format('woff'); }";
                if (css.styleSheet)  
                    css.styleSheet.cssText = fontStyle; 
                else  
                    css.appendChild(cloneDoc.createTextNode(fontStyle));

                cloneDoc.getElementsByTagName("head")[0].appendChild(css);
                console.log(cloneDoc)
            }
            */
        })
            .then(canvas => {
                self.helpInfo = {
                    toolbarImg: canvas.toDataURL(),
                    width: canvas.width,
                    height: canvas.height,
                    buttons: {},
                    primaryButtons: {},
                    groups: 0
                }
            });

        for (const btn of this.buttons) {
            const info = btn.getHelpInfo();
            this.helpInfo.buttons[info.id] = info;
            if (info.primaryGrpBtnId) { // Is this a "child" of a group?
                // Keep track of which buttons are at the front of collapsing groups
                // and the group member ids
                if (info.primaryGrpBtnId in self.helpInfo.primaryButtons) {
                    self.helpInfo.primaryButtons[info.primaryGrpBtnId].push(info.id);
                }
                else {
                    self.helpInfo.primaryButtons[info.primaryGrpBtnId] = [info.id];
                }
            }
        }
    }

    /** Slide everything to the left offscreen 75px, rotate the button */
    hide() {
        this.toolbarContainer.style('left', '-65px');
        this.hideToolbarButton.style('left', '-20px');
        this.hideToolbarIcon.style('transform', 'rotate(-180deg)');
        d3.select('#d3_content_div').style('margin-left', '-65px');
        this.hidden = true;
    }

    /** Slide everything to the right and rotate the button */
    show() {
        const self = this;

        // Run this the first time the toolbar appears so that none of the
        // buttons have changed from the default state.
        if (!this.helpInfo) {
            this.toolbarContainer.on('transitionend', e => {
                if (d3.event.propertyName == 'left') {
                    self._setupHelp().then(e => { this.toolbarContainer.on('transitionend', null) })
                }
            });
        }

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


    _addButton(btn) {
        this.buttons.push(btn);
        return btn;
    }

    /**
     * Associate all of the buttons on the toolbar with a method in N2UserInterface.
     * @param {N2UserInterface} n2ui A reference to the UI object
     */
    _setupButtonFunctions(n2ui) {
        const self = this; // For callbacks that change "this". Alternative to using .bind().
        const tooltipBox = d3.select(".tool-tip");

        this._addButton(new N2ToolbarButtonClick('#searchButtonId', tooltipBox,
            "Collapse model to only variables that match search term",
            e => {
                self.searchCount.html('0 matches');

                self.searchBar.node().value = '';
                d3.select('#searchbar-and-label').attr('class', 'searchbar-visible');

                // This is necessary rather than just calling focus() due to the
                // transition animation
                window.setTimeout(function () {
                    self.searchBar.node().focus();
                }, 200);

                // Retract search bar when focus is lost
                self.searchBar.on('focusout', function () {
                    d3.select('#searchbar-and-label').attr('class', 'searchbar-hidden')
                    self.searchBar.on('focusout', null);
                });
            })
        );

        this._addButton(new N2ToolbarButtonClick('#reset-graph', tooltipBox,
            "View entire model starting from root", e => { n2ui.homeButtonClick(); }));

        this._addButton(new N2ToolbarButtonClick('#undo-graph', tooltipBox,
            "Move back in view history", e => { n2ui.backButtonPressed() }));

        this._addButton(new N2ToolbarButtonClick('#redo-graph', tooltipBox,
            "Move forward in view history", e => { n2ui.forwardButtonPressed() }));

        this._addButton(new N2ToolbarButtonClick('#collapse-element', tooltipBox,
            "Control variable collapsing",
            e => { n2ui.collapseAll(n2ui.n2Diag.zoomedElement) }));

        this._addButton(new N2ToolbarButtonClick('#collapse-element-2', tooltipBox,
            "Collapse only variables in current view",
            function (target) {
                n2ui.collapseAll(n2ui.n2Diag.zoomedElement);
                self._setRootButton(target);
            }));

        this._addButton(new N2ToolbarButtonClick('#collapse-all', tooltipBox,
            "Collapse all variables in entire model",
            function (target) {
                n2ui.collapseAll(n2ui.n2Diag.model.root);
                self._setRootButton(target);
            }));

        this._addButton(new N2ToolbarButtonClick('#expand-element', tooltipBox,
            "Expand only variables in current view",
            function (target) {
                n2ui.expandAll(n2ui.n2Diag.zoomedElement);
                self._setRootButton(target);
            }));

        this._addButton(new N2ToolbarButtonClick('#expand-all', tooltipBox,
            "Expand all variables in entire model",
            function (target) {
                n2ui.expandAll(n2ui.n2Diag.model.root);
                self._setRootButton(target);
            }));

        this._addButton(new N2ToolbarButtonClick('#hide-connections', tooltipBox,
            "Set connections visibility",
            function (target) {
                n2ui.n2Diag.clearArrows();
                self._setRootButton(target);
            }));

        this._addButton(new N2ToolbarButtonClick('#hide-connections-2', tooltipBox,
            "Remove all connection arrows",
            function (target) {
                n2ui.n2Diag.clearArrows();
                self._setRootButton(target);
            }));

        this._addButton(new N2ToolbarButtonClick('#show-all-connections', tooltipBox,
            "Show all connections in view",
            function (target) {
                n2ui.n2Diag.showAllArrows();
                self._setRootButton(target);
            }));

        this._addButton(new N2ToolbarButtonClick('#linear-solver-button', tooltipBox,
            "Control solver tree display",
            function (target) {
                n2ui.setSolvers(true);
                n2ui.showSolvers();
            }));

        this._addButton(new N2ToolbarButtonClick('#linear-solver-button-2', tooltipBox,
            "Show linear solvers",
            function (target) {
                n2ui.setSolvers(true);
                n2ui.showSolvers();
                self._setRootButton(target);
            }));

        this._addButton(new N2ToolbarButtonClick('#non-linear-solver-button', tooltipBox,
            "Show non-linear solvers",
            function (target) {
                n2ui.setSolvers(false);
                n2ui.showSolvers();
                self._setRootButton(target);
            }));

        this._addButton(new N2ToolbarButtonClick('#no-solver-button', tooltipBox,
            "Hide solvers",
            function (target) {
                n2ui.hideSolvers();
                self._setRootButton(target);
            }));

        this._addButton(new N2ToolbarButtonToggle('#legend-button', tooltipBox,
            ["Show legend", "Hide legend"],
            pred => { return n2ui.legend.hidden; },
            e => { n2ui.toggleLegend(); }
        ));

        this._addButton(new N2ToolbarButtonToggle('#desvars-button', tooltipBox,
            ["Show optimization variables", "Hide optimization variables"],
            pred => { return n2ui.desVars; },
            e => { n2ui.toggleDesVars(); }
        ));

        this._addButton(new N2ToolbarButtonNoClick('#text-slider-button', tooltipBox,
            "Set text height"));
        this._addButton(new N2ToolbarButtonNoClick('#depth-slider-button', tooltipBox,
            "Set collapse depth"));
        this._addButton(new N2ToolbarButtonNoClick('#model-slider-button', tooltipBox,
            "Set model height"));

        this._addButton(new N2ToolbarButtonNoClick('#save-load-button', tooltipBox,
            "Save or Load an Image or View"));

        this._addButton(new N2ToolbarButtonClick('#save-button', tooltipBox,
            "Save to SVG", e => { n2ui.n2Diag.saveSvg() }));

        this._addButton(new N2ToolbarButtonClick('#save-state-button', tooltipBox,
            "Save View", e => { n2ui.saveState() }));

        this._addButton(new N2ToolbarButtonClick('#load-state-button', tooltipBox,
            "Load View", e => { n2ui.loadState() }));

        this._addButton(new N2ToolbarButtonToggle('#info-button', tooltipBox,
            ["Hide detailed node information", "Show detailed node information"],
            pred => { return n2ui.nodeInfoBox.active; },
            e => {
                n2ui.nodeInfoBox.clear();
                n2ui.nodeInfoBox.toggle();
            }
        ));

        this._addButton(new N2ToolbarButtonToggle('#question-button', tooltipBox,
            ["Hide N2 diagram help", "Show N2 diagram help"],
            pred => { return !!(d3.select(".window-theme-help").size()); },
            e => { new N2Help(self.helpInfo); }
        ));

        // Don't add this to the array of tracked buttons because it confuses
        // the help screen generation
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
        this.toolbar.select('#model-slider')
            .on('input', function () {
                d3.select('#model-slider-label').html(this.value + "%");
            })
            .on('mouseup', function () {
                n2ui.n2Diag.manuallyResized = true;
                const modelHeight = window.innerHeight * (parseInt(this.value) / 100);
                n2ui.n2Diag.verticalResize(modelHeight);
                const gapSpace = (n2ui.n2Diag.dims.size.partitionTreeGap - 3) +
                    n2ui.n2Diag.dims.size.unit;
            });

        this.toolbar.select('#model-slider-fit')
            .on('click', function () {
                n2ui.n2Diag.manuallyResized = false;
                d3.select('#model-slider').node().value = '95';
                d3.select('#model-slider-label').html("95%")
                n2ui.n2Diag.verticalResize(window.innerHeight * .95);

                const gapSpace = (n2ui.n2Diag.dims.size.partitionTreeGap - 3) +
                    n2ui.n2Diag.dims.size.unit;
            })
    }
}
