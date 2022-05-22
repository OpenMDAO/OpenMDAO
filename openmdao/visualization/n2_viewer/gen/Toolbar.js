// <<hpp_insert gen/ToolbarButtonClick.js>>
// <<hpp_insert gen/ToolbarButtonToggle.js>>
// <<hpp_insert gen/ClickHandler.js>>
// <<hpp_insert gen/DiagramHelp.js>>

/**
 * Manage the set of buttons and tools at the left of the diagram.
 * @typedef Toolbar
 * @property {Boolean} hidden Whether the toolbar is visible or not.
 */
class Toolbar {
    /**
     * Set up the event handlers for mouse hovering and clicking.
     * @param {UserInterface} ui Reference to the main interface object
     */
    constructor(ui) {
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

        this._setupButtonFunctions(ui);
        this._setupHelp();
        this._helpWindow = null;
    }

    /**
     * Generate the data structure that describes all of the toolbar buttons. Nothing
     * is actually rendered here (that is done in the Help constructor.)
     */
    _setupHelp() {
        const toolbarBRect = this.toolbarContainer.node().getBoundingClientRect();

        this.helpInfo = {
            width: toolbarBRect.width,
            height: toolbarBRect.height,
            buttons: {},
            primaryButtons: {},
            groups: 0
        };

        for (const btn of this.buttons) {
            const info = btn.getHelpInfo();
            this.helpInfo.buttons[info.id] = info;
            if (info.primaryGrpBtnId) { // Is this a "child" of a group?
                // Keep track of which buttons are at the front of collapsing groups
                // and the group member ids
                if (info.primaryGrpBtnId in this.helpInfo.primaryButtons) {
                    this.helpInfo.primaryButtons[info.primaryGrpBtnId].push(info.id);
                }
                else {
                    this.helpInfo.primaryButtons[info.primaryGrpBtnId] = [info.id];
                }
            }
        }
    }

    /** Either create the help window the first time or redisplay it */
    _showHelp() {
        if (!this._helpWindow) this._helpWindow = new DiagramHelp(this.helpInfo);
        else this._helpWindow.show().modal(true);
    }

    /** Slide everything to the left offscreen, rotate the button */
    hide() {
        this.toolbarContainer.style('left', '-65px');
        this.hideToolbarIcon.style('transform', 'rotate(-180deg)');
        d3.select('#d3_content_div').style('margin-left', '-65px');
        this.hidden = true;
    }

    /** Slide everything to the right and rotate the button */
    show() {
        this.hideToolbarIcon.style('transform', 'rotate(0deg)');
        this.toolbarContainer.style('left', '0px');
        d3.select('#d3_content_div').style('margin-left', '0px');
        this.hidden = false;
    }

    /** Automatically select between visibile/hidden. */
    toggle() {
        if (this.hidden) this.show();
        else this.hide();
    }

    /**
     * When an expanded button is clicked, update the 'root' button to the same icon/function.
     * @param {HTMLElement} clickedNode The element where the event was triggered.
     */
    _setRootButton(clickedNode) {
        const container = d3.select(clickedNode.parentNode.parentNode);
        if (!container.classed('expandable')) return;

        const button = d3.select(clickedNode);
        const rootButton = container.select('i:not(.caret)');

        rootButton
            .attr('class', button.attr('class'))
            .attr('id', button.attr('id'))
            .on('click', button.on('click'));
    }

    /**
     * Minimal management of buttons which will be described on the help window.
     * @param {ToolbarButton} btn The button to add.
     * @returns {ToolbarButton} A reference to the new button.
     */
    _addButton(btn) {
        this.buttons.push(btn);
        return btn;
    }

    /**
     * When buttons are added out-of-order, the help screen can be jumbled. Using
     * this method gives control over where the button goes into the array which
     * corresponds to the order it appears on the help screen. This doesn't affect
     * the appearance of the functional toolbar.
     * @param {ToolbarButton} btn The button to add.
     * @param {Number} idx The position to insert the button into the array.
     * @returns {ToolbarButton} A reference to the new button.
     */
    _addButtonAtIndex(btn, idx) {
        this.buttons.splice(idx, 0, btn);
        return btn;
    }

    /** Get a snapshot of the search term and match count. */
    getSearchState() {
        return {
                'term': this.searchBar.property('value'),
                'matches': this.searchCount.html()
        }
    }

    /** Restore a snapshot of the search term and match count. */
    setSearchState(searchInfo) {
        this.searchBar.property('value', searchInfo.term);
        this.searchCount.html(searchInfo.matches);
    }

    /**
     * Associate all of the buttons on the toolbar with a method in UserInterface.
     * @param {UserInterface} ui A reference to the UI object
     */
    _setupButtonFunctions(ui) {
        const self = this; // For callbacks that change "this". Alternative to using .bind().
        const tooltipBox = d3.select(".tool-tip");

        this._addButton(new ToolbarButtonClick('#searchButtonId', tooltipBox,
            "Collapse model to variables matching search term",
            () => {
                if (self.searchBar.node().value == '') {
                    self.searchCount.html('0 matches');
                }

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

        this._addButton(new ToolbarButtonClick('#reset-graph', tooltipBox,
            "View entire model starting from root", () => ui.homeButtonClick()));

        this._addButton(new ToolbarButtonClick('#undo-graph', tooltipBox,
            "Move back in view history", () => ui.backButtonPressed()));

        this._addButton(new ToolbarButtonClick('#redo-graph', tooltipBox,
            "Move forward in view history", () => ui.forwardButtonPressed()));

        this._addButton(new ToolbarButtonClick('#collapse-element', tooltipBox,
            "Control variable collapsing",
            () => ui.collapseAll(ui.diag.zoomedElement)));

        this._addButton(new ToolbarButtonClick('#collapse-element-2', tooltipBox,
            "Collapse only variables in current view",
            (e, target) => {
                ui.collapseAll(ui.diag.zoomedElement);
                self._setRootButton(target);
            }));

        this._addButton(new ToolbarButtonClick('#collapse-all', tooltipBox,
            "Collapse all variables in entire model",
            (e, target) => {
                ui.collapseAll(ui.diag.model.root);
                self._setRootButton(target);
            }));

        this._addButton(new ToolbarButtonClick('#expand-element', tooltipBox,
            "Expand only variables in current view",
            (e, target) => { 
                ui.expandAll(ui.diag.zoomedElement);
                self._setRootButton(target);
            }));

        this._addButton(new ToolbarButtonClick('#expand-all', tooltipBox,
            "Expand all variables in entire model",
            (e, target) => {
                ui.expandAll(ui.diag.model.root);
                self._setRootButton(target);
            }));

        this._addButton(new ToolbarButtonToggle('#info-button', tooltipBox,
            ["Hide detailed node information", "Show detailed node information"],
            () => { return ui.click.isNodeInfo; },
            () => { ui.click.toggle('nodeinfo'); })).setHelpInfo("Select left-click action");

        this._addButton(new ToolbarButtonToggle('#info-button-2', tooltipBox,
            ["Hide detailed node information", "Show detailed node information"],
            () => { return ui.click.isNodeInfo; },
            (e, target) => {
                ui.click.toggle('nodeinfo');
                self._setRootButton(target);
            })).setHelpInfo("Toggle detailed node info mode");

        this._addButton(new ToolbarButtonToggle('#collapse-target', tooltipBox,
            ["Exit collapse/expand mode", "Enter collapse/expand mode"],
            () => { return ui.click.clickEffect == ClickHandler.ClickEffect.Collapse; },
            (e, target) => {
                ui.click.toggle('collapse');
                self._setRootButton(target);
            })).setHelpInfo("Toggle collapse/expand mode");

        this._addButton(new ToolbarButtonToggle('#filter-target', tooltipBox,
            ["Exit variable filtering mode", "Enter variable filtering mode"],
            () => { return ui.click.clickEffect == ClickHandler.ClickEffect.Filter; },
            (e, target) => {
                ui.click.toggle('filter');
                self._setRootButton(target);
            })).setHelpInfo("Toggle variable filtering mode");

        this._addButton(new ToolbarButtonClick('#hide-connections', tooltipBox,
            "Set connections visibility",
            () => ui.diag.clearArrows()));

        this._addButton(new ToolbarButtonClick('#hide-connections-2', tooltipBox,
            "Remove all connection arrows",
            (e, target) => { ui.diag.clearArrows(); self._setRootButton(target); }));

        this._addButton(new ToolbarButtonClick('#show-all-connections', tooltipBox,
            "Show all connections in view",
            (e, target) => { ui.diag.showAllArrows(); self._setRootButton(target); }));

        this._addButton(new ToolbarButtonNoClick('#text-slider-button', tooltipBox,
            "Set text height"));
        this._addButton(new ToolbarButtonNoClick('#depth-slider-button', tooltipBox,
            "Set collapse depth"));
        this._addButton(new ToolbarButtonNoClick('#model-slider-button', tooltipBox,
            "Set model height"));

        this._addButton(new ToolbarButtonNoClick('#save-load-button', tooltipBox,
            "Save or load an image or view"));

        this._addButton(new ToolbarButtonClick('#save-button', tooltipBox,
            "Save to SVG", () => ui.diag.saveSvg() ));

        this._addButton(new ToolbarButtonClick('#save-state-button', tooltipBox,
            "Save View", () => ui.saveState() ));

        this._addButton(new ToolbarButtonClick('#load-state-button', tooltipBox,
            "Load View", () => ui.loadState() ));

        this._addButton(new ToolbarButtonToggle('#legend-button', tooltipBox,
            ["Show legend", "Hide legend"],
            () => ui.legend.hidden,
            (e, target) => { ui.toggleLegend(); self._setRootButton(target); }))
            .setHelpInfo("Toggle legend");

        this._addButton(new ToolbarButtonClick('#question-button', tooltipBox,
            "Display help window",
            (e, target) => { self._showHelp(); self._setRootButton(target); }));

        this._addButton(new ToolbarButtonClick('#question-button-2', tooltipBox,
            "Display help window",
            (e, target) => { self._showHelp(); self._setRootButton(target); }));

        // Don't add this to the array of tracked buttons because it confuses
        // the help screen generation
        new ToolbarButtonToggle('#hide-toolbar', tooltipBox,
            ["Show toolbar", "Hide toolbar"],
            () => self.hidden, () => self.toggle());

        // The font size slider is a range input
        this.toolbar.select('#text-slider').on('input', function() {
            const fontSize = this.value;
            ui.diag.fontSizeSelectChange(fontSize);

            const fontSizeIndicator = self.toolbar.select('#font-size-indicator');
            fontSizeIndicator.html(fontSize + ' px');
        });

        // The model height slider is a range input
        this.toolbar.select('#model-slider')
            .on('input', function() {
                d3.select('#model-slider-label').html(`${this.value}%`);
            })
            .on('mouseup', function() {
                ui.diag.manuallyResized = true;
                const modelHeight = window.innerHeight * (parseInt(this.value) / 100);
                ui.diag.verticalResize(modelHeight);
            });

        this.toolbar.select('#model-slider-fit')
            .on('click', function() {
                ui.diag.manuallyResized = false;
                d3.select('#model-slider').node().value = '95';
                d3.select('#model-slider-label').html("95%")
                ui.diag.verticalResize(window.innerHeight * .95);
            })
    }
}
