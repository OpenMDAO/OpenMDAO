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
        if (! EMBEDDED) this.show();

        d3.select('#model-slider').node().value = sliderHeight;

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

    /** Associate all of the buttons on the toolbar with a method in N2UserInterface. */
    _setupButtonFunctions(n2ui) {
        const self = this; // For callbacks that change "this". Alternative to using .bind().

        // There are a lot of simple click events, so iterate over this array of
        // [ selector, function ] elements and add a click handler for each.
        const clickEventArray = [
            ['#reset-graph', e => { n2ui.homeButtonClick() }],
            ['#undo-graph', e => { n2ui.backButtonPressed() }],
            ['#redo-graph', e => { n2ui.forwardButtonPressed() }],
            ['#expand-element', e => { n2ui.uncollapseButtonClick(n2ui.n2Diag.zoomedElement) }],
            ['#expand-all', e => { n2ui.uncollapseButtonClick(n2ui.n2Diag.model.root) }],
            ['#collapse-element', e => { n2ui.collapseOutputsButtonClick(n2ui.n2Diag.zoomedElement) }],
            ['#collapse-element-2',function() { n2ui.collapseOutputsButtonClick(n2ui.n2Diag.zoomedElement); self._setRootButton(this) }],
            ['#collapse-all', function() { n2ui.collapseOutputsButtonClick(n2ui.n2Diag.model.root); self._setRootButton(this) }],
            ['#expand-element', function() { n2ui.uncollapseButtonClick(n2ui.n2Diag.zoomedElement); self._setRootButton(this) }],
            ['#expand-all', function() { n2ui.uncollapseButtonClick(n2ui.n2Diag.model.root); self._setRootButton(this) }],
            ['#show-connections', e => { n2ui.n2Diag.showArrows(); self._setRootButton(this) }],
            ['#hide-connections', function() { n2ui.n2Diag.clearArrows(); self._setRootButton(this) }],
            ['#show-connections-2', function() { n2ui.n2Diag.showArrows(); self._setRootButton(this) }],
            ['#show-all-connections', function() { n2ui.n2Diag.showAllArrows(); self._setRootButton(this) }],
            ['#legend-button', e => { n2ui.toggleLegend() }],
            ['#linear-solver-button', e => { n2ui.toggleSolverNamesCheckboxChange() }],
            ['#save-button', e => { n2ui.n2Diag.saveSvg() }],
            ['#info-button', e => { n2ui.toggleNodeData() }],
            ['#hide-toolbar', e => { self.toggle() }],
            ['#question-button', DisplayModal ]
        ];

        for (let evt of clickEventArray) {
            d3.select(evt[0]).on('click', evt[1]);
        }

        // The font size slider is a range input
        this.toolbar.select('#text-slider').on('input', function() {
            const fontSize = this.value;
            n2ui.n2Diag.fontSizeSelectChange(fontSize);

            const fontSizeIndicator = self.toolbar.select('#font-size-indicator');
            fontSizeIndicator.html(fontSize + ' px');
        });

        // The model height slider is a range input
        this.toolbar.select('#model-slider').on('mouseup', function() {
            const modelHeight = parseInt(this.value);
            n2ui.n2Diag.verticalResize(modelHeight);
        });
    }
}