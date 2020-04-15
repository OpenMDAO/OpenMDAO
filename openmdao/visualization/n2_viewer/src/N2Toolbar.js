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

        // There are a lot of simple click and mouse over/leave/move events, so iterate over this array of arrays.
        // There are two kinds of sub-arrays in this array.
        // For buttons that always have the same tooltip no matter the state of the application, the array is of length 3
        //    and has these elements:
        //        [ selector, click function, tooltip text ]
        // For buttons that have two different tooltips depending on the state of the application, the array is of length 5
        //    and has these elements:
        //        [ selector, click function, predicate function, tooltip text when predicate is true, tooltip text when predicate is false ]
        const clickEventArray = [
            ['#reset-graph', e => { n2ui.homeButtonClick() }, "View entire model starting from root"],
            ['#undo-graph', e => { n2ui.backButtonPressed() }, "Move back in view history"],
            ['#redo-graph', e => { n2ui.forwardButtonPressed()}, "Move forward in view history"],

            ['#collapse-element', e => { n2ui.collapseOutputsButtonClick(n2ui.n2Diag.zoomedElement)}, "Control variable collapsing" ],
            ['#collapse-element-2',function() { n2ui.collapseOutputsButtonClick(n2ui.n2Diag.zoomedElement); self._setRootButton(this) },
            	"Collapse only variables in current view"],
            ['#collapse-all', function() { n2ui.collapseOutputsButtonClick(n2ui.n2Diag.model.root); self._setRootButton(this) },
            	"Collapse all variables in entire model"],
            ['#expand-element', function() { n2ui.uncollapseButtonClick(n2ui.n2Diag.zoomedElement); self._setRootButton(this) },
            	"Expand only variables in current view"],
            ['#expand-all', function() { n2ui.uncollapseButtonClick(n2ui.n2Diag.model.root); self._setRootButton(this) },
            	"Expand all variables in entire model"],

            ['#show-connections', e => { n2ui.n2Diag.showArrows(); self._setRootButton(this) }, "Set connections visibility"],
            ['#hide-connections', function() { n2ui.n2Diag.clearArrows(); self._setRootButton(this) }, "Hide all connection arrows"],
            ['#show-connections-2', function() { n2ui.n2Diag.showArrows(); self._setRootButton(this) }, "Show pinned connection arrows"],
            ['#show-all-connections', function() { n2ui.n2Diag.showAllArrows(); self._setRootButton(this) }, "Show all connections in model"],

            ['#linear-solver-button', e => { n2ui.toggleSolverNamesCheckboxChange() }, pred => { return  n2ui.n2Diag.showLinearSolverNames },
            	"Show non-linear solvers", "Show linear solvers"],

            ['#legend-button', e => { n2ui.toggleLegend() }, pred => { return n2ui.legend.hidden }, "Show legend", "Hide legend"],

            ['#text-slider-button', null, "Set text height"],
            ['#depth-slider-button', null, "Set collapse depth"],
            ['#model-slider-button', null, "Set model height"],

            ['#save-button', e => { n2ui.n2Diag.saveSvg() }, "Save to SVG"],

            ['#info-button', e => { n2ui.toggleNodeData() }, pred => { return !d3.select('#info-button').attr('class').includes('active-tab-icon') },
            	"Show connection matrix information", "Hide connection matrix information"],

            ['#question-button', DisplayModal, pred => { return  parentDiv.querySelector("#myModal").style.display === "block"},
            	"Hide N2 diagram help", "Show N2 diagram help" ],

            ['#hide-toolbar', e => { self.toggle() }, pred => { return  this.hidden }, "Show toolbar", "Hide toolbar"],
        ];

        for (let evt of clickEventArray) {
            let toolbarButton = d3.select(evt[0]);
            let displayText = "";

            toolbarButton
                .on('click', function(d) {
                    evt[1]();
                    if (evt.length == 3) {
                        displayText = evt[2];
                    } else {
                        displayText = evt[2]() ? evt[3] : evt[4];
                    }
                    n2ui.n2Diag.dom.toolTip.text(displayText);
                })
                .on("mouseover", function(d) {
                    if (evt.length == 3) {
                        displayText = evt[2];
                    } else {
                        displayText = evt[2]() ? evt[3] : evt[4];
                    }
                    return n2ui.n2Diag.dom.toolTip.text(displayText).style("visibility", "visible");
                })
                .on("mouseleave", function(d) {
                    return n2ui.n2Diag.dom.toolTip.style("visibility", "hidden");
                })
                .on("mousemove", function() {
                    return n2ui.n2Diag.dom.toolTip.style("top", (d3.event.pageY - 30) + "px")
                            .style("left", (d3.event.pageX + 5) + "px");
                });
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