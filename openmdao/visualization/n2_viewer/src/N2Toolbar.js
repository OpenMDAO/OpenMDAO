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

        this.n2ui = n2ui;
        this.toolbarContainer = d3.select('#toolbarLoc');
        this.toolbar = d3.select('#true-toolbar');
        this.hideToolbarButton = d3.select('#hide-toolbar');
        this.hideToolbarIcon = this.hideToolbarButton.select('i');
        this.searchBar = d3.select('#awesompleteId');

        this.hidden = false;

        d3.select('#model-slider').node.value = sliderHeight;

        this._setupExpandableButtons();
        this._setupButtonFunctions();

        // Expand/contract the search bar
        d3.select('#searchbar-container')
            .on('mouseover', function() {
                self.searchBar.style('width', '200px');
                self.toolbarContainer.style('z-index', '5');
            })
            .on('mouseout', function() {
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

    _setupExpandableButtons() {
        const self = this;

        // Open expandable buttons when hovered over
        d3.selectAll('.expandable > div')
            .on('mouseover', function() {
                self.toolbarContainer.style('z-index', '5');
                d3.select(this).style('max-width', '200px');
            })
            .on('mouseout', function() {
                d3.select(this).style('max-width', '0');
                self.toolbarContainer.style('z-index', '1')
            })


        // When an expanded button is clicked, update the 'root' button
        // to the same icon and function.
        d3.selectAll('.toolbar-group-expandable > i')
            .on('click', function() {
                let container = d3.select(this.parentNode.parentNode);
                let button = d3.select(this);

                container.select('i')
                    .attr('class', button.attr('class'))
                    .attr('id', button.attr('id'))
                    .on('click', button.node.onclick);
            })
    }

    /** Associate all of the buttons on the toolbar with a method in N2UserInterface. */
    _setupButtonFunctions() {
        const self = this; // For callbacks that change "this". Alternative to using .bind().
        const n2ui = this.n2ui;

        this.toolbar.select('#reset-graph').on('click', function() {
            n2ui.homeButtonClick();
        });
        this.toolbar.select('#undo-graph').on('click', function() {
            n2ui.backButtonPressed();
        });
        this.toolbar.select('#redo-graph').on('click', function() {
            n2ui.forwardButtonPressed();
        });
        this.toolbar.select('#expand-element').on('click', function() {
            n2ui.uncollapseButtonClick(n2ui.n2Diag.zoomedElement);
        });
        this.toolbar.select('#expand-all').on('click', function() {
            n2ui.uncollapseButtonClick(n2ui.n2Diag.model.root);
        });
        this.toolbar.select('#collapse-element').on('click', function() {
            n2ui.collapseOutputsButtonClick(n2ui.n2Diag.zoomedElement);
        });
        this.toolbar.select('#collapse-element-2').on('click', function() {
            n2ui.collapseOutputsButtonClick(n2ui.n2Diag.zoomedElement);
        });
        this.toolbar.select('#collapse-all').on('click', function() {
            n2ui.collapseOutputsButtonClick(n2ui.n2Diag.model.root);
        });
        this.toolbar.select('#expand-element').on('click', function() {
            n2ui.uncollapseButtonClick(n2ui.n2Diag.zoomedElement);
        });
        this.toolbar.select('#expand-all').on('click', function() {
            n2ui.uncollapseButtonClick(n2ui.n2Diag.model.root);
        });
        this.toolbar.select('#collapse-element').on('click', function() {
            n2ui.collapseOutputsButtonClick(n2ui.n2Diag.zoomedElement);
        });
        this.toolbar.select('#collapse-all').on('click', function() {
            n2ui.collapseOutputsButtonClick(n2ui.n2Diag.model.root);
        });
        this.toolbar.select('#hide-connections').on('click', function() {
            n2ui.n2Diag.clearArrows();
        });
        this.toolbar.select('#show-connections').on('click', function() {
            n2ui.n2Diag.showArrows();
        });
        this.toolbar.select('#show-all-connections').on('click', function() {
            n2ui.n2Diag.showAllArrows();
        });
        this.toolbar.select('#legend-button').on('click', function() {
            n2ui.toggleLegend();
        });
        this.toolbar.select('#linear-solver-button').on('click', function() {
            n2ui.toggleSolverNamesCheckboxChange();
        });

        this.toolbar.select('#text-slider').on('input', function(e) {
            const fontSize = e.target.value;
            n2ui.n2Diag.fontSizeSelectChange(fontSize);

            const fontSizeIndicator = self.toolbar.select('#font-size-indicator');
            fontSizeIndicator.attr('innerHTML', fontSize + ' px');
        });

        this.toolbar.select('#model-slider').on('mouseup', function(e) {
            const modelHeight = parseInt(e.target.value);
            n2ui.n2Diag.verticalResize(modelHeight);
        });

        this.toolbar.select('#save-button').on('click', function() {
            n2ui.n2Diag.saveSvg();
        });

        this.toolbar.select('#info-button').on('click', function() {
            n2ui.toggleNodeData();
        });

        d3.select('#question-button').on('click', DisplayModal);

        this.hideToolbarButton.on('click', function () {
            if (self.hidden) { self.show(); }
            else { self.hide(); }
        })
    }
}