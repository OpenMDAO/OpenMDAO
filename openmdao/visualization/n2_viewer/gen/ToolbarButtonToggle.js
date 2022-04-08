// <<hpp_insert gen/ToolbarButtonClick.js>>

/**
 * Manage toolbar buttons that alternate states when clicked.
 * @typedef ToolbarButtonToggle
 * @property {Object} tooltipBox A reference to the tool-tip element.
 * @property {Array} tooltips One or two tooltips to display.
 * @property {Object} toolbarButton A reference to the toolbar button.
 * @property {Function} clickFn The function to call when clicked.
 * @property {Function} predicateFn Function returning a boolean representing the state.
 */
class ToolbarButtonToggle extends ToolbarButtonClick {
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
    click(e, target) {
        this.clickFn(e, target);

        this.tooltipBox
            .text(this.predicateFn() ? this.tooltips[0] : this.tooltips[1])
            .style("visibility", "visible");

    }
}
