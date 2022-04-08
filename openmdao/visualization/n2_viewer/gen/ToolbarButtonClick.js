// <<hpp_insert gen/ToolbarButtonNoClick.js>>

/**
 * Manage clickable toolbar buttons
 * @typedef ToolbarButtonClick
 * @property {Object} tooltipBox A reference to the tool-tip element.
 * @property {Array} tooltips One or two tooltips to display.
 * @property {Object} toolbarButton A reference to the toolbar button.
 * @property {Function} clickFn The function to call when clicked.
 */
class ToolbarButtonClick extends ToolbarButtonNoClick {
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

        this.toolbarButton.on('click', function (e) { self.click(e, this); });
    }

    /**
     * Defined separately so the derived class can override
     * @param {Object} target Reference to the HTML element that was clicked
     */
    click(e, target) {
        this.clickFn(e, target);
    }
}
