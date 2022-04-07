/**
 * Base class for toolbar button events. Show, hide, or
 * move the tool tip box.
 * @typedef ToolbarButtonNoClick
 * @property {Object} tooltipBox A reference to the tool-tip element.
 * @property {Array} tooltips One or two tooltips to display.
 * @property {Object} toolbarButton A reference to the toolbar button.
 */
class ToolbarButtonNoClick {
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
    mouseOver(e) {
        this.tooltipBox
            .text(this.tooltips[0])
            .style("visibility", "visible");
    }

    /** When the mouse leaves the element, hide the tool tip */
    mouseLeave(e) {
        this.tooltipBox.style("visibility", "hidden");
    }

    /** Keep the tool-tip near the mouse */
    mouseMove(e) {
        this.tooltipBox.style("top", (e.pageY - 30) + "px")
            .style("left", (e.pageX + 5) + "px");
    }

    /**
     * Use when the info displayed on the help screen is different than the tooltip.
     * @param {String} helpText The info to display on the help screen for this button.
     * @returns {ToolbarButtonNoClick} Reference to this.
     */
    setHelpInfo(helpText) {
        this.help = helpText;
        return this;
    }

    /**
     * Grab all the info about the button that will help with generating the help screen.
     */
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
