// <<hpp_insert gen/Window.js>>
// <<hpp_insert gen/ValueInfo.js>>

/**
 * Manage a table containing all available metadata properties for
 * the currently active node, as well as whether the table is
 * visible or not.
 * @typedef NodeInfo
 */
class NodeInfo extends Window {
    /**
     * Build a list of the properties we care about and set up
     * references to the HTML elements.
     * @param {UserInterface} ui Reference to the diagram's UserInterface object.
     * @param {Boolean} [addDefaultProps = true] Whether to add default properties to be shown.
     */
    constructor(ui, addDefaultProps = true) {
        super('nodeInfo-' + uuidv4());
        this.values = {};
    
        this.ui = ui;
        this.table = this.body.append('table').attr('class', 'node-info-table');
        this.tbody = this.table.append('tbody');
        this.dataDiv = this.main.append('div').attr('class', 'node-info-data');
        this.theme('node-info');
        this.hideCloseButton();

        // Potential properties
        if (addDefaultProps) {
            this.propList = [
                new InfoPropDefault('path', 'Absolute Name'),
                new InfoPropDefault('class', 'Class'),
                new InfoPropDefault('type', 'Type', true),
                new InfoPropArray('val', 'Val', this.values),
            ];
        }
        else {
            this.propList = null;
        }

        // Becomes active when node info mode is selected on toolbar
        this.active = false;
    }

    /** Make the info box visible if it's hidden */
    activate() {
        this.active = true;
        this.hidden = false;
        return this;
    }

    /** Make the info box hidden if it's visible */
    deactivate() {
        this.clear();
        this.active = false;
        this.hidden = true;
        return this;
    }

    /** Toggle the active mode */
    toggle() {
        if (this.active) return this.deactivate();

        return this.activate();
    }

    /** Create a new persistent window by copying our contents */
    pin() {
        if (this.tbody.html() == '') return; // Was already pinned so is empty

        new PersistentNodeInfo(this);
        this.hidden = true;
        this.clear();
        return this;
    }

    /** Place the window near the mouse in a semi-intelligent manner */
    moveNearMouse(event, offset = 15) {
        if (!this.active) return this;

        return super.moveNearMouse(event, offset);
    }

    /** Wipe the contents of the table body */
    clear() {
        if (!this.active) return;
        // this.hidden = true;

        this.setList({
            width: null, height: null, left: null, right: null,
            top: null, bottom: null
        })
        this.dataDiv.html('');
        this.tbody.html('');

        // Don't just replace with {} because some InfoProps rely
        // on the reference to this.values:
        // wipeObj(this.values);
        return this;
    }

    /**
     * Iterate over the list of known properties and display them
     * if the specified object contains them.
     * @param {Object} event The related event so we can get position.
     * @param {TreeNode} node The node to examine.
     * @param {String} color Match the color of the node for the header/footer.
     */
    update(event, node, color) {
        if (!this.active) return;

        this.clear();

        this.name = node.path;
        this.ribbonColor(color);

        if (DebugFlags.info && node.hasChildren()) {
            InfoPropDefault.addRowWithVal(this.tbody, 'Children', node.children.length);
            InfoPropDefault.addRowWithVal(this.tbody, 'Descendants', node.numDescendants);
            InfoPropDefault.addRowWithVal(this.tbody, 'Leaves', node.draw.numLeaves);
            InfoPropDefault.addRowWithVal(this.tbody, 'Manually Expanded', node.draw.manuallyExpanded.toString());
        }

        for (const prop of this.propList) { prop.addRow(this.tbody, node); }

        this.sizeToContent()
            .title(node.name)
            .moveNearMouse(event)
            .show();
    }
}

/**
 * Make a persistent copy of the NodeInfo panel and handle its drag/close events
 * @typedef PersistentNodeInfo
 */
class PersistentNodeInfo extends WindowDraggable {
    constructor(nodeInfo) {
        super('persistentNodeInfo-' + uuidv4(), '#' + nodeInfo.window.attr('id'));

        // Avoid just copying the reference because nodeInfo.values will be wiped:
        this.values = JSON.parse(JSON.stringify(nodeInfo.values));
        this.ui = nodeInfo.ui;

        this._setupShowMoreButtons(nodeInfo.name)
            ._setupCopyButtons()
            .showCloseButton()
            .show();

        this.tooltipBox = d3.select(".tool-tip");
        this.closeButton
            .on("mouseover", this.mouseOver.bind(this))
            .on("mouseleave", this.mouseLeave.bind(this))
            .on("mousemove", this.mouseMove.bind(this));
    }

    /** When the mouse enters the element, show the tool tip */
    mouseOver() {
        this.tooltipBox
            .text('Shift-click to close all.')
            .style("visibility", "visible");
    }

    /** When the mouse leaves the element, hide the tool tip */
    mouseLeave() {
        this.tooltipBox.style("visibility", "hidden");
    }

    /** Keep the tool-tip near the mouse */
    mouseMove(e) {
        this.tooltipBox.style("top", (e.pageY - 30) + "px")
            .style("left", (e.pageX + 5) + "px");
    }

    /** Set up event handlers for any "Show More" buttons in the panel */
    _setupShowMoreButtons(name) {
        const self = this;

        for (const valName in this.values) {
            if (this.values[valName].isTruncated) {
                this.window
                    .select(`button#${valName}.show_value_button`)
                    .on('click', c => {
                        const values = self.values[valName];
                        ValueInfo.add(name, values.val, values.min, values.max, self);
                    })
            }
        }

        return this;
    }

    /** Set up event handlers for any "Copy" buttons in the panel */
    _setupCopyButtons() {
        for (const valName in this.values) {
            this.window
                .select(`button#${valName}.copy_value_button`)
                .on('click', c => {
                    const copyText = d3.select("#input-for-pastebuffer");
                    copyText.text(this.values[valName].copyStr);
                    copyText.node().select();
                    document.execCommand('copy');
                })
        }

        return this;
    }

    /** Override so that shift-click closes all PersistentNodeInfo windows. */
    close(e) {
        this.tooltipBox.style("visibility", "hidden");
        window.getSelection().empty(); // Shift-clicking also selects text, so unselect it

        if (e.shiftKey) {
            const allPNIWin = d3.selectAll('[id^="persistentNodeInfo-"]');

            allPNIWin.select('.window-close-button')
                .on('click', null)
                .on("mouseover", null)
                .on("mouseleave", null)
                .on("mousemove", null);
            allPNIWin.remove();
        }
        else {
            this.closeButton
                .on("mouseover", null)
                .on("mouseleave", null)
                .on("mousemove", null);
            super.close(e);
        }
    }
}
