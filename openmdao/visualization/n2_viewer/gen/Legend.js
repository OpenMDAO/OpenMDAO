// <<hpp_insert gen/WindowDraggable.js>>

/**
 * Draw a symbol describing each of the element types.
 * @typedef Legend
 * @property {Boolean} shown Whether the legend is currently drawn or not.
 */
class Legend extends WindowDraggable {
    /**
     * Initializes the legend object.
     * @param {ModelData} modelData Symbols are only displayed if they're in the model
     */
    constructor(modelData) {
        super('n2win-legend');

        // TODO: The legend should't have to search through modelData itself,
        // this info can be collected as modelData is built.
        this.nodes = modelData.tree.children;
        this._initItemTypes();

        this.n2Symbols = [];

        const legendDiv = d3.select('#legend-div');
        legendDiv.style('visibility', null);
        this.body.node().appendChild(legendDiv.node());
        this._setDisplayBooleans(this.nodes);

        this.title('<span class="icon-key"></span>');
        this._setupContents();

        this.theme('legend');
        this.sizeToContent();
        this.move(50, -10);
    }

    /**
     * Set the types to check for in the model to determine if they
     * need to be displayed in the legend.
     */
    _initItemTypes() {
        this.showSysVar = {
            'group': false,
            'input': false,
            'unconnectedInput': false,
            'output': false,
            'collapsed': true,
            'connection': true
        };

        this.showSymbols = {
            'scalar': false,
            'vector': false,
            'collapsedVariables': false
        };

        this.sysAndVar = [
            { 'name': "Connection", 'color': Style.color.connection },
            { 'name': "Collapsed", 'color': Style.color.collapsed },
        ];
    }

    /** Override Window::close() to just hide the legend. */
    close() {
        d3.select('#legend-button').attr('class', 'fas icon-key');
        this.hide();
        return this;
    }

    /**
     * Determine which types of nodes are present in the model and only mark
     * those types of keys to be displayed in the legend.
     * @param {Object} nodes The model node tree.
     */
    _setDisplayBooleans(nodes) {
        for (const node of nodes) {
            const {
                group,
                input,
                unconnectedInput,
                output,
                collapsed,
                connection
            } = this.showSysVar;

            if (node.hasChildren()) {
                if (!this.showSysVar.group && node.isGroup()) {
                    this.showSysVar.group = true;
                    this.sysAndVar.push({
                        'name': 'Group',
                        'color': Style.color.group
                    })
                }
                this._setDisplayBooleans(node.children);
            }
            else {
                if (!this.showSysVar.input && node.isInput()) {
                    this.showSysVar.input = true;
                    this.sysAndVar.push({
                        'name': 'Input',
                        'color': Style.color.input
                    })
                }
                else if (!this.showSysVar.output && node.isOutput()) {
                    this.showSysVar.output = true;
                    this.sysAndVar.push({
                        'name': 'Output',
                        'color': Style.color.output
                    })
                }
                else if (!this.showSysVar.unconnectedInput && node.isUnconnectedInput()) {
                    this.showSysVar.unconnectedInput = true;
                    this.sysAndVar.push({
                        'name': 'Unconnected Input',
                        'color': Style.color.unconnectedInput
                    })
                }
            }
        }
    }

    /**
     * Create elements in the legend divs for the supplied item
     * @param {Object} item Contains the name and color of the item
     * @param {Object} container The div to append into
     */
    _addItem(item, container, cssClass = '') {
        const newDiv = container
            .append('div')
            .attr('class', 'legend-box-container');

        newDiv.append('div')
            .attr('class', `legend-box ${cssClass}`)
            .style('background-color', item.color);

        newDiv.append('p')
            .html(item.name);
    }

    /** Add symbols for all of the items that were discovered */
    _setupContents() {
        const sysVarContainer = d3.select('#sys-var-legend');
        for (let item of this.sysAndVar) this._addItem(item, sysVarContainer);

        sysVarContainer.style('width', sysVarContainer.node().scrollWidth + 'px')
    }
}
