/**
 * Manage the cursor state and process click events based on it.
 * @typedef ClickHandler
 */
class ClickHandler {
    static ClickEffect = {
        Normal: 0,
        NodeInfo: 1,
        Collapse: 2,
        Filter: 3
    };

    /**
     * Set up the initial mode and define parameters for the other modes. 
     * @param {NodeInfo} nodeInfoBox Reference to the NodeInfo window to activate/deactivate it.
     */
    constructor() {
        // This tracks the current mode:
        this.clickEffect = ClickHandler.ClickEffect.Normal;

        this.modeData = {
            nodeinfo: {
                val: ClickHandler.ClickEffect.NodeInfo,
                icon: 'i.icon-info-circle',
                cursor: 'node-data-cursor',
                obj: null // to be set to nodeInfoBox later
            },
            collapse: {
                val: ClickHandler.ClickEffect.Collapse,
                icon: 'i.icon-collapse-target',
                cursor: 'collapse-mode-cursor',
                obj: null
            },
            filter: {
                val: ClickHandler.ClickEffect.Filter,
                icon: 'i.icon-filter-target',
                cursor: 'filter-mode-cursor',
                obj: null
            }
        };
    }

    update(nodeInfoBox) { this.modeData.nodeinfo.obj = nodeInfoBox; }

    get isNormal() { return this.clickEffect == ClickHandler.ClickEffect.Normal; }
    get isNodeInfo() { return this.clickEffect == ClickHandler.ClickEffect.NodeInfo; }
    get isCollapse() { return this.clickEffect == ClickHandler.ClickEffect.Collapse; }
    get isFilter() { return this.clickEffect == ClickHandler.ClickEffect.Filter; }

    /** Make sure the string used as a mode name is recognized */
    _validateMode(modeName, funcName) {
        if (! modeName in this.modeData)
            throw(`Unknown mode name '${modeName}' passed to ClickHandler.${funcName}().`)
    }

    /**
     * Color the active icon, change the mouse pointer, and update the mode state.
     * @param {String} modeName The name of the mode to activate.
     * @returns Reference to this ClickHandler object.
     */
    activate(modeName) {
        this._validateMode(modeName, 'activate');

        const m = this.modeData[modeName];
        d3.selectAll(m.icon).classed('active-tab-icon', true);
        d3.select('#all-diagram-content').classed(m.cursor, true);
        this.clickEffect = m.val;
        if (m.obj) { m.obj.activate(); }

        return this;
    }

    /**
     * Return the icon, mouse pointer, and mode state to the default.
     * @param {String} modeName The name of the mode to deactivate.
     * @returns Reference to this ClickHandler object.
     */
    deactivate(modeName) {
        this._validateMode(modeName, 'deactivate');

        const m = this.modeData[modeName];
        d3.selectAll(m.icon).classed('active-tab-icon', false);
        d3.select('#all-diagram-content').classed(m.cursor, false);
        this.clickEffect = ClickHandler.ClickEffect.Normal;
        if (m.obj) { m.obj.deactivate(); }

        return this;
    }

    /**
     * Iterate over all modes and run deactivate() on each.
     * @returns Reference to this ClickHandler object.
     */
    deactivateAll() {
        for (const modeName in this.modeData) {
            this.deactivate(modeName);
        }

        return this;
    }

    /**
     * If the specified mode is not active, activate it; if another non-default mode
     * is active, deactivate it first. If the specified mode is active, deactivate it.
     * @param {String} modeName Name of the mode to toggle.
     * @returns Reference to this ClickHandler object.
     */
    toggle(modeName) {
        this._validateMode(modeName, 'toggle');
        
        if (this.clickEffect == this.modeData[modeName].val) { return this.deactivate(modeName); }

        if (this.clickEffect != ClickHandler.ClickEffect.Normal) { this.deactivateAll(); }
        
        return this.activate(modeName);
    }
}
