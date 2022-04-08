/**
 * @typedef {Object} SymbolType
 * @property {string} name What the symbol is called.
 */
class SymbolType {

    /**
     * Determine the name and whether it's a declared partial based on info
     * from the provided node.
     * @param {N2MatrixCell} cell The object to select the type from.
     * @param {ModelData} model Reference to the model to get some info from it.
     */
    constructor(cell, model) {
        this.name = null;

        this._init();

        // Update properties based on the the referenced node.
        this.getType(cell, model);
    }

    /**
     * Separate these calls from the constructor so that subclasses can
     * set values before execution.
     */
    _init() { }

    /** 
     * Decide what object the cell will be drawn as, based on its position
     * in the matrix, type, source, target, and/or other conditions.
     * @param {N2MatrixCell} cell The cell to operate on.
     * @param {ModelData} model Reference to the entire model.
     */
    getType(cell, model) { }
}
