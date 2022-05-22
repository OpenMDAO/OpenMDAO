/**
 * @typedef {Object} SymbolType
 * @property {string} name What the symbol is called.
 */
class SymbolType {

    /**
     * Determine the name and whether it's a declared partial based on info
     * from the provided node.
     * @param {MatrixCell} cell The object to select the type from.
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
     * @param {MatrixCell} cell The cell to operate on.
     */
    getType(cell) {
        if (cell.srcObj.isFilter() || cell.tgtObj.isFilter()) {
            this.name = 'filter';
        }
        else if (cell.srcObj.isGroup() || cell.tgtObj.isGroup()) {
            this.name = 'group';
        }
        else if (cell.onDiagonal()) {
            this.name = 'vector';
        }
        else if (cell.srcObj.isInputOrOutput() || cell.tgtObj.isInputOrOutput() ) {
            this.name = 'vectorVector';
        }
        else {
            console.warn("Completely unrecognized symbol type for cell ", cell)
            throw ("Completely unrecognized symbol type.")
        }
    }
}
