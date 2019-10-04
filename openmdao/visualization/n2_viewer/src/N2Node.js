/**
 * A visible node that is part of the matrix.
 * @typedef {Object} N2Node
 * @property {number} row Vertical coordinate of the node in the matrix.
 * @property {number} col Horizontal coordinate of the node in the matrix.
 * @property {Object} srcObj The node in the model tree this node is associated with.
 * @property {Object} tgtObj The model tree node that this one outputs to.
 * @property {string} id The srcObj id appended with the tgtObj id.
 * @property {SymbolType} symbolType Info about the type of symbol represented by the node.
*/
class N2Node {

    /**
     * Initialize the node.
     * @param {number} row Vertical coordinate of the node in the matrix.
     * @param {number} col Horizontal coordinate of the node in the matrix.
     * @param {Object} srcObj The node in the model tree this node is associated with.
     * @param {Object} tgtObj The model tree node that this one outputs to.
     * @param {ModelData} model Reference to the model to get some info from it.
    */
    constructor(row, col, srcObj, tgtObj, model) {
        this.row = row;
        this.col = col;
        this.srcObj = srcObj;
        this.tgtObj = tgtObj;
        this.id = srcObj.id + "_" + tgtObj.id;

        this.symbolType = new SymbolType(this, model);
    }

    /**
     * Determine if this node is on the main diagonal of the matrix.
     * @return {Boolean} True if row equals column.
    */
    onDiagonal() {
        return (this.row == this.col);
    }

}