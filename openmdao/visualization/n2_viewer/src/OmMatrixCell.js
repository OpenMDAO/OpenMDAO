// <<hpp_insert gen/MatrixCell.js>>
// <<hpp_insert src/OmCellRenderer.js>>
// <<hpp_insert src/OmSymbolType.js>>

/**
 * A visible cell in the matrix grid.
 * @typedef {Object} OmMatrixCell
 * @property {number} row Vertical coordinate of the cell in the matrix.
 * @property {number} col Horizontal coordinate of the cell in the matrix.
 * @property {OmTreeNode} srcObj The node in the model tree this cell is associated with.
 * @property {OmTreeNode} tgtObj The model tree node that this outputs to.
 * @property {string} id The srcObj id appended with the tgtObj id.
 * @property {OmSymbolType} symbolType Info about the type of symbol represented by the node.
 * @property {CellRenderer} renderer The object that draws the cell.
 */
class OmMatrixCell extends MatrixCell {
    /**
     * Initialize the cell.
     * @param {number} row Vertical coordinate of the cell in the matrix.
     * @param {number} col Horizontal coordinate of the cell in the matrix.
     * @param {OmTreeNode} srcObj The node in the model tree this node is associated with.
     * @param {OmTreeNode} tgtObj The model tree node that this outputs to.
     * @param {ModelData} model Reference to the model to get some info from it.
     */
    constructor(row, col, srcObj, tgtObj, model) {
        super(row, col, srcObj, tgtObj, model);
    }

    _setSymbolType(model) {
        this.symbolType = new OmSymbolType(this, model);
    }

    /**
     * Choose a color based on our location and state of the associated OmTreeNode.
     */
    color() {
        if (this.symbolType.potentialDeclaredPartial &&
            this.symbolType.declaredPartial) return OmStyle.color.declaredPartial;

        if (this.onDiagonal()) {
            if (this.obj.draw.minimized) return OmStyle.color.collapsed;
            if (this.obj.isAutoIvcInput()) return OmStyle.color.autoivcInput;
            if (this.obj.isConnectedInput()) return OmStyle.color.input;
            if (this.obj.isUnconnectedInput()) return OmStyle.color.unconnectedInput;
            return (this.obj.implicit) ?
                OmStyle.color.outputImplicit :
                OmStyle.color.outputExplicit;
        }

        return OmStyle.color.connection;
    }

    /** Choose a renderer based on our SymbolType. */
    _newRenderer() {
        const renderer = super._newRenderer();
        if (renderer) return renderer;

        const color = this.color();

        switch (this.symbolType.name) {
            case "scalarScalar":
                return new OmScalarScalarCell(color, this.id);
            case "scalarVector":
                return new OmScalarVectorCell(color, this.id);
            case "vectorScalar":
                return new OmVectorScalarCell(color, this.id);
            case "vectorVector":
                return new OmVectorVectorCell(color, this.id);
            case "scalarGroup":
                return new OmScalarGroupCell(color, this.id);
            case "groupScalar":
                return new OmGroupScalarCell(color, this.id);
            case "vectorGroup":
                return new OmVectorGroupCell(color, this.id);
            case "groupVector":
                return new OmGroupVectorCell(color, this.id);
            case "groupGroup":
                return new OmGroupGroupCell(color, this.id);
            default:
                throw(`No known renderer for ${this.symbolType.name}`);
        }
    }
}
