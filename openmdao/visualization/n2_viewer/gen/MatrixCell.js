// <<hpp_insert gen/CellRenderer.js>>

/**
 * A visible cell in the matrix grid.
 * @typedef {Object} MatrixCell
 * @property {number} row Vertical coordinate of the cell in the matrix.
 * @property {number} col Horizontal coordinate of the cell in the matrix.
 * @property {TreeNode} srcObj The node in the model tree this cell is associated with.
 * @property {TreeNode} tgtObj The model tree node that this outputs to.
 * @property {string} id The srcObj id appended with the tgtObj id.
 * @property {SymbolType} symbolType Info about the type of symbol represented by the node.
 * @property {CellRenderer} renderer The object that draws the cell.
 */
class MatrixCell {
    /**
     * Initialize the cell.
     * @param {Number} row Vertical coordinate of the cell in the matrix.
     * @param {Number} col Horizontal coordinate of the cell in the matrix.
     * @param {TreeNode} srcObj The node in the model tree this node is associated with.
     * @param {TreeNode} tgtObj The model tree node that this outputs to.
     * @param {ModelData} model Reference to the model to get some info from it.
     */
    constructor(row, col, srcObj, tgtObj, model) {
        this.row = row;
        this.col = col;
        this.srcObj = this.obj = srcObj;
        this.tgtObj = tgtObj;
        this.id = MatrixCell.makeId(srcObj.id, tgtObj.id);

        this._setSymbolType(model);
        this.renderer = this._newRenderer();

        this.offScreen = {
            "top": {
                "incoming": new Set(),
                "outgoing": new Set()
            },
            "bottom": {
                "incoming": new Set(),
                "outgoing": new Set()
            },
            "total": 0
        }
    }

    _setSymbolType(model) {
        this.symbolType = new SymbolType(this, model);
    }

    static makeId(srcId, tgtId = null) {
        if (! tgtId || srcId == tgtId) return "node_" + srcId;
        
        return "conn_" + srcId + "_to_" + tgtId;
    }

    /**
     * Determine if this node is on the main diagonal of the matrix.
     * @return {Boolean} True if row equals column.
     */
    onDiagonal() {
        return (this.row == this.col);
    }

    /**
     * Determine if this node is in the upper-right triangle of the matrix.
     * @return {Boolean} True if column is greater than row.
     */
    inUpperTriangle() {
        return (this.col > this.row);
    }

    /**
     * Determine if this node is in the lower-left triangle of the matrix.
     * @return {Boolean} True if row is greater than column.
     */
    inLowerTriangle() {
        return (this.row > this.col);
    }

    /**
     * Select the mouseover callback depending on whether we're on the diagonal.
     * TODO: Remove these globals
     */
    getMouseoverFunc() {
        return (this.onDiagonal() ? n2MouseFuncs.overOnDiag : n2MouseFuncs.overOffDiag);
    }

    /**
    * Select the mousemove callback depending on whether we're on the diagonal.
    * TODO: Remove these globals
    */
    getMousemoveFunc() {
        return (this.onDiagonal() ? n2MouseFuncs.moveOnDiag : null);
    }

    /**
     * Choose a color based on our location and state of the associated TreeNode.
     */
    color() {
        if (this.onDiagonal()) {
            if (this.obj.draw.minimized) return Style.color.collapsed;
            if (this.obj.isConnectedInput()) return Style.color.input;
            if (this.obj.isOutput()) return Style.color.output;
            if (this.obj.isUnconnectedInput()) return Style.color.unconnectedInput;
        }

        return Style.color.connection;
    }

    /**
     * A connection going "off-screen" was detected between two nodes.
     * Determine whether the arrow should be in the top or bottom section of the
     * matrix based on rootIndex, and add to the appropriate array of
     * tracked offscreen connections.
     * @param {TreeNode} srcNode Where the connection starts.
     * @param {TreeNode} tgtNode Where the connection ends.
     */
    addOffScreenConn(srcNode, tgtNode) {
        let offscreenNode = null, dir = '', side = '';

        if (srcNode === this.tgtObj) {
            dir = 'outgoing';
            offscreenNode = tgtNode;           
        }
        else {
            dir = 'incoming';
            offscreenNode = srcNode;
        }
        side = (srcNode.rootIndex < tgtNode.rootIndex)? 'top' : 'bottom';

        const offScreenSet = this.offScreen[side][dir];

        if (!offScreenSet.has(offscreenNode)) {
            offScreenSet.add(offscreenNode);
            this.offScreen.total++;
        }
    }

    /** Choose a renderer based on our SymbolType. */
    _newRenderer() {
        if (this.color() == Style.color.connection) {
            if (this.inUpperTriangle()) return new ConnectorUpper(this.color(), this.id);

            return new ConnectorLower(this.color(), this.id)
        }

        const color = this.color();

        switch (this.symbolType.name) {
            case "scalar":
                return new ScalarCell(color, this.id);
            case "vector":
                return new VectorCell(color, this.id);
            case "group":
                return new GroupCell(color, this.id);
            case "filter":
                return new FilterCell(color, this.id);
            default:
                return null;
        }
    }

    /**
     * Highlight the variable nodes *associated* with the cell, not the cell
     * itself. The default is for cells on the diagonal to highlight the
     * variable directly across from them.
     * @param {String} [varType = 'self'] Either 'self', 'source', or 'target'
     *   to indicate the variable name to highlight.
     * @param {String} [direction = 'self'] Either 'self', 'input', or 'output'
     *   to indicate the style of the highlighting.
     */
    highlight(varType = 'self', direction = 'self') {

        let obj = (varType == 'target') ? this.tgtObj : this.srcObj;
        if (obj.draw.filtered) { obj = obj.draw.filterParent; }
        const treeId = TreeNode.pathToId(obj.path);
        const treeNode = d3.select('rect#' + treeId);

        let fill = treeNode.style('fill');
        if (direction == 'input') fill = Style.color.inputArrow;
        else if (direction == 'output') fill = Style.color.outputArrow;

        d3.select('#highlight-bar').append('rect')
            .attr('x', 0)
            .attr('y', treeNode.node().parentNode.transform.baseVal[0].matrix.f)
            .attr('rx', 4)
            .attr('ry', 4)
            .attr('width', 8)
            .attr('height', treeNode.attr('height'))
            .attr('stroke', Style.color.treeStroke)
            .attr('fill', fill);
    }
}
