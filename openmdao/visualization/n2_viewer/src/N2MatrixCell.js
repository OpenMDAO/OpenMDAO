/** Base class for all cell renderers */
class N2CellRenderer {
    /**
     * Set values shared by objects of all derived class types.
     * @param {string} color The color to render all shapes in.
     * @param {string} className The CSS class to tag primary shapes with, used for selecting.
     */
    constructor(color, className, id) {
        this.color = color;
        this.className = className;
        this.id = 'cellShape_' + id; // To ensure it doesn't start with a number
    }

    static updateDims(baseWidth, baseHeight) {
        if (!N2CellRenderer.dims) {
            N2CellRenderer.prevDims = {
                "size": {
                    "width": 0,
                    "height": 0
                },
                "bottomRight": {
                    "x": 0,
                    "y": 0
                },
                "topLeft": {
                    "x": 0,
                    "y": 0
                }
            }
        }
        else {
            for (let prop of ["size", "bottomRight", "topLeft"]) {
                Object.assign(N2CellRenderer.prevDims[prop],
                    N2CellRenderer.dims[prop]);
            }
        }

        N2CellRenderer.dims = {
            "size": {
                "width": baseWidth,
                "height": baseHeight
            },
            "bottomRight": {
                "x": baseWidth * .5,
                "y": baseHeight * .5
            },
            "topLeft": {
                "x": baseWidth * -.5,
                "y": baseHeight * -.5
            }
        }
    }

    /** Enable access to the static dims variable through "this". */
    get dims() {
        return N2CellRenderer.dims;
    }

    /** Enable access to the static prevDims variable through "this". */
    get prevDims() {
        return N2CellRenderer.prevDims;
    }

    /** Act like an abstract base class force derived classes to define. */
    update() {
        throw ("ERROR: N2CellRenderer.update() called.")
    }

    /** Act like an abstract base class force derived classes to define. */
    render() {
        throw ("ERROR: N2CellRenderer.render() called.")
    }

    /** Reposition an SVG element based on dimensions of the current cell size. */
    updateCurrent(svgGroup) {
        return this.update(svgGroup, this.dims);
    }

    /** Reposition an SVG element based on dimensions of the previous cell size. */
    updatePrevious(svgGroup) {
        return this.update(svgGroup, this.prevDims);
    }

    /** Add an SVG element based on dimensions of the current cell size. */
    renderCurrent(svgGroup) {
        return this.render(svgGroup, this.dims);
    }

    /** Add an SVG element based on dimensions of the previous cell size. */
    renderPrevious(svgGroup) {
        return this.render(svgGroup, this.prevDims);
    }
}

/** Draws/updates an SVG circle for scalar types, with a transition animation. */
class N2ScalarBase extends N2CellRenderer {

    /**
     * Invoke the superclass constructor with these values and "sMid" as a CSS class.
     * @param {Object} dims Layout and dimensions for the current cell spec.
     * @param {Object} prevDims Layout and dimensions for the previous cell spec.
     * @param {string} color The color to render all shapes in.
     */
    constructor(color, id) {
        super(color, "sMid", id);
    }

    /**
     * Select the element with D3 if not already done, attach a transition
     * and resize the shape.
     * @param svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while resizing/repositioning.
     * @param {selection} [d3Elem = null ] The selection created in render().
     */
    update(svgGroup, dims, d3Elem = null) {
        if (!d3Elem) d3Elem = d3.select(svgGroup).select("." + this.className)
            .transition(sharedTransition);
        return d3Elem
            .attr("rx", dims.bottomRight.x * .6)
            .attr("ry", dims.bottomRight.y * .6);
    }

    /** 
     * Get the D3 selection for the appropriate group and append a filled ellipse.
     * @param {Object} svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while rendering.
     */
    render(svgGroup, dims) {
        let d3Elem = d3.select(svgGroup)
            .append("ellipse")
            .attr("class", this.className)
            .attr("id", this.id)
            .style("fill", this.color);

        return this.update(svgGroup, dims, d3Elem)
    }
}

/** Draws/updates an SVG rect for vector types, with a transition animation. */
class N2VectorBase extends N2CellRenderer {

    /**
     * Invoke the superclass constructor with these values and "vMid" as a CSS class.
     * @param {Object} dims Layout and dimensions for the current cell spec.
     * @param {Object} prevDims Layout and dimensions for the previous cell spec.
     * @param {string} color The color to render all shapes in.
     */
    constructor(color, id) {
        super(color, "vMid", id);
    }

    /**
     * Select the element with D3 if not already done, attach a transition
     * and resize the shape.
     * @param svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while resizing/repositioning.
     * @param {selection} [d3Elem = null ] The selection created in render().
     */
    update(svgGroup, dims, d3Elem = null) {
        if (!d3Elem) d3Elem = d3.select(svgGroup).select("." + this.className)
            .transition(sharedTransition);

        let ret = d3Elem
            .attr("x", dims.topLeft.x * .6)
            .attr("y", dims.topLeft.y * .6)
            .attr("width", dims.bottomRight.x * 1.2)
            .attr("height", dims.bottomRight.y * 1.2);

        return ret;
    }

    /** 
     * Get the D3 selection for the appropriate group and append a filled rectangle.
     * @param {Object} svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while rendering.
     */
    render(svgGroup, dims) {
        let d3Elem = d3.select(svgGroup).append('rect')
            .attr("class", this.className)
            .attr("id", this.id)
            .style("fill", this.color);

        return this.update(svgGroup, dims, d3Elem);
    }
}

class N2Connector extends N2CellRenderer {
    /**
     * Invoke the superclass constructor with these values and "vMid" as a CSS class.
     * @param {Object} dims Layout and dimensions for the current cell spec.
     * @param {Object} prevDims Layout and dimensions for the previous cell spec.
     * @param {string} color The color to render all shapes in.
     */
    constructor(color, id) {
        super(color, "vMid", id);
    }

    _transform(scale) { throw ("ERROR: N2Connector._transform() called.") }

    /**
     * Select the element with D3 if not already done, attach a transition
     * and resize the shape.
     * @param svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while resizing/repositioning.
     * @param {selection} [d3Elem = null ] The selection created in render().
     */
    update(svgGroup, dims, d3Elem = null) {
        if (!d3Elem) d3Elem = d3.select(svgGroup).select("." + this.className)
            .transition(sharedTransition);

        let ret = d3Elem.attr("transform", this._transform(dims.size.width/10.0));

        return ret;
    }

    /** 
     * Get the D3 selection for the appropriate group and append a filled arrow.
     * @param {Object} svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while rendering.
     */
    render(svgGroup, dims) {
        let d3Elem = d3.select(svgGroup).append('use')
            .attr("class", this.className)
            .attr("id", this.id)
            .style("fill", this.color)
            .attr("x", -5)
            .attr("y", -5)
            .attr("xlink:href", "#matrix-connector-square")

        return this.update(svgGroup, dims, d3Elem);
    }
}

/** Draws/updates an SVG rect with a border for group types, with a transition animation. */
class N2GroupBase extends N2CellRenderer {
    /**
     * Invoke the superclass constructor with these values and "gMid" as a CSS class.
     * @param {Object} dims Layout and dimensions for the current cell spec.
     * @param {Object} prevDims Layout and dimensions for the previous cell spec.
     * @param {string} color The color to render all shapes in.
     */
    constructor(color, id) {
        super(color, "gMid", id);
    }

    /**
     * Select the each of the border elements with D3 if not already done,
     * attach a transition and resize each shape individually.
     * @param {Object} dims The cell spec to use while resizing/repositioning.
     * @param {selection} d3Group The pre-selected group to work with.
     * @param {Array} border Border selections created in _renderBorder().
     */
    _updateBorder(dims, d3Group, border) {
        if (!border) {
            // Find the existing border rects if not provided.
            border = [
                d3Group.select(".bordR1").transition(sharedTransition),
                d3Group.select(".bordR2").transition(sharedTransition),
                d3Group.select(".bordR3").transition(sharedTransition),
                d3Group.select(".bordR4").transition(sharedTransition)
            ];
        }

        border[0]
            .attr("x", dims.topLeft.x).attr("y", dims.topLeft.y)
            .attr("width", dims.size.width).attr("height", dims.size.height * .1);

        border[1]
            .attr("x", dims.topLeft.x).attr("y", dims.topLeft.y)
            .attr("width", dims.size.width * .1).attr("height", dims.size.height);

        border[2]
            .attr("x", dims.bottomRight.x * .8).attr("y", dims.topLeft.y)
            .attr("width", dims.size.width * .1).attr("height", dims.size.height);

        border[3]
            .attr("x", dims.topLeft.x).attr("y", dims.bottomRight.y * .8)
            .attr("width", dims.size.width).attr("height", dims.size.height * .1);
    }

    /**
     * Draw four skinny rectangles at the perimeter of the group.
     * @param {Object} d3Group D3 selection for the group that we add to.
     */
    _renderBorder(d3Group) {
        let border = [
            d3Group.append("rect").attr("class", "bordR1").style("fill", this.color),
            d3Group.append("rect").attr("class", "bordR2").style("fill", this.color),
            d3Group.append("rect").attr("class", "bordR3").style("fill", this.color),
            d3Group.append("rect").attr("class", "bordR4").style("fill", this.color)
        ];

        return border;
    }

    /**
     * Select the element with D3 if not already done, attach a transition
     * and resize the shape.
     * @param svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while resizing/repositioning.
     * @param {selection} [d3Elem = null] The selection created in render().
     * @param {Array} [border = null] Border selections created in _renderBorder().
     */
    update(svgGroup, dims, d3Elem = null, border = null) {
        let d3Group = d3.select(svgGroup);
        if (!d3Elem) d3Elem = d3Group.select("." + this.className)
            .transition(sharedTransition);

        this._updateBorder(dims, d3Group, border);

        return d3Elem
            .attr("x", dims.topLeft.x * .6)
            .attr("y", dims.topLeft.y * .6)
            .attr("width", dims.size.width * .6)
            .attr("height", dims.size.height * .6);
    }

    /** 
     * Get the D3 selection for the appropriate group and append a filled rectangle,
     * then call _renderBorder() to put a border around it.
     * @param {Object} svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while rendering.
     */
    render(svgGroup, dims) {
        let d3Group = d3.select(svgGroup);
        let border = this._renderBorder(d3Group);
        let d3Elem = d3Group
            .append("rect")
            .attr("class", this.className)
            .attr("id", this.id)
            .style("fill", this.color);

        return this.update(svgGroup, dims, d3Elem, border);
    }
}

class N2ScalarCell extends N2ScalarBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2VectorCell extends N2VectorBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2GroupCell extends N2GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2ScalarScalarCell extends N2ScalarBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2ScalarVectorCell extends N2VectorBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2VectorScalarCell extends N2VectorBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2VectorVectorCell extends N2VectorBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2ConnectorUpper extends N2Connector {
    constructor(color, id) {
        super(color, id);
    }

    /** Generate a string to use for the transform attribute */
    _transform(scale) { return('scale(' + scale + ')'); }
}

class N2ConnectorLower extends N2Connector {
    constructor(color, id) {
        super(color, id);
    }

    /** Generate a string to use for the transform attribute */
    _transform(scale) { return('scale(' + scale + ') rotate(180)'); }
}

class N2ScalarGroupCell extends N2GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2GroupScalarCell extends N2GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2VectorGroupCell extends N2GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2GroupVectorCell extends N2GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}

class N2GroupGroupCell extends N2GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}

// TODO: Consider moving color management into CSS styles

/**
 * A visible cell in the N2 matrix.
 * @typedef {Object} N2MatrixCell
 * @property {number} row Vertical coordinate of the cell in the matrix.
 * @property {number} col Horizontal coordinate of the cell in the matrix.
 * @property {N2TreeNode} srcObj The node in the model tree this cell is associated with.
 * @property {N2TreeNode} tgtObj The model tree node that this outputs to.
 * @property {string} id The srcObj id appended with the tgtObj id.
 * @property {SymbolType} symbolType Info about the type of symbol represented by the node.
 */
class N2MatrixCell {
    /**
     * Initialize the cell.
     * @param {number} row Vertical coordinate of the cell in the matrix.
     * @param {number} col Horizontal coordinate of the cell in the matrix.
     * @param {N2TreeNode} srcObj The node in the model tree this node is associated with.
     * @param {N2TreeNode} tgtObj The model tree node that this outputs to.
     * @param {ModelData} model Reference to the model to get some info from it.
     * @param {N2CellRenderer} renderer The object that draws the cell.
     */
    constructor(row, col, srcObj, tgtObj, model) {
        this.row = row;
        this.col = col;
        this.srcObj = this.obj = srcObj;
        this.tgtObj = tgtObj;
        this.id = N2MatrixCell.makeId(srcObj.id, tgtObj.id);

        this.symbolType = new SymbolType(this, model);
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
    mouseover() {
        return (this.onDiagonal() ? n2MouseFuncs.overOnDiag :
            n2MouseFuncs.overOffDiag);
    }

    /**
    * Select the mousemove callback depending on whether we're on the diagonal.
    * TODO: Remove these globals
    */
    mousemove() {
        return (this.onDiagonal() ? n2MouseFuncs.moveOnDiag : null);
    }

    /**
     * Choose a color based on our location and state of the associated N2TreeNode.
     */
    color() {
        if (this.symbolType.potentialDeclaredPartial &&
            this.symbolType.declaredPartial) return N2Style.color.declaredPartial;

        if (this.onDiagonal()) {
            if (this.obj.isMinimized) return N2Style.color.collapsed;
            if (this.obj.isAutoIvcInput()) return N2Style.color.autoivcInput;
            if (this.obj.isConnectedInput()) return N2Style.color.input;
            if (this.obj.isUnconnectedInput()) return N2Style.color.unconnectedInput;
            return (this.obj.implicit) ?
                N2Style.color.outputImplicit :
                N2Style.color.outputExplicit;
        }

        return N2Style.color.connection;
    }


    /**
     * A connection going "off-screen" was detected between two nodes.
     * Determine whether the arrow should be in the top or bottom section of the
     * matrix based on rootIndex, and add to the appropriate array of
     * tracked offscreen connections.
     * @param {N2TreeNode} srcNode Where the connection starts.
     * @param {N2TreeNode} tgtNode Where the connection ends.
     */
    addOffScreenConn(srcNode, tgtNode) {
        let debugStr = ": " + srcNode.absPathName + " -> " + tgtNode.absPathName;

        if (srcNode === this.tgtObj) {
            // Outgoing
            if (srcNode.rootIndex < tgtNode.rootIndex) {
                // Top
                debugInfo("New offscreen outgoing connection on top" + debugStr);
                this.offScreen.top.outgoing.add(tgtNode);
            }
            else {
                // Bottom
                debugInfo("New offscreen outgoing connection on bottom" + debugStr);
                this.offScreen.bottom.outgoing.add(tgtNode);
            }
        }
        else {
            // Incoming
            if (srcNode.rootIndex < tgtNode.rootIndex) {
                // Top
                debugInfo("New offscreen incoming connection on top" + debugStr);
                this.offScreen.top.incoming.add(srcNode);
            }
            else {
                // Bottom
                debugInfo("New offscreen incoming connection on bottom" + debugStr);
                this.offScreen.bottom.incoming.add(srcNode);
            }
        }

        this.offScreen.total++;
        // debugInfo("Total offscreen connections found: " + this.offScreen.total);
    }

    /** Choose a renderer based on our SymbolType. */
    _newRenderer() {
        if (this.color() == N2Style.color.connection) {
            if (this.inUpperTriangle()) return new N2ConnectorUpper(this.color(), this.id);

            return new N2ConnectorLower(this.color(), this.id)
        }

        const color = this.color();

        switch (this.symbolType.name) {
            case "scalar":
                return new N2ScalarCell(color, this.id);
            case "vector":
                return new N2VectorCell(color, this.id);
            case "group":
                return new N2GroupCell(color, this.id);
            case "scalarScalar":
                return new N2ScalarScalarCell(color, this.id);
            case "scalarVector":
                return new N2ScalarVectorCell(color, this.id);
            case "vectorScalar":
                return new N2VectorScalarCell(color, this.id);
            case "vectorVector":
                return new N2VectorVectorCell(color, this.id);
            case "scalarGroup":
                return new N2ScalarGroupCell(color, this.id);
            case "groupScalar":
                return new N2GroupScalarCell(color, this.id);
            case "vectorGroup":
                return new N2VectorGroupCell(color, this.id);
            case "groupVector":
                return new N2GroupVectorCell(color, this.id);
            case "groupGroup":
                return new N2GroupGroupCell(color, this.id);
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

        const obj = (varType == 'target') ? this.tgtObj : this.srcObj;
        const treeId = obj.absPathName.replace(/[\.:]/g, '_');
        const treeNode = d3.select('rect#' + treeId);

        let fill = treeNode.style('fill');
        if (direction == 'input') fill = N2Style.color.inputArrow;
        else if (direction == 'output') fill = N2Style.color.outputArrow;

        d3.select('#highlight-bar').append('rect')
            .attr('x', 0)
            .attr('y', treeNode.node().parentNode.transform.baseVal[0].matrix.f)
            .attr('rx', 4)
            .attr('ry', 4)
            .attr('width', 8)
            .attr('height', treeNode.attr('height'))
            .attr('stroke', N2Style.color.treeStroke)
            .attr('fill', fill);
    }
}
