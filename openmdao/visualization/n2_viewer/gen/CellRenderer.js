// <<hpp_insert gen/SymbolType.js>>

/** Base class for all cell renderers */
class CellRenderer {
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

    static updateDims(baseWidth, baseHeight, pcntSize = 0.6) {
        if (!CellRenderer.dims) {
            CellRenderer.prevDims = {
                "size": {
                    "width": 0,
                    "height": 0,
                    "percent": pcntSize
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
            for (const prop of ["size", "bottomRight", "topLeft"]) {
                Object.assign(CellRenderer.prevDims[prop],
                    CellRenderer.dims[prop]);
            }
        }

        CellRenderer.dims = {
            "size": {
                "width": baseWidth,
                "height": baseHeight,
                "percent": pcntSize
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
        return CellRenderer.dims;
    }

    /** Enable access to the static prevDims variable through "this". */
    get prevDims() {
        return CellRenderer.prevDims;
    }

    /** Act like an abstract base class force derived classes to define. */
    update() {
        throw ("ERROR: CellRenderer.update() called.")
    }

    /** Act like an abstract base class force derived classes to define. */
    render() {
        throw ("ERROR: CellRenderer.render() called.")
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
class ScalarBase extends CellRenderer {

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
            .attr("rx", dims.bottomRight.x * dims.size.percent)
            .attr("ry", dims.bottomRight.y * dims.size.percent);
    }

    /** 
     * Get the D3 selection for the appropriate group and append a filled ellipse.
     * @param {Object} svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while rendering.
     */
    render(svgGroup, dims) {
        const d3Elem = d3.select(svgGroup)
            .append("ellipse")
            .attr("class", this.className)
            .attr("id", this.id)
            .style("fill", this.color);

        return this.update(svgGroup, dims, d3Elem)
    }
}

/** Draws/updates an SVG rect for vector types, with a transition animation. */
class VectorBase extends CellRenderer {

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

        const ret = d3Elem
            .attr("x", dims.topLeft.x * dims.size.percent)
            .attr("y", dims.topLeft.y * dims.size.percent)
            .attr("width", dims.bottomRight.x * dims.size.percent * 2)
            .attr("height", dims.bottomRight.y * dims.size.percent * 2);


        return ret;
    }

    /** 
     * Get the D3 selection for the appropriate group and append a filled rectangle.
     * @param {Object} svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while rendering.
     */
    render(svgGroup, dims) {
        const d3Elem = d3.select(svgGroup).append('rect')
            .attr("class", this.className)
            .attr("id", this.id)
            .style("fill", this.color);

        return this.update(svgGroup, dims, d3Elem);
    }
}

class Connector extends CellRenderer {
    /**
     * Invoke the superclass constructor with these values and "vMid" as a CSS class.
     * @param {Object} dims Layout and dimensions for the current cell spec.
     * @param {Object} prevDims Layout and dimensions for the previous cell spec.
     * @param {string} color The color to render all shapes in.
     */
    constructor(color, id) {
        super(color, "vMid", id);
    }

    _transform(scale) { throw ("ERROR: Connector._transform() called.") }

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

        const ret = d3Elem.attr("transform", this._transform(dims.size.width/10.0));

        return ret;
    }

    /** 
     * Get the D3 selection for the appropriate group and append a filled arrow.
     * @param {Object} svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while rendering.
     */
    render(svgGroup, dims) {
        const d3Elem = d3.select(svgGroup).append('use')
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
class GroupBase extends CellRenderer {
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
        const border = [
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
        const d3Group = d3.select(svgGroup);
        if (!d3Elem) d3Elem = d3Group.select("." + this.className)
            .transition(sharedTransition);

        this._updateBorder(dims, d3Group, border);

        return d3Elem
            .attr("x", dims.topLeft.x * dims.size.percent)
            .attr("y", dims.topLeft.y * dims.size.percent)
            .attr("width", dims.size.width * dims.size.percent)
            .attr("height", dims.size.height * dims.size.percent);
    }

    /** 
     * Get the D3 selection for the appropriate group and append a filled rectangle,
     * then call _renderBorder() to put a border around it.
     * @param {Object} svgGroup Reference to SVG <g> element associated with data.
     * @param {Object} dims The cell spec to use while rendering.
     */
    render(svgGroup, dims) {
        const d3Group = d3.select(svgGroup);
        const border = this._renderBorder(d3Group);
        const d3Elem = d3Group
            .append("rect")
            .attr("class", this.className)
            .attr("id", this.id)
            .style("fill", this.color);

        return this.update(svgGroup, dims, d3Elem, border);
    }
}

class ScalarCell extends ScalarBase {
    constructor(color, id) {
        super(color, id);
    }
}

class VectorCell extends VectorBase {
    constructor(color, id) {
        super(color, id);
    }
}

class GroupCell extends GroupBase {
    constructor(color, id) {
        super(color, id);
    }
}

class ConnectorUpper extends Connector {
    constructor(color, id) {
        super(color, id);
    }

    /** Generate a string to use for the transform attribute */
    _transform(scale) { return('scale(' + scale + ')'); }
}

class ConnectorLower extends Connector {
    constructor(color, id) {
        super(color, id);
    }

    /** Generate a string to use for the transform attribute */
    _transform(scale) { return('scale(' + scale + ') rotate(180)'); }
}

class FilterCell extends VectorBase {
    constructor(color, id) {
        super(color, id);
    }
}
