/**
 * Use the model tree to build a matrix, display it, and perform
 * operations with it.
 */
class N2Matrix {
    constructor(nodes) {
        n2Dx0 = n2Dx;
        n2Dy0 = n2Dy;

        n2Dx = WIDTH_N2_PX / nodes.length;
        n2Dy = HEIGHT_PX / nodes.length;

        this.buildStructure(nodes);
        this.populateSymbols(nodes);
        this.drawingPrep(nodes);
    }

    /**
     * Determine if a node exists at the specified location.
     * @param {number} row Row number of the node
     * @param {number} col Column number of the node
     * @returns False if the row doesn't exist or a column doesn't exist
     *  in the row; true otherwise.
     */
    exists(row, col) {
        if (this.matrix.hasOwnProperty(row) &&
            this.matrix[row].hasOwnProperty(col) &&
            typeof this.matrix[row][col] !== 'undefined' ) { return true; }
        return false;
    }

    /**
     * Safe method to access a node at the specified location.
     * @param {number} row Row number of the node
     * @param {number} col Column number of the node
     * @param {boolean} doThrow Whether to throw an exception if node undefined.
     * @returns The node if it exists, undefined otherwise.
     */
    node(row, col, doThrow = false) {
        if (this.exists(row, col)) {
            return this.matrix[row][col];
        }
        else if (doThrow) {
            throw "No node in matrix at (" + row + ", " + col + ").";
        }

        return undefined;
    }

    /**
     * Set up nested objects resembling a two-dimensional array as the
     * matrix, but not an actual two dimensional array because most of
     * it would be unused.
     * @param {Object[]} nodes 
     */
    buildStructure(nodes) {
        this.matrix = {};

        for (let srcIdx = 0; srcIdx < nodes.length; ++srcIdx) {
            let srcObj = nodes[srcIdx];

            // These nodes are on the diagonal.
            if (!this.exists(srcIdx)) { this.matrix[srcIdx] = {}; }
            this.matrix[srcIdx][srcIdx] = {
                "row": srcIdx,
                "col": srcIdx,
                "obj": srcObj,
                "id": srcObj.id + "_" + srcObj.id
            };

            let targets = srcObj.targetsParamView;

            for (let tgtObj of targets) {
                let tgtIdx = nodes.indexOf(tgtObj);
                if (tgtIdx != -1) {
                    this.matrix[srcIdx][tgtIdx] = {
                        "row": srcIdx,
                        "col": tgtIdx,
                        "obj": srcObj,
                        "id": srcObj.id + "_" + tgtObj.id
                    };
                }
            }

            if (srcObj.type === "param" || srcObj.type === "unconnected_param") {
                for (let j = srcIdx + 1; j < nodes.length; ++j) {
                    let tgtObj = nodes[j];
                    if (srcObj.parentComponent !== tgtObj.parentComponent) break;

                    if (tgtObj.type === "unknown") {
                        let tgtIdx = j;
                        this.matrix[srcIdx][tgtIdx] = {
                            "row": srcIdx,
                            "col": tgtIdx,
                            "obj": srcObj,
                            "id": srcObj.id + "_" + tgtObj.id
                        };
                    }
                }
            }
        }
    }

    populateSymbols(nodes) {
        this.symbols = {
            'scalar': [],
            'vector': [],
            'group': [],
            'scalarScalar': [],
            'scalarVector': [],
            'vectorScalar': [],
            'vectorVector': [],
            'scalarGroup': [],
            'groupScalar': [],
            'vectorGroup': [],
            'groupVector': [],
            'groupGroup': [],
            'declaredPartials': {
                'vectorVector': [],
                'scalarScalar': [],
                'vectorScalar': [],
                'scalarVector': []
            }
        };

        let regex = /unknown|param|unconnected_param/;

        for (let row in this.matrix) {
            for (let col in this.matrix[row]) {
                let d = this.matrix[row][col];
                let tgtObj = nodes[d.col],
                    srcObj = nodes[d.row];

                if (d.col == d.row) { //on diagonal
                    if (srcObj.type === "subsystem") { // group
                        this.symbols.group.push(d);
                    } else if (srcObj.type.match(regex)) {
                        if (srcObj.dtype === "ndarray") { // vector
                            this.symbols.vector.push(d);
                        } else { // scalar
                            this.symbols.scalar.push(d);
                        }
                    }

                }
                else if (srcObj.type === "subsystem") {
                    if (tgtObj.type === "subsystem") { // groupGroup
                        this.symbols.groupGroup.push(d);
                    }
                    else if (tgtObj.type.match(regex)) {
                        if (tgtObj.dtype === "ndarray") { // groupVector
                            this.symbols.groupVector.push(d);
                        }
                        else { // groupScalar
                            this.symbols.groupScalar.push(d);
                        }
                    }
                }
                else if (srcObj.type.match(regex)) {
                    if (srcObj.dtype === "ndarray") {
                        if (tgtObj.type.match(regex)) {
                            if (tgtObj.dtype === "ndarray" ||
                                tgtObj.type.match(/param|unconnected_param/)) { // vectorVector
                                this.symbols.vectorVector.push(d);

                                let partials_string = tgtObj.absPathName + " > " + srcObj.absPathName;
                                if (modelData.declare_partials_list.includes(partials_string)) {
                                    this.symbols.declaredPartials.vectorVector.push(d);
                                }

                            }
                            else { // vectorScalar
                                this.symbols.vectorScalar.push(d);
                                let partials_string = tgtObj.absPathName + " > " + srcObj.absPathName;
                                if (modelData.declare_partials_list.includes(partials_string)) {
                                    this.symbols.declaredPartials.vectorScalar.push(d);
                                }
                            }
                        }
                        else if (tgtObj.type === "subsystem") { // vectorGroup
                            this.symbols.vectorGroup.push(d);
                        }
                    }
                    else {
                        if (tgtObj.type.match(regex)) {
                            if (tgtObj.dtype === "ndarray") { // scalarVector
                                this.symbols.scalarVector.push(d);
                                let partials_string = tgtObj.absPathName + " > " + srcObj.absPathName;
                                if (modelData.declare_partials_list.includes(partials_string)) {
                                    this.symbols.declaredPartials.scalarVector.push(d);
                                }
                            }
                            else { // scalarScalar
                                this.symbols.scalarScalar.push(d);
                                let partials_string = tgtObj.absPathName + " > " + srcObj.absPathName;
                                if (modelData.declare_partials_list.includes(partials_string)) {
                                    this.symbols.declaredPartials.scalarScalar.push(d);
                                }
                            }
                        }
                        else if (tgtObj.type === "subsystem") { // scalarGroup
                            this.symbols.scalarGroup.push(d);
                        }
                    }
                }
            }
        }
    }

    drawingPrep(nodes) {
        let currentBox = { "startI": 0, "stopI": 0 };

        d3RightTextNodesArrayZoomedBoxInfo = [currentBox];

        for (let ri = 1; ri < nodes.length; ++ri) {
            //boxes
            let el = nodes[ri];
            let startINode = nodes[currentBox.startI];
            if (startINode.parentComponent &&
                el.parentComponent &&
                startINode.parentComponent === el.parentComponent) {
                ++currentBox.stopI;
            }
            else {
                currentBox = { "startI": ri, "stopI": ri };
            }
            d3RightTextNodesArrayZoomedBoxInfo.push(currentBox);
        }

        drawableN2ComponentBoxes = [];
        // draw grid lines last so that they will always be visible
        for (let i = 0; i < d3RightTextNodesArrayZoomedBoxInfo.length; ++i) {
            let box = d3RightTextNodesArrayZoomedBoxInfo[i];
            if (box.startI == box.stopI) continue;
            let el = nodes[box.startI];
            if (!el.parentComponent) {
                throw "Parent component not found in box.";
            }
            box.obj = el.parentComponent;
            i = box.stopI;
            drawableN2ComponentBoxes.push(box);
        }

        //do this so you save old index for the exit()
        gridLines = [];
        if (nodes.length < LEVEL_OF_DETAIL_THRESHOLD) {
            for (let i = 0; i < nodes.length; ++i) {
                let obj = nodes[i];
                var gl = { "i": i, "obj": obj };
                gridLines.push(gl);
            }
        }
    }
}