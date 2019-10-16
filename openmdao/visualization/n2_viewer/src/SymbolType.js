/**
 * @typedef {Object} SymbolType
 * @property {string} name What the symbol is called.
 * @property {Boolean} declaredPartial Whether the symbol is a declared partial.
 */
class SymbolType {

    /**
     * Determine the name and whether it's a declared partial based on info
     * from the provided node.
     * @param {N2MatrixNode} node The object to select the type from.
     * @param {ModelData} model Reference to the model to get some info from it.
     */
    constructor(node, model) {
        this.name = null;
        this.declaredPartial = false;

        // Update properties based on the the referenced node.
        this.getType(node, model);
    }

    /** 
     * Decide what object the node will be drawn as, based on its position
     * in the matrix, type, source, target, and/or other conditions.
    */
    getType(node, model) {
        if (node.onDiagonal()) {
            if (node.srcObj.type == "subsystem") {
                this.name = 'group';
                return;
            }

            if (node.srcObj.type.match(paramOrUnknownRegex)) {
                if (node.srcObj.dtype == "ndarray") {
                    this.name = 'vector';
                    return;
                }

                this.name = 'scalar';
                return;
            }

            throw ("Unknown symbol type for node on diagonal.");
        }

        if (node.srcObj.type == "subsystem") {
            if (node.tgtObj.type == "subsystem") {
                this.name = 'groupGroup';
                return;
            }

            if (node.tgtObj.type.match(paramOrUnknownRegex)) {
                if (node.tgtObj.dtype == "ndarray") {
                    this.name = 'groupVector';
                    return;
                }

                this.name = 'groupScalar';
                return;
            }

            throw ("Unknown group symbol type.");
        }

        if (node.srcObj.type.match(paramOrUnknownRegex)) {

            if (node.srcObj.dtype == "ndarray") {
                if (node.tgtObj.type.match(paramOrUnknownRegex)) {
                    if (node.tgtObj.dtype == "ndarray" ||
                        node.tgtObj.type.match(paramRegex)) {

                        this.name = 'vectorVector';
                        this.declaredPartial = model.isDeclaredPartial(node.srcObj, node.tgtObj);

                        return;
                    }

                    this.name = 'vectorScalar';
                    this.declaredPartial = model.isDeclaredPartial(node.srcObj, node.tgtObj);
                    return;
                }

                if (node.tgtObj.type === "subsystem") {
                    this.name = 'vectorGroup';
                    return;
                }

                throw ("Unknown vector symbol type.");
            }

            if (node.tgtObj.type.match(paramOrUnknownRegex)) {
                if (node.tgtObj.dtype == "ndarray") {
                    this.name = 'scalarVector';
                    this.declaredPartial = model.isDeclaredPartial(node.srcObj, node.tgtObj);

                    return;
                }

                this.name = 'scalarScalar';
                this.declaredPartial = model.isDeclaredPartial(node.srcObj, node.tgtObj);

                return;
            }

            if (node.tgtObj.type == "subsystem") {
                this.name = 'scalarGroup';
                return;
            }

            throw ("Unknown vector or scalar symbol type.");
        }

        throw ("Completely unrecognized symbol type.")
    }
}