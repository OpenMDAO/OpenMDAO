/**
 * @typedef {Object} SymbolType
 * @property {string} name What the symbol is called.
 * @property {Boolean} declaredPartial Whether the symbol is a declared partial.
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
        this.potentialDeclaredPartial = false;
        this.declaredPartial = false;

        // Update properties based on the the referenced node.
        this.getType(cell, model);
    }

    /** 
     * Decide what object the cell will be drawn as, based on its position
     * in the matrix, type, source, target, and/or other conditions.
    */
    getType(cell, model) {
        if (cell.onDiagonal()) {
            if (cell.srcObj.isSubsystem()) {
                this.name = 'group';
                return;
            }

            if (cell.srcObj.isParamOrUnknown()) {
                if (cell.srcObj.dtype == "ndarray") {
                    this.name = 'vector';
                    return;
                }

                this.name = 'scalar';
                return;
            }

            throw ("Unknown symbol type for cell on diagonal.");
        }

        if (cell.srcObj.isSubsystem()) {
            if (cell.tgtObj.isSubsystem()) {
                this.name = 'groupGroup';
                return;
            }

            if (cell.tgtObj.isParamOrUnknown()) {
                if (cell.tgtObj.dtype == "ndarray") {
                    this.name = 'groupVector';
                    return;
                }

                this.name = 'groupScalar';
                return;
            }

            throw ("Unknown group symbol type.");
        }

        if (cell.srcObj.isParamOrUnknown()) {
            if (cell.srcObj.dtype == "ndarray") {
                if (cell.tgtObj.isParamOrUnknown()) {
                    if (cell.tgtObj.dtype == "ndarray" || cell.tgtObj.isParam()) {
                        this.name = 'vectorVector';
                        this.potentialDeclaredPartial = true;
                        this.declaredPartial =
                            model.isDeclaredPartial(cell.srcObj, cell.tgtObj);

                        return;
                    }

                    this.name = 'vectorScalar';
                    this.potentialDeclaredPartial = true;
                    this.declaredPartial =
                        model.isDeclaredPartial(cell.srcObj, cell.tgtObj);
                    return;
                }

                if (cell.tgtObj.isSubsystem()) {
                    this.name = 'vectorGroup';
                    return;
                }

                throw ("Unknown vector symbol type.");
            }

            if (cell.tgtObj.isParamOrUnknown()) {
                if (cell.tgtObj.dtype == "ndarray") {
                    this.name = 'scalarVector';
                    this.potentialDeclaredPartial = true;
                    this.declaredPartial =
                        model.isDeclaredPartial(cell.srcObj, cell.tgtObj);

                    return;
                }

                this.name = 'scalarScalar';
                this.potentialDeclaredPartial = true;
                this.declaredPartial =
                    model.isDeclaredPartial(cell.srcObj, cell.tgtObj);

                return;
            }

            if (cell.tgtObj.isSubsystem()) {
                this.name = 'scalarGroup';
                return;
            }

            throw ("Unknown vector or scalar symbol type.");
        }

        throw ("Completely unrecognized symbol type.")
    }
}