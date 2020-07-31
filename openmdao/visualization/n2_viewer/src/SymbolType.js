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

        // Indicates that the type of symbol CAN be a declared partial
        // AND both source and target objects are part of the same component.
        this.potentialDeclaredPartial = false;

        this.declaredPartial = false;

        // Update properties based on the the referenced node.
        this.getType(cell, model);
    }

    /**
     * For symbols types that CAN be a declared partial, check whether they're
     * part of the same component and that they're in the declared partial list.
     * @param {N2TreeNode} cell The cell to operate on.
     * @param {ModelData} model Reference to the entire model.
     */
    _setDeclaredPartialInfo(cell, model) {
        if (cell.tgtObj.parentComponent === cell.srcObj.parentComponent) {
            this.potentialDeclaredPartial = true;
            this.declaredPartial = model.isDeclaredPartial(cell.srcObj, cell.tgtObj);
        }
    }

    /** 
     * Decide what object the cell will be drawn as, based on its position
     * in the matrix, type, source, target, and/or other conditions.
     * @param {N2MatrixCell} cell The cell to operate on.
     * @param {ModelData} model Reference to the entire model.
     */
    getType(cell, model) {
        if (cell.onDiagonal()) {
            if (cell.srcObj.isSubsystem()) this.name = 'group';
            else if (cell.srcObj.isInputOrOutput()) {
                if (cell.srcObj.dtype == "ndarray") this.name = 'vector';
                else this.name = 'scalar';
            }
            else {
                throw ("Output symbol type '" + cell.srcObj.type + "' for cell on diagonal.");
            }
        }
        else if (cell.srcObj.isSubsystem()) {
            if (cell.tgtObj.isSubsystem()) this.name = 'groupGroup';
            else if (cell.tgtObj.isInputOrOutput()) {
                if (cell.tgtObj.dtype == "ndarray") this.name = 'groupVector';
                else this.name = 'groupScalar';
            }
            else throw ("Output group symbol type.");
        }
        else if (cell.srcObj.isInputOrOutput()) {
            if (cell.srcObj.dtype == "ndarray") {
                if (cell.tgtObj.isInputOrOutput()) {
                    if (cell.tgtObj.dtype == "ndarray" || cell.tgtObj.isInput()) {
                        this.name = 'vectorVector';
                        this._setDeclaredPartialInfo(cell, model);
                    }
                    else {
                        this.name = 'vectorScalar';
                        this._setDeclaredPartialInfo(cell, model);
                    }
                }

                else if (cell.tgtObj.isSubsystem()) this.name = 'vectorGroup';
                else throw ("Output vector symbol type.");
            }

            else if (cell.tgtObj.isInputOrOutput()) {
                if (cell.tgtObj.dtype == "ndarray") {
                    this.name = 'scalarVector';
                    this._setDeclaredPartialInfo(cell, model);
                }
                else {
                    this.name = 'scalarScalar';
                    this._setDeclaredPartialInfo(cell, model);
                }
            }

            else if (cell.tgtObj.isSubsystem()) this.name = 'scalarGroup';

            else throw ("Output vector or scalar symbol type.");
        }
        else throw ("Completely unrecognized symbol type.")
    }
}
