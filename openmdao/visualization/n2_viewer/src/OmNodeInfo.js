// <<hpp_insert gen/NodeInfo.js>>
// <<hpp_insert src/InfoPropOptions.js>>

/**
 * Manage a table containing all available metadata properties for
 * the currently active node, as well as whether the table is
 * visible or not.
 * @typedef OmNodeInfo
 */
class OmNodeInfo extends NodeInfo {
    /**
     * Build a list of the properties we care about and set up
     * references to the HTML elements.
     */
    constructor(ui) {
        super(ui);

        // Add potential properties to super class listing
        const newPropList = [
            this.propList[0], // path
            new InfoPropDefault('promotedName', 'Promoted Name'),
            this.propList[1], // class
            new InfoPropDefault('surrogate_name', 'Surrogate'),
            this.propList[2], // type
            new InfoPropDefault('dtype', 'DType'),

            new InfoPropDefault('units', 'Units'),
            new InfoPropDefault('shape', 'Shape'),
            new InfoPropYesNo('is_discrete', 'Discrete'),
            new InfoPropMessage('initial_value', '** Note **',
                                'Non-local values are not available under MPI, showing initial value.'),
            new InfoPropYesNo('distributed', 'Distributed'),
            this.propList[3], // val
            new InfoPropNumber('val_min', 'Minimum'),
            new InfoPropDefault('val_min_indices', 'Minimum Indices'),
            new InfoPropNumber('val_max', 'Maximum'),
            new InfoPropDefault('val_max_indices', 'Maximum Indices'),
            new InfoPropDefault('subsystem_type', 'Subsystem Type', true),
            new InfoPropDefault('component_type', 'Component Type', true),
            new InfoPropYesNo('implicit', 'Implicit'),
            new InfoPropYesNo('is_parallel', 'Parallel'),
            new InfoPropDefault('linear_solver', 'Linear Solver'),
            new InfoPropDefault('nonlinear_solver', 'Non-Linear Solver'),
            new InfoPropExpr('expressions', 'Expressions'),

            new InfoPropOptions('options', 'Options'),
            new InfoPropOptions('linear_solver_options', 'Linear Solver Options', 'linear'),
            new InfoPropOptions('nonlinear_solver_options', 'Non-Linear Solver Options', 'nonlinear'),
        ];

        this.propList = newPropList;
    }
}

/**
 * Manage a table containing all available metadata properties for
 * the currently active node, as well as whether the table is
 * visible or not.
 * @typedef OmSolverNodeInfo
 */
 class OmSolverNodeInfo extends NodeInfo {
    /**
     * Build a list of the properties we care about and set up
     * references to the HTML elements.
     */
    constructor(ui) {
        super(ui, false);

        // Potential solver properties
        this.propList = [
            new InfoPropDefault('path', 'Absolute Name'),
            new InfoPropOptions('linear_solver_options', 'Linear Solver Options', 'linear'),
            new InfoPropOptions('nonlinear_solver_options', 'Non-Linear Solver Options', 'nonlinear'),
        ];
    }
}
