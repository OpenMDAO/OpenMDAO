// <<hpp_insert gen/Legend.js>>

/**
 * Draw a symbol describing each of the element types.
 * @typedef OmLegend
 * @property {Boolean} shown Whether the legend is currently drawn or not.
 */
class OmLegend extends Legend {
    /**
     * Initializes the legend object.
     * @param {ModelData} modelData Symbols are only displayed if they're in the model
     */
    constructor(modelData) {
        super(modelData);
    }

    /**
     * Set the types to check for in the model to determine if they
     * need to be displayed in the legend.
     */
    _initItemTypes() {
        this.showSysVar = {
            'group': false,
            'component': false,
            'input': false,
            'desvars': false,
            'unconnectedInput': false,
            'autoivcInput': false,
            'outputExplicit': false,
            'outputImplicit': false,
            'collapsed': true,
            'connection': true,
            'declaredPartial': true
        };

        this.showSymbols = {
            'scalar': false,
            'vector': false,
            'collapsedVariables': false
        };

        this.sysAndVar = [
            { 'name': "Connection", 'color': OmStyle.color.connection },
            { 'name': "Collapsed", 'color': OmStyle.color.collapsed },
            { 'name': "Declared Partial", 'color': OmStyle.color.declaredPartial }
        ];
                
        const rootLinearSolver =
            OmStyle.solverStyleObject.find(x => x.ln === modelData.tree.linear_solver);
        const rootNonLinearSolver =
            OmStyle.solverStyleObject.find(x => x.nl === modelData.tree.nonlinear_solver);
        this.linearSolvers = [
            { 'name': modelData.tree.linear_solver, 'color': rootLinearSolver.color }
        ];

        this.nonLinearSolvers = [
            { 'name': modelData.tree.nonlinear_solver, 'color': rootNonLinearSolver.color }
        ];
    }

    /**
     * Determine which types of nodes are present in the model and only mark
     * those types of keys to be displayed in the legend.
     * @param {Object} nodes The model node tree.
     */
    _setDisplayBooleans(nodes) {
        for (const node of nodes) {
            const {
                group,
                component,
                desvar,
                input,
                unconnectedInput,
                outputExplicit,
                outputImplicit,
                collapsed,
                connection,
                declaredPartial
            } = this.showSysVar;

            const linearSolver = node.linear_solver;
            const nonLinearSolver = node.nonlinear_solver;

            const linearSolverIndex =
                this.linearSolvers.indexOf(this.linearSolvers.find(x => x.name === linearSolver))
            const nonLinearSolverIndex =
                this.nonLinearSolvers.indexOf(this.nonLinearSolvers.find(x => x.name === nonLinearSolver));

            if (linearSolverIndex < 0 && linearSolver !== undefined) {
                let solverStyle = OmStyle.solverStyleObject.find(x => x.ln === linearSolver);
                this.linearSolvers.push({
                    'name': solverStyle.ln,
                    'color': solverStyle.color
                });
            }

            if (nonLinearSolverIndex < 0 && nonLinearSolver !== undefined) {
                let solverStyle = OmStyle.solverStyleObject.find(x => x.nl === nonLinearSolver);
                this.nonLinearSolvers.push({
                    'name': solverStyle.nl,
                    'color': solverStyle.color
                });
            }

            if (node.hasChildren()) {
                if (!this.showSysVar.group && node.isGroup()) {
                    this.showSysVar.group = true;
                    this.sysAndVar.push({
                        'name': 'Group',
                        'color': OmStyle.color.group
                    })
                }
                else if (!this.showSysVar.component && node instanceof OmTreeNode &&
                    node.isComponent()) {
                    this.showSysVar.component = true;
                    this.sysAndVar.push({
                        'name': 'Component',
                        'color': OmStyle.color.component
                    })
                }
                this._setDisplayBooleans(node.children);
            }
            else {
                if (!this.showSysVar.input && node.isInput()) {
                    this.showSysVar.input = true;
                    this.sysAndVar.push({
                        'name': 'Input',
                        'color': OmStyle.color.input
                    })
                }
                else if (!this.showSysVar.outputExplicit && node instanceof OmTreeNode &&
                    node.isExplicitOutput()) {
                    this.showSysVar.outputExplicit = true;
                    this.sysAndVar.push({
                        'name': 'Explicit Output',
                        'color': OmStyle.color.outputExplicit
                    })
                }
                else if (!this.showSysVar.outputImplicit && node instanceof OmTreeNode &&
                    node.isImplicitOutput()) {
                    this.showSysVar.outputImplicit = true;
                    this.sysAndVar.push({
                        'name': 'Implicit Output',
                        'color': OmStyle.color.outputImplicit
                    })
                }
                else if (!this.showSysVar.autoivcInput && node instanceof OmTreeNode &&
                    node.isAutoIvcInput()) {
                    this.showSysVar.autoivcInput = true;
                    this.sysAndVar.push({
                        'name': 'Auto-IVC Input',
                        'color': OmStyle.color.autoivcInput
                    })
                }
                else if (!this.showSysVar.unconnectedInput && node.isUnconnectedInput()) {
                    this.showSysVar.unconnectedInput = true;
                    this.sysAndVar.push({
                        'name': 'Unconnected Input',
                        'color': OmStyle.color.unconnectedInput
                    })
                }
                else if (!this.showSysVar.desvar) {
                    this.showSysVar.desvar = true;
                    this.sysAndVar.push({
                        'name': 'Optimization Variables',
                        'color': OmStyle.color.desvar
                    })
                }
            }
        }
    }

    /** Add symbols for all of the items that were discovered */
    _setupContents() {
        super._setupContents();

        const solversLegend = d3.select('#solvers-legend')
        for (let item of this.linearSolvers) this._addItem(item, solversLegend);

        solversLegend.style('width', solversLegend.node().scrollWidth + 'px');

    }

    /**
     * Wipe the current solvers legend area and populate with the current setting.
     */
    updateSolvers() {
        const solversLegendTitle = d3.select('#solvers-legend-title');
        solversLegendTitle.text(OmTreeNode.showLinearSolverNames?
            "Linear Solvers" : "Non-Linear Solvers");

        const solversLegend = d3.select('#solvers-legend');
        solversLegend.html('');

        const solvers = OmTreeNode.showLinearSolverNames? this.linearSolvers : this.nonLinearSolvers;
        for (const item of solvers) this._addItem(item, solversLegend);

        solversLegend.style('width', solversLegend.node().scrollWidth + 'px');
    }
}
