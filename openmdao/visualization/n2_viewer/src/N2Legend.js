/**
 * Draw a symbol describing each of the element types.
 * @typedef N2Legend
 * @property {Boolean} shown Whether the legend is currently drawn or not.
 */
class N2Legend extends N2WindowDraggable {
    /**
     * Initializes the legend object.
     * @param {ModelData} modelData Symbols are only displayed if they're in the model
     */
    constructor(modelData) {
        super('n2win-legend');
    
        // TODO: The legend should't have to search through modelData itself,
        // this info can be collected as modelData is built.
        this.nodes = modelData.tree.children;
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

        this.showN2Symbols = {
            'scalar': false,
            'vector': false,
            'collapsedVariables': false
        };

        this.sysAndVar = [
            { 'name': "Connection", 'color': N2Style.color.connection },
            { 'name': "Collapsed", 'color': N2Style.color.collapsed },
            { 'name': "Declared Partial", 'color': N2Style.color.declaredPartial }
        ];

        this.n2Symbols = [];
        const rootLinearSolver =
            N2Style.solverStyleObject.find(x => x.ln === modelData.tree.linear_solver);
        const rootNonLinearSolver =
            N2Style.solverStyleObject.find(x => x.nl === modelData.tree.nonlinear_solver);
        this.linearSolvers = [
            { 'name': modelData.tree.linear_solver, 'color': rootLinearSolver.color }
        ];

        this.nonLinearSolvers = [
            { 'name': modelData.tree.nonlinear_solver, 'color': rootNonLinearSolver.color }
        ];

        const legendDiv = d3.select('#legend-div');
        legendDiv.style('visibility', null);
        this.body.node().appendChild(legendDiv.node());      
        this._setDisplayBooleans(this.nodes);

        this.title('<span class="icon-key"></span>');
        this._setupContents();

        this.theme('legend');
        this.sizeToContent();
        this.move(50, -10);
    }

    /** Override N2Window::close() to just hide the legend. */
    close() {
        d3.select('#legend-button').attr('class', 'fas icon-key');
        this.hide();
        return this;
    }

    /**
     * Determine which types of nodes are present in the model and only mark
     * those types of keys to be displayed in the legend.
     * @param {Object} nodes The model node tree.
     */
    _setDisplayBooleans(nodes) {
        for (let node of nodes) {
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
                let solverStyle = N2Style.solverStyleObject.find(x => x.ln === linearSolver);
                this.linearSolvers.push({
                    'name': solverStyle.ln,
                    'color': solverStyle.color
                });
            }

            if (nonLinearSolverIndex < 0 && nonLinearSolver !== undefined) {
                let solverStyle = N2Style.solverStyleObject.find(x => x.nl === nonLinearSolver);
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
                        'color': N2Style.color.group
                    })
                }
                else if (!this.showSysVar.component && node.isComponent()) {
                    this.showSysVar.component = true;
                    this.sysAndVar.push({
                        'name': 'Component',
                        'color': N2Style.color.component
                    })
                }
                this._setDisplayBooleans(node.children);
            }
            else {
                if (!this.showSysVar.input && node.isInput()) {
                    this.showSysVar.input = true;
                    this.sysAndVar.push({
                        'name': 'Input',
                        'color': N2Style.color.input
                    })
                }
                else if (!this.showSysVar.outputExplicit && node.isExplicitOutput()) {
                    this.showSysVar.outputExplicit = true;
                    this.sysAndVar.push({
                        'name': 'Explicit Output',
                        'color': N2Style.color.outputExplicit
                    })
                }
                else if (!this.showSysVar.outputImplicit && node.isImplicitOutput()) {
                    this.showSysVar.outputImplicit = true;
                    this.sysAndVar.push({
                        'name': 'Implicit Output',
                        'color': N2Style.color.outputImplicit
                    })
                }
                else if (!this.showSysVar.autoivcInput && node.isAutoIvcInput()) {
                    this.showSysVar.autoivcInput = true;
                    this.sysAndVar.push({
                        'name': 'Auto-IVC Input',
                        'color': N2Style.color.autoivcInput
                    })
                }
                else if (!this.showSysVar.unconnectedInput && node.isUnconnectedInput()) {
                    this.showSysVar.unconnectedInput = true;
                    this.sysAndVar.push({
                        'name': 'Unconnected Input',
                        'color': N2Style.color.unconnectedInput
                    })
                }
                else if (!this.showSysVar.desvar) {
                    this.showSysVar.desvar = true;
                    this.sysAndVar.push({
                        'name': 'Optimization Variables',
                        'color': N2Style.color.desvar
                    })
                }
            }
        }
    }

    /**
     * Create elements in the legend divs for the supplied item
     * @param {Object} item Contains the name and color of the item
     * @param {Object} container The div to append into
     */
    _addItem(item, container) {
        const newDiv = container
            .append('div')
            .attr('class', 'legend-box-container');

        newDiv.append('div')
            .attr('class', 'legend-box')
            .style('background-color', item.color);

        newDiv.append('p')
            .html(item.name);
    }

    /** Add symbols for all of the items that were discovered */
    _setupContents() {
        const sysVarContainer = d3.select('#sys-var-legend');
        for (let item of this.sysAndVar) this._addItem(item, sysVarContainer);

        sysVarContainer.style('width', sysVarContainer.node().scrollWidth + 'px')

        const solversLegend = d3.select('#solvers-legend')
        for (let item of this.linearSolvers) this._addItem(item, solversLegend);

        solversLegend.style('width', solversLegend.node().scrollWidth + 'px');

    }

    /**
     * Wipe the current solvers legend area and populate with the other type.
     * @param {Boolean} linear True to use linear solvers, false for non-linear.
     */
    toggleSolvers(linear) {
        const solversLegendTitle = d3.select('#solvers-legend-title');
        solversLegendTitle.text(linear ? "Linear Solvers" : "Non-Linear Solvers");

        const solversLegend = d3.select('#solvers-legend');
        solversLegend.html('');

        const solvers = linear ? this.linearSolvers : this.nonLinearSolvers;
        for (let item of solvers) this._addItem(item, solversLegend);

        solversLegend.style('width', solversLegend.node().scrollWidth + 'px');
    }
}

/**
 * Display a modal window with helpful information.
 * @typedef N2Help
 */
class N2Help extends N2Window {
    constructor() {
        super();
        this.theme('help')
            .setList({ left: '100px', top: '20px', right: '100px', height: '800px' })
            .title('Instructions')
            .footerText('OpenMDAO Model Hierarchy and N2 diagram');

        this.body.append('p')
            .text(
                'Left clicking on a node in the partition tree will navigate to that node. ' +
                'Right clicking on a node in the model hierarchy will collapse/expand it. ' +
                'A click on any element in the N2 diagram will allow those arrows to persist.');

        this.body.append('h1').text('Toolbar Help');
        this.body.append('svg').append('use').attr('href', '#help-graphic');

        this.show().modal(true);
    }
}
