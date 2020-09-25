/**
 * Draw a symbol describing each of the element types.
 * @typedef N2Legend
 * @property {Boolean} shown Whether the legend is currently drawn or not.
 */
class N2Legend {
    /**
     * Initializes the legend object. 
     * @param {ModelData} modelData Symbols are only displayed if they're in the model
     */
    constructor(modelData) {
        this._div = d3.select("#legend-div");

        // TODO: The legend should't have to search through modelData itself,
        // this info can be collected as modelData is built.
        this.nodes = modelData.tree.children;
        this.showSysVar = {
            'group': false,
            'component': false,
            'input': false,
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
            { 'name': "Declared Partial", 'color': N2Style.color.declaredPartial}
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

        this._setDisplayBooleans(this.nodes);
        this._setupContents();

        // Get the initial setting from the style sheet
        this.hidden = (this._div.style('visibility') == 'hidden');

        this._setupDrag();

        let self = this;
        this.closeDiv = d3.select('#close-legend');
        this.closeButton = this.closeDiv.select('p');

        this.closeDiv
            .on('mouseenter', e => { self.closeButton.style('color', 'red'); })
            .on('mouseout', e => { self.closeButton.style('color', 'black'); })
            .on('click', e => { 
                self.hide(); 
                self.closeButton.style('color', 'black');
                d3.select('#legend-button').attr('class', 'fas icon-key');
            })
    }

    _setDisplayBooleans(nodes) {
        for (let node of nodes) {
            const {
                group,
                component,
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

    /** Listen for the event to begin dragging the legend */
    _setupDrag() {
        const self = this;

        this._div.on('mousedown', function() {
            let dragDiv = d3.select(this);
            dragDiv.style('cursor', 'grabbing')
                // top style needs to be set explicitly before releasing bottom:
                .style('top', dragDiv.style('top'))   
                .style('bottom', 'initial');

            self._startPos = [d3.event.clientX, d3.event.clientY]
            self._offset = [d3.event.clientX - parseInt(dragDiv.style('left')), 
                d3.event.clientY - parseInt(dragDiv.style('top'))];

            let w = d3.select(window)
                .on("mousemove", e => {
                    dragDiv
                        .style('top', (d3.event.clientY - self._offset[1]) + 'px')
                        .style('left', (d3.event.clientX - self._offset[0]) + 'px');
                })
                .on("mouseup", e => {
                    dragDiv.style('cursor', 'grab');
                    w.on("mousemove", null).on("mouseup", null);
                    
                });

            d3.event.preventDefault();
        })
    }

    hide() {
        this._div.style('visibility', 'hidden');
        this.hidden = true;
    }

    show() {
        this._div.style('visibility', 'visible');
        this.hidden = false;
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

    /**
     * If legend is shown, hide it; if it's hidden, show it.
     * @param {Boolean} showLinearSolverNames Determines solver name type displayed.
     * @param {Object} solverStyles Solver names, types, and styles including color.
     */
    toggle(showLinearSolverNames, solverStyles) {
        if (this.hidden) this.show();
        else this.hide();
    }
}
