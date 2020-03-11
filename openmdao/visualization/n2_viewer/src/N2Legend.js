/**
 * Draw a box under the diagram describing each of the element types.
 * @typedef N2Legend
 * @property {String} title The label to put at the top of the legend.
 * @property {Boolean} shown Whether the legend is currently drawn or not.
 */
class N2Legend {
    /**
     * Initializes the legend object, but doesn't draw it yet.
     * @param {String} [title = "LEGEND"] The label at the top of the legend box.
     */
    constructor(modelData) {
        this.nodes = modelData.tree.children;
        this.showSysVar = {
            group: false,
            component: false,
            input: false,
            unconnectedInput: false,
            outputExplicit: false,
            outputImplicit: false,
            collapsed: true,
            connection: true
        };

        this.showN2Symbols = {
            scalar: false,
            vector: false,
            collapsedVariables: false
        };

        this.sysAndVar = [{
            name: "Connection",
            color: N2Style.color.connection
        }, {
            name: "Collapsed",
            color: N2Style.color.collapsed
        }];
        this.n2Symbols = [];
        const rootLinearSolver = N2Style.solverStyleObject.find(x => x.ln === modelData.tree.linear_solver);
        const rootNonLinearSolver = N2Style.solverStyleObject.find(x => x.nl === modelData.tree.nonlinear_solver);
        this.linearSolvers = [{
            name: modelData.tree.linear_solver,
            color: rootLinearSolver.color
        }];
        this.nonLinearSolvers = [{
            name: modelData.tree.nonlinear_solver,
            color: rootNonLinearSolver.color
        }];

        this.setDisplayBooleans(this.nodes);


        this.title = "Legend";
        this.shown = false; // Not shown until show() is called

        this._div = d3.select("#legend-div");


        this.setup();
    }

    setDisplayBooleans(nodes) {
        for (let i = 0; i < nodes.length; i++) {
            const node = nodes[i];
            const {
                group,
                component,
                input,
                unconnectedInput,
                outputExplicit,
                outputImplicit,
                collapsed,
                connection
            } = this.showSysVar;


            const linearSolver = node.linear_solver;
            const nonLinearSolver = node.nonlinear_solver;

            const linearSolverIndex = this.linearSolvers.indexOf(this.linearSolvers.find(x => x.name === linearSolver))
            const nonLinearSolverIndex = this.nonLinearSolvers.indexOf(this.nonLinearSolvers.find(x => x.name === nonLinearSolver));

            if (linearSolverIndex < 0 && linearSolver !== undefined) {
                let solverStyle = N2Style.solverStyleObject.find(x => x.ln === linearSolver);
                this.linearSolvers.push({
                    name: solverStyle.ln,
                    color: solverStyle.color
                });
            }
            if (nonLinearSolverIndex < 0 && nonLinearSolver !== undefined) {
                let solverStyle = N2Style.solverStyleObject.find(x => x.nl === nonLinearSolver);
                this.nonLinearSolvers.push({
                    name: solverStyle.nl,
                    color: solverStyle.color
                });
            }

            if (node.children !== undefined && node.children.length > 0) {
                if (!this.showSysVar.group && node.subsystem_type === 'group') {
                    this.showSysVar.group = true;
                    this.sysAndVar.push({
                        name: 'Group',
                        color: N2Style.color.group
                    })
                } else if (!this.showSysVar.component && node.subsystem_type === 'component') {
                    this.showSysVar.component = true;
                    this.sysAndVar.push({
                        name: 'Component',
                        color: N2Style.color.component
                    })
                }
                this.setDisplayBooleans(node.children);
            } else {
                if (!this.showSysVar.input && node.type === 'param') {
                    this.showSysVar.input = true;
                    this.sysAndVar.push({
                        name: 'Input',
                        color: N2Style.color.param
                    })
                } else if (!this.showSysVar.outputExplicit && node.type === 'unknown' && !node.implicit) {
                    this.showSysVar.outputExplicit = true;
                    this.sysAndVar.push({
                        name: 'Explicit Output',
                        color: N2Style.color.unknownExplicit
                    })
                } else if (!this.showSysVar.outputImplicit && node.type === 'unknown' && node.implicit) {
                    this.showSysVar.outputImplicit = true;
                    this.sysAndVar.push({
                        name: 'Implicit Output',
                        color: N2Style.color.unknownImplicit
                    })
                } else if (!this.showSysVar.unconnectedInput && node.type === "unconnectedParam") {
                    this.showSysVar.unconnectedInput = true;
                    this.sysAndVar.push({
                        name: 'Unconnected Input',
                        color: N2Style.color.unconnectedParam
                    })
                }
            }
        }


    }

    setup() {
        const sysVarContainer = document.getElementById("sys-var-legend");
        const linearContainer = document.getElementById("linear-legend");

        for (let i = 0; i < this.sysAndVar.length; i++) {
            const item = this.sysAndVar[i];

            const legendBoxContainer = document.createElement("div");
            legendBoxContainer.setAttribute('class', 'legend-box-container');

            const legendBox = document.createElement('div');
            legendBox.setAttribute('class', 'legend-box');
            legendBox.style.backgroundColor = item.color;

            const title = document.createElement('p');
            title.innerHTML = item.name;

            legendBoxContainer.appendChild(legendBox);
            legendBoxContainer.appendChild(title);

            sysVarContainer.appendChild(legendBoxContainer);
        }

        for (let i = 0; i < this.linearSolvers.length; i++) {
            const item = this.linearSolvers[i];

            const legendBoxContainer = document.createElement("div");
            legendBoxContainer.setAttribute('class', 'legend-box-container');

            const legendBox = document.createElement('div');
            legendBox.setAttribute('class', 'legend-box');
            legendBox.style.backgroundColor = item.color;

            const title = document.createElement('p');
            title.innerHTML = item.name;

            legendBoxContainer.appendChild(legendBox);
            legendBoxContainer.appendChild(title);

            linearContainer.appendChild(legendBoxContainer);
        }
    }

    hide() {
        this._div.style('display', 'none');
        this.shown = false;
    }

    show(showLinearSolverNames, solverStyles) {
        this.shown = true;
        this._div.style('display', 'flex');
    }

    toggleSolvers(linear) {
        const solversContainer = document.getElementById("linear-legend");
        if (linear) {
            solversContainer.innerHTML = '';
            for (let i = 0; i < this.linearSolvers.length; i++) {
                const item = this.linearSolvers[i];

                const legendBoxContainer = document.createElement("div");
                legendBoxContainer.setAttribute('class', 'legend-box-container');

                const legendBox = document.createElement('div');
                legendBox.setAttribute('class', 'legend-box');
                legendBox.style.backgroundColor = item.color;

                const title = document.createElement('p');
                title.innerHTML = item.name;

                legendBoxContainer.appendChild(legendBox);
                legendBoxContainer.appendChild(title);

                solversContainer.appendChild(legendBoxContainer);
            }
        } else {
            solversContainer.innerHTML = '';
            for (let i = 0; i < this.nonLinearSolvers.length; i++) {
                const item = this.nonLinearSolvers[i];

                const legendBoxContainer = document.createElement("div");
                legendBoxContainer.setAttribute('class', 'legend-box-container');

                const legendBox = document.createElement('div');
                legendBox.setAttribute('class', 'legend-box');
                legendBox.style.backgroundColor = item.color;

                const title = document.createElement('p');
                title.innerHTML = item.name;

                legendBoxContainer.appendChild(legendBox);
                legendBoxContainer.appendChild(title);

                solversContainer.appendChild(legendBoxContainer);
            }
        }
    }


    /**
     * If legend is shown, hide it; if it's hidden, show it.
     * @param {Boolean} showLinearSolverNames Determines solver name type displayed.
     * @param {Object} solverStyles Solver names, types, and styles including color.
     */
    toggle(showLinearSolverNames, solverStyles) {
        if (this.shown) {
            this.hide();
        } else this.show(showLinearSolverNames, solverStyles);
    }


}