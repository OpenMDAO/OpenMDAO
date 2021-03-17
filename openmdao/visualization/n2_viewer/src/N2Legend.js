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
    /**
     * Using data generated by N2Toolbar, document each of the buttons.
     * @param {Object} helpInfo Data structure containing N2Toolbar button info.
     */
    constructor(helpInfo) {
        super();

        const version = d3.select('div#all_pt_n2_content_div').attr('data-openmdao-version');

        this.theme('help')
            .setList({ left: '100px', top: '20px', right: '100px', height: '850px' })
            .title('Instructions')
            .footerText(`OpenMDAO Version ${version} Model Hierarchy and N2 diagram`);

        // Take ownership of the help div contents defined in index.html
        const newParent = this.body.node();
        const oldParent = d3.select('#toolbar-help-container').node();

        while (oldParent.childNodes.length > 0) {
            newParent.appendChild(oldParent.childNodes[0]);
        }

        oldParent.remove();

        this.helpDiv = this.body.select('div.help-graphic');
        this.helpSvg = this.helpDiv.select('#help-graphic-svg');

        this._addButtonHelpText(helpInfo);

        this.show().modal(true);
    }

    /** Override N2Window::close() to just hide the help window. */
    close() {
        this.modal(false);
        this.hide();
        return this;
    }

    /**
     * Add a line of text for each button. Buttons not contained in an expandable
     * menu are labeled to the left, while the expansion buttons are described
     * in groups at the right.
     * @param {Object} helpInfo Data structure containing N2Toolbar button info.
     */
    _addButtonHelpText(helpInfo) {
        let topPx = 40;
        for (const btnId in helpInfo.buttons) {
            const btn = helpInfo.buttons[btnId];

            // Detail the "primary" toolbar buttons
            if (!btn.expansionItem) {
                const btnText = this.helpDiv.append('p').attr('class', 'help-text');
                btn.helpWinText = btnText;
                btnText.style('left', helpInfo.width + 5 + 'px')
                    .style('top', btn.bbox.top + btn.bbox.height / 2 - 13 + 'px')
                    .text(btn.desc);

                // Detail the expansion buttons
                let grp = null;
                if (btnId in helpInfo.primaryButtons) {
                    btnText.classed('help-button-group', true)
                    grp = this.helpDiv.append('div')
                        .attr('id', btnId + '-help-group')
                        .attr('class', 'help-button-group')
                        .style('top', topPx + 'px')

                    for (const memId of helpInfo.primaryButtons[btnId]) {
                        const memClasses = d3.select('#' + memId).attr('class').split(/ /);

                        let memClass = '';
                        for (const mc of memClasses) {
                            if (mc.match(/^icon-/)) {
                                memClass = mc;
                                break;
                            }
                        }
                        topPx += 38;
                        grp.append('p').attr('class', 'help-text')
                            .html(`<i class="fas ${memClass} help-text-icon"></i>${helpInfo.buttons[memId].desc}`)
                    }

                    topPx += 30;

                    this._drawGroupLines(btnText, grp)
                }
            }
        }
    }

    /**
     * Draw a path in SVG connecting the button that opens the expansion
     * to the boxes containing descriptions of the individual buttons.
     */
    _drawGroupLines(btnText, grp) {
        const winBRect = this.helpDiv.node().getBoundingClientRect(),
            textBRect = btnText.node().getBoundingClientRect(),
            grpBRect = grp.node().getBoundingClientRect();

        const coords = {
            ul: { x: textBRect.right - winBRect.left + 5, y: textBRect.top - winBRect.top },
            ur: { x: grpBRect.left - winBRect.left, y: grpBRect.top - winBRect.top + 0.5 },
            bl: { x: textBRect.right - winBRect.left + 5, y: textBRect.bottom - winBRect.top },
            br: { x: grpBRect.left - winBRect.left, y: grpBRect.bottom - winBRect.top - 0.5 }
        }

        const curve = {
            ul: { x: coords.ul.x + 5, y: coords.ul.y + 5 },
            bl: { x: coords.bl.x + 5, y: coords.bl.y - 5 }
        }

        const path = `M${coords.ur.x},${coords.ur.y} L${coords.ul.x},${coords.ul.y} ` +
            `C${curve.ul.x},${curve.ul.y} ${curve.bl.x},${curve.bl.y} ${coords.bl.x},${coords.bl.y} ` +
            `L${coords.br.x},${coords.br.y}`

        this.helpSvg.append('path').attr('d', path).attr('class', 'help-line');
    }
}
