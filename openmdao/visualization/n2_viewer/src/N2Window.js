class N2Window {
    /**
     * Keep increasing z-index of the focused window to keep it on top.
     * Max z-index is 2147483647. It will be unusual here for it to climb
     * above 100, and even extreme cases (e.g. a diagram that's been in use
     * for weeks with lots of windows) shouldn't get above a few thousand.
     */
    static zIndex = 10;
    static container = null;

    /**
     * Clone the template window defined in index.html, setup some
     * references to various elements, and add the drag event listener.
     * @param {String} [newId = null] HTML id for the new window. A UUID is generated if null.
     */
    constructor(newId = null) {
        this._window = d3.select('#window-template')
            .clone(true)
            .attr('id', newId ? newId : uuidv4())

        if (!N2Window.container) {
            N2Window.container = d3.select('#n2-windows');
        }

        this._header = this._window.select('.window-header');
        this._title = this._header.select('.window-title');
        this._closeButton = this._header.select('.window-close-button');
        this._body = this._window.select('.window-body');
        this._footer = this._window.select('.window-footer');

        const self = this;
        this._closeButton.on('click', e => { self.close(); })

        this._setupDrag();
        this.bringToFront(true);
        this._setupResizers();
    }

    /**
     * Listen for the event to begin dragging the window. The start of the event
     * also brings the window to the front.
     */
    _setupDrag() {
        const self = this;

        this._header.on('mousedown', function () {
            const dragDiv = self._window;

            self.bringToFront();
            dragDiv.style('cursor', 'grabbing')
                .select('.window-header').style('cursor', 'grabbing');

            const dragStart = [d3.event.pageX, d3.event.pageY];
            let newTrans = [0, 0];

            const w = d3.select(window)
                .on("mousemove", e => {
                    newTrans = [ d3.event.pageX - dragStart[0], d3.event.pageY - dragStart[1] ];
                    dragDiv.style('transform', `translate(${newTrans[0]}px, ${newTrans[1]}px)`)
                })
                .on("mouseup", e => {
                    // Convert the translate to style position
                    self._setPos(self._getPos());

                    dragDiv.style('cursor', 'auto')
                        .style('transform', null)
                        .select('.window-header')
                        .style('cursor', 'grab');
                    
                    // Remove event listeners
                    w.on("mousemove", null).on("mouseup", null);
                });

            d3.event.preventDefault();
        });
    }

    /**
     * Compute the position of all four sides of the window relative to the container.
     * @returns {Object} Each key represents the position in pixels.
     */
    _getPos() {
        const parentPos = N2Window.container.node().getBoundingClientRect(),
            childPos = this._window.node().getBoundingClientRect();

        let posInfo = {
            top: childPos.top - parentPos.top,
            right: parentPos.right - childPos.right,
            bottom: parentPos.bottom - childPos.bottom,
            left: childPos.left - parentPos.left,
            width: childPos.width,
            height: childPos.height,
            parentWidth: parentPos.width,
            parentHeight: parentPos.height
        };

        return posInfo;
    }

    _setPos(newPos) {
        // All of the values need to be set because some may have started as "auto"
        for (const s of ['top', 'right', 'bottom', 'left', 'width', 'height']) {
            this._window.style(s, `${newPos[s]}px`);
        }
    }

    /**
     * Add event handlers for each of the 8 resizer elements surrounding the window.
     */
    _setupResizers() {
        const self = this;

        // Reference to div containing the 8 resizer elements
        const resizerDiv = this._window.select('.resize');

        // Base names of all the resizer elements
        const resizerNames = ['top', 'top-right', 'right', 'bottom-right',
            'bottom', 'bottom-left', 'left', 'top-left'];

        // For each side, 'mult' refers to whether a coordinate is to be added or subtracted.
        // 'idx' refers to the index of the delta x or y value of the new mouse position.
        const dirVals = {
            top: { mult: 1, idx: 1 },
            right: { mult: -1, idx: 0 },
            bottom: { mult: -1, idx: 1 },
            left: { mult: 1, idx: 0 }
        }

        const dragDiv = self._window;

        // Set up a mousedown event listener for each of the 8 elements.
        for (let name of resizerNames) {
            const resizer = resizerDiv.select(`.rsz-${name}`);
            resizer.on('mousedown', function () {
                const dirs = name.split('-'); // From the name, figure out which diections we're handling
                const startSize = self._getPos();

                self.bringToFront();

                const dragStart = [d3.event.pageX, d3.event.pageY];
                let newPos = [0, 0]; // Delta values of the current mouse position vs. start position
                let newSize = {}; // Object to store newly computed positions in

                const w = d3.select(window)
                    .on("mousemove", e => {
                        newPos = [d3.event.pageX - dragStart[0], d3.event.pageY - dragStart[1]];
                        Object.assign(newSize, startSize)

                        for (let i in dirs) {
                            newSize[dirs[i]] = startSize[dirs[i]] + (newPos[dirVals[dirs[i]].idx] * dirVals[dirs[i]].mult);
                        }
                        newSize.width = startSize.parentWidth - (newSize.right + newSize.left);
                        newSize.height = startSize.parentHeight - (newSize.top + newSize.bottom);

                        self._setPos(newSize);
                    })
                    .on("mouseup", e => {
                        w.on("mousemove", null).on("mouseup", null);
                    });

                d3.event.preventDefault();
            });
        }
    }

    get hidden() { return this._window.classed('window-inactive'); }

    bringToFront(force = false) {
        if (force || this._window.style('z-index') < N2Window.zIndex) {
            N2Window.zIndex++;
            this._window.style('z-index', N2Window.zIndex);
        }
    }

    show() {
        this._window
            .classed('window-inactive', false)
            .classed('window-active', true)
        this.bringToFront();
        this._getPos();

        return this;
    }

    hide() {
        this._window
            .classed('window-inactive', true)
            .classed('window-active', false)

        return this;
    }

    title(newTitle = null) {
        if (newTitle) this._title.html(newTitle);
        else newTitle = this._title.html();

        return newTitle;
    }

    /**
     * Change the styling of the window to a preset theme and/or return it.
     * @param {String} [newTheme = null] The name of the theme to change to. No change if null.
     * @returns {String} The name of the theme.
     */
    theme(newTheme = null) {
        const classes = this._window.attr('class');
        const curTheme = classes.replace(/^.*(window-theme-\S+).*$/, "$1")

        if (newTheme && newTheme != curTheme) {
            this.window
                .classed(`window-theme-${curTheme}`, false)
                .classed(`window-theme-${newTheme}`, true)
        }
        else {
            newTheme = curTheme;
        }

        return newTheme;
    }

    set(options) {
        for (const optName in options) {
            switch (optName) {
                case 'title':
                    this.title(options[optName]);
                    break;

                case 'width':
                    this._window.style(optName, options[optName]);
                    break;

                case 'height':
                    this._window.style(optName, options[optName]);
                    break;

                case 'top':
                    this._window.style(optName, options[optName]);

                case 'left':
                    this._window.style(optName, options[optName]);
            }
        }

        return this;
    }

    close() {
        this._header.on('mousedown', null);
        this._closeButton.on('click', null);
        this._window.remove();
    }
}