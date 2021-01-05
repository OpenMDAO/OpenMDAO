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
     * references to various elements.
     * @param {String} [newId = null] HTML id for the new window. A UUID is generated if null.
     */
    constructor(newId = null) {
        this._window = d3.select('#window-template')
            .clone(true)
            .attr('style', null)
            .attr('id', newId ? newId : uuidv4())

        if (!N2Window.container) {
            N2Window.container = d3.select('#n2-windows');
        }

        this._main = this._window.select('.main-window');
        this._header = this._window.select('.window-header');
        this._title = this._header.select('.window-title');
        this._closeButton = this._header.select('.window-close-button');
        this._body = this._window.select('.window-body');
        this._footer = this._window.select('.window-footer');

        const self = this;
        this._closeButton.on('click', e => { self.close(); })
        
        this.bringToFront(true);
    }

    get header() { return this._header; }
    get body() { return this._body; }
    get footer() { return this._footer; }

    /**
     * Compute the position of all four sides of the window relative to the container.
     * @returns {Object} Each key represents the position in pixels.
     */
    _getPos(container = N2Window.container.node()) {
        const parentPos = container.getBoundingClientRect(),
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
            this.set(s, `${newPos[s]}px`);
        }
    }

    get hidden() { return this._window.classed('window-inactive'); }
    
    set hidden(hide) { 
        if (!hide) this.bringToFront();
        this._window.classed('window-inactive', hide);
    }

    bringToFront(force = false) {
        if (force || this._window.style('z-index') < N2Window.zIndex) {
            N2Window.zIndex++;
            this._window.style('z-index', N2Window.zIndex);
        }
    }

    show() {
        this.hidden = false;
        return this;
    }

    hide() {
        this.hidden = true;
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
     * @returns if newTheme is null, a string with the current theme name;
     *  otherwise a reference to this
     */
    theme(newTheme = null) {
        const contents = this._window.select('div.window-contents'),
            classes = contents.attr('class'),
            curTheme = classes.replace(/^.*(window-theme-\S+).*$/, "$1")

        if (newTheme) {
            contents
                .classed(`window-theme-${curTheme}`, false)
                .classed(`window-theme-${newTheme}`, true)
        }
        else {
            return curTheme;
        }

        return this;
    }

    /**
     * Set a style or special property for the window. Recognized special
     * properties: title, theme.
     * @param {String} opt The name of the style/property to set
     * @param {String} val The value to set it to.
     * @returns {Object} Reference to this.
     */
    set(opt, val) {
        switch (opt) {
            case 'title':
                this.title(val);
                break;
            case 'theme':
                this.theme(val);
                break;
            default:
                this._window.style(opt, val);
        }

        return this;
    }

    /**
     * Iterate over a list of styles/properties w/values and set them.
     * @param {Object} options Dictionary of style/value pairs.
     * @returns {Object} Reference to this.
     */
    setList(options) {
        for (const optName in options) {
            this.set(optName, options[optName]);
        }

        return this;
    }

    close() {
        this._closeButton.on('click', null);
        this._window.remove();
    }

    showFooter() {
        this._footer.classed('window-inactive', false);
        return this;
    }

    /**
     * Change the color of both the header and footer
     * @param {String} color An HTML-compatible color value
     */
    ribbonColor(color) {
        this._header.style('background-color', color);
        this._footer.style('background-color', color);

        return this;
    }

    /**
     * Relocate the window to a position near the mouse
     * @param {Object} event The triggering event containing the position.
     * @param {Number} [offset = 15] Distance from mouse to place window.
     */
    move(event, offset = 15) {
        if (!this.active) return;

        let pos = this._getPos();

        // Mouse is in left half of browser, put window to right of mouse
        if (event.clientX < window.innerWidth / 2) {
            pos.left = event.pageX + offset;
            pos.right = pos.left + pos.width;
        }
        // Mouse is in right half of browser, put window to left of mouse
        else {
            pos.right = event.pageX - offset;
            pos.left = pos.right - pos.width;
        }

        // Mouse is in top half of browser, put window below mouse
        if (event.clientY < window.innerHeight / 2) {
            pos.top = event.pageY + offset;
            pos.bottom = pos.top + pos.height;
        }
        // Mouse is in bottom half of browser, put window above mouse
        else {
            pos.bottom = event.pageY - offset;
            pos.top = pos.bottom - pos.height;
        }

        this._setPos(pos);
        return this;
    }

    sizeToContent() {
        const contentWidth = this.body.node().scrollWidth,
            contentHeight = this.body.node().scrollHeight,
            headerHeight = this._header.node().scrollHeight,
            footerHeight = this._footer.classed('window-inactive')?
                parseInt(this._window.select('.window-contents').style('border-radius')) :
                this._footer.node().scrollHeight;

        const totalHeight = contentHeight + headerHeight + footerHeight + 2;

        this.setList({width: contentWidth + 'px', height: totalHeight + 'px'});
    }
}

class N2WindowDraggable extends N2Window {
    /** Execute the base class constructor and set up drag event handler */
    constructor(newId = null) {
        super(newId);
        this._setupDrag();
    }

    close() {
        this._header.on('mousedown', null);
        super.close();
    }

    /**
     * Listen for the event to begin dragging the window. The start of the event
     * also brings the window to the front.
     */
    _setupDrag() {
        const self = this;

        this._header
            .classed('window-draggable-header', true)
            .on('mousedown', function () {
            const dragDiv = self._window;

            self.bringToFront();
            dragDiv.style('cursor', 'grabbing')
                .select('.window-header').style('cursor', 'grabbing');

            const dragStart = [d3.event.pageX, d3.event.pageY];
            let newTrans = [0, 0];

            const w = d3.select(window)
                .on("mousemove", e => {
                    newTrans = [d3.event.pageX - dragStart[0], d3.event.pageY - dragStart[1]];
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
}

class N2WindowResizable extends N2WindowDraggable {
    constructor(newId = null) {
        super(newId);
        this._setupResizers();
    }

    /**
     * Add event handlers for each of the 8 resizer elements surrounding the window.
     */
    _setupResizers() {
        const self = this;

        const resizerClassNames = {
            'top': 'horizontal',
            'top-right': 'corner',
            'right': 'vertical',
            'bottom-right': 'corner',
            'bottom': 'horizontal',
            'bottom-left': 'corner',
            'left': 'vertical',
            'top-left': 'corner'
        }

        // Add div to contain the 8 resizer elements
        const resizerDiv = this._window.select('.main-window')
            .append('div')
            .attr('class', 'resize');

        // For each side, 'mult' refers to whether a coordinate is to be added or subtracted.
        // 'idx' refers to the index of the delta x or y value of the new mouse position.
        const dirVals = {
            top: { mult: 1, idx: 1 },
            right: { mult: -1, idx: 0 },
            bottom: { mult: -1, idx: 1 },
            left: { mult: 1, idx: 0 }
        }

        // Set up a mousedown event listener for each of the 8 elements.
        for (const name in resizerClassNames) {
            // Add the div that the resizer mouse event handler will be on
            const resizer = resizerDiv.append('div')
                // Class style settings determine where each div is positioned
                .attr('class', `rsz-${name} rsz-${resizerClassNames[name]}`);

            const dirs = name.split('-'); // From class name, figure out which directions to handle
            resizer.on('mousedown', function () {
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
                            newSize[dirs[i]] = startSize[dirs[i]] +
                                (newPos[dirVals[dirs[i]].idx] * dirVals[dirs[i]].mult);
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
}