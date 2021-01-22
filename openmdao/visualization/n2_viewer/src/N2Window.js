/**
 * Create a clone of the #window-template defined in index.html and provide
 * management functions such as setting size, position, ribbon color, etc.
 * @typedef N2Window
 */
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
     * @param {String} [cloneId = null] The id of a window to clone other than
     *  the original template.
     */
    constructor(newId = null, cloneId = null) {
        // The primary reference for the new window
        this._window = d3.select(cloneId ? cloneId : '#window-template')
            .clone(true)
            .attr('id', newId ? newId : 'n2win' + uuidv4());

        if (!N2Window.container) {
            N2Window.container = d3.select('#n2-windows');
        }

        this._main = this._window.select('.main-window'); // Not referenced very often
        this._header = this._window.select('.window-header');
        this._title = this._header.select('.window-title');
        this._closeButton = this._window.select('.window-close-button');
        this._body = this._window.select('.window-body');
        this._footer = this._window.select('.window-footer');

        this._enabledModal = false;

        const self = this;
        this._closeButton.on('click', e => { self.close(); })

        this.bringToFront(true);
    }

    // Read-only access to stored references
    get main() { return this._main; }
    get window() { return this._window; }
    get header() { return this._header; }
    get body() { return this._body; }
    get footer() { return this._footer; }
    get closeButton() { return this._closeButton; }

    /**
     * Compute the position of all four sides of the window relative to the container.
     * CAUTION: This only works correctly for a displayed element.
     * @param {Object} [container = N2Window.container.node()] HTML element containing window.
     * @returns {Object} Each key represents the position in pixels.
     */
    _getPos(container = N2Window.container.node()) {
        const parentPos = container.getBoundingClientRect(),
            childPos = this.window.node().getBoundingClientRect();

        // If hidden, move offscreen and display to get correct width and height
        if (this.hidden) {
            this.set('left', '-15000px').show();

            const tmpPos = this.window.node().getBoundingClientRect();
            childPos.width = tmpPos.width;
            childPos.height = tmpPos.height;

            this.hide().set('left', childPos.left);
        }

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

    /**
     * Update the window geometry with new info.
     * @param {Object} newPos Contains the bounding box data.
     * @returns {N2Window} Reference to this.
     */
    _setPos(newPos) {
        // All of the values need to be set because some may have started as "auto"
        for (const s of ['top', 'left', 'bottom', 'right', 'width', 'height']) {
            this.set(s, `${newPos[s]}px`);
        }

        return this;
    }

    /** Return whether the window is displayed or not. */
    get hidden() { return this.window.classed('window-inactive'); }

    /**
     * Change whether the windows is displayed or not.
     * @param {Boolean} hide The new state of the window.
     */
    set hidden(hide) {
        if (!hide) this.bringToFront();
        this.window.classed('window-inactive', hide);
    }

    /**
     * Make the window the highest z-index we know of, and increment that afterwards.
     * @param {Boolean} force Do it even if the current z-index is already highest.
     * @param {Number} [inc = 1] The amount to increase z-index by.
     * @return {N2Window} Reference to this.
     */
    bringToFront(force = false, inc = 1) {
        if (force || this.window.style('z-index') < N2Window.zIndex) {
            N2Window.zIndex += inc;
            this.window.style('z-index', N2Window.zIndex);
        }

        return this;
    }

    /** Make the window visible and return a self-reference */
    show() {
        this.hidden = false;
        return this;
    }

    /** Make the window invisible and return a self-reference */
    hide() {
        this.hidden = true;
        return this;
    }

    /**
     * Set the title if specified or return the current one.
     * @param {String} [newTitle = null] The optional new title.
     * @returns Reference to this if new title set; otherwise String with the current title.
     */
    title(newTitle = null) {
        if (newTitle) {
            this._title.html(newTitle);
            return this;
        }

        return this._title.text();
    }

    /**
     * Change the styling of the window to a preset theme and/or return it.
     * @param {String} [newTheme = null] The name of the theme to change to. No change if null.
     * @returns if newTheme is null, a string with the current theme name;
     *  otherwise a reference to this
     */
    theme(newTheme = null) {
        const contents = this.main.select('.window-contents');
        const classes = contents.attr('class');
        const curTheme = classes.replace(/^.*(window-theme-\S+).*$/, "$1");

        if (newTheme) {
            contents
                .classed(`${curTheme}`, false)
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
                this.window.style(opt, val);
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

    /** Delete the window element from the document and remove the event handler. */
    close() {
        this.closeButton.on('click', null);
        this.modal(false);
        this.window.remove();
    }

    /** Make the close button invisibile and return a reference to this */
    hideCloseButton() {
        this.closeButton.classed('window-inactive', true);
        return this;
    }

    /** Make the close button visibile and return a reference to this */
    showCloseButton() {
        this.closeButton.classed('window-inactive', false);
        return this;
    }

    /** Display the footer ribbon */
    showFooter(footerText = null) {
        this.footer.classed('window-inactive', false);
        this.footer.select('span').text(footerText);
        return this;
    }

    /**
     * Change the color of both the header and footer
     * @param {String} color An HTML-compatible color value
     */
    ribbonColor(color) {
        this.header.style('background-color', color);
        this.footer.style('background-color', color);

        return this;
    }

    move(top, left) {
        let pos = this._getPos();
        pos.left = left;
        pos.top = top;

        pos.right = pos.parentWidth - pos.width - left;
        pos.bottom = pos.parentHeight - pos.height - top;

        this._setPos(pos);

        return this;
    }

    /**
     * Relocate the window to a position near the mouse
     * @param {Object} event The triggering event containing the position.
     * @param {Number} [offset = 15] Distance from mouse to place window.
     */
    moveNearMouse(event, offset = 15) {
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

    /**
     * Since the window is absolutely-positioned with top, left, bottom, right set, we have
     * to manually adjust things if we want the contents to determine the width and height.
     * This should be called anytime content is added and the size is expected to change.
     * TODO: Create a flag and event handler to do this automatically, maybe via
     * MutationObserver.
     * @returns {N2Window} Reference to this.
     */
    sizeToContent() {
        const isHidden = this.hidden, leftPos = this.window.style('left');

        // To size correctly, the window has to be displayed, so if it's
        // hidden, move it offscreen and work there.
        if (isHidden) {
            this.set('left', '-15000px');
            this.show();
        }

        let contentWidth = this.body.node().scrollWidth,
            contentHeight = this.body.node().scrollHeight,
            headerHeight = this.header.node().offsetHeight,
            footerHeight = this.footer.classed('window-inactive') ? 0 : this.footer.node().offsetHeight;

        const totalHeight = contentHeight + headerHeight + footerHeight + 2;

        const newSize = {
            width: contentWidth + 'px',
            height: totalHeight + 'px'
        };
        this.setList(newSize);

        // Put the window back where it belongs if we moved it
        if (isHidden) {
            this.hide();
            this.set('left', leftPos);
        }

        return this;
    }

    /**
     * Make visible the div that separates this window from everything else
     * @param {Boolean} [enable = null] Turn on modal mode if true, off if false.
     * @returns Modal setting if enable is null, otherwise current modal state.
     */
    modal(enable = null) {
        const modalDiv = d3.select('.n2-windows-modal-bg');
        if (enable === null) { return this._enabledModal; }

        if (enable) {
            this.bringToFront(true, 2);
            modalDiv.style('z-index', N2Window.zIndex - 1);
        }

        modalDiv.classed('window-inactive', !enable);
        this._enabledModal = enable;
        return this;
    }
}

/**
 * Extends N2Window by allowing the window to be dragged with a mousedown on the header/title.
 * @typedef N2WindowDraggable
 */
class N2WindowDraggable extends N2Window {
    /** Execute the base class constructor and set up drag event handler */
    constructor(newId = null, cloneId = null) {
        super(newId, cloneId);
        this._setupDrag();
    }

    /** Remove the mousedown event handler and call the superclass close() */
    close() {
        this.header.on('mousedown', null);
        super.close();
    }

    /**
     * Listen for the event to begin dragging the window. The start of the event
     * also brings the window to the front.
     */
    _setupDrag() {
        const self = this;

        this.header
            .classed('window-draggable-header', true)
            .on('mousedown', function () {
                const dragDiv = self.window;

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

/**
 * Extends N2WindowDraggable by setting up 8 divs around the perimeter of the window
 * that change the cursor with mouseover, and allow resizing with mousedown.
 * @typedef N2WindowDraggable
 */
class N2WindowResizable extends N2WindowDraggable {
    constructor(newId = null, cloneId = null, sizeOpts = {}) {
        super(newId, cloneId);

        this.min = {
            width: exists(sizeOpts.minWidth) ? sizeOpts.minWidth : 200,
            height: exists(sizeOpts.minHeight) ? sizeOpts.minHeight : 200
        };

        this.max = {
            width: exists(sizeOpts.maxWidth) ? sizeOpts.maxWidth : window.innerWidth,
            height: exists(sizeOpts.maxHeight) ? sizeOpts.maxHeight : window.innerHeight
        };

        this._setupResizers();
    }

    // Read-only getters
    get minWidth() { return this.min.width; }
    get minHeight() { return this.min.height; }
    get maxWidth() { return this.max.width; }
    get maxHeight() { return this.max.height; }

    // Write-only setters
    set minWidth(val) { this.min.width = val; }
    set minHeight(val) { this.min.height = val; }
    set maxWidth(val) { this.max.width = val; }
    set maxHeight(val) { this.max.height = val; }

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
        const resizerDiv = this.window.select('.main-window')
            .append('div')
            .attr('class', 'resize');

        // For each side, 'mult' refers to whether a coordinate is to be added or subtracted.
        // 'idx' refers to the index of the delta x or y value of the new mouse position.
        // 'dir' is the direction name to check for the min/max size.
        const dirVals = {
            top: { mult: 1, idx: 1, dir: 'height' },
            right: { mult: -1, idx: 0, dir: 'width' },
            bottom: { mult: -1, idx: 1, dir: 'height' },
            left: { mult: 1, idx: 0, dir: 'width' }
        }

        // Set up a mousedown event listener for each of the 8 elements.
        for (const name in resizerClassNames) {
            // Add the div that the resizer mouse event handler will be on
            const resizer = resizerDiv.append('div')
                // Class style settings determine where each div is positioned
                .attr('class', `rsz-${name} rsz-${resizerClassNames[name]}`);

            const dirs = name.split('-'); // From class name, figure out which directions to handle
            resizer.on('mousedown', function () {
                const startDims = self._getPos();

                self.bringToFront();

                const dragStart = [d3.event.pageX, d3.event.pageY];
                let newPos = [0, 0]; // Delta values of the current mouse position vs. start position
                let newSize = {}; // Object to store newly computed positions in

                const w = d3.select(window)
                    .on("mousemove", e => {
                        newPos = [d3.event.pageX - dragStart[0], d3.event.pageY - dragStart[1]];
                        Object.assign(newSize, startDims)

                        for (let i in dirs) { // One iter for straight, two for diag
                            const dv = dirVals[dirs[i]],
                                startPos = startDims[dirs[i]],
                                startSize = startDims[dv.dir];

                            // Calculate the amount the dimension can change to without
                            // violating the set width or height limits of the window.
                            const dimMin = Math.max(0, startPos + startSize - self.min[dv.dir]),
                                dimMax = Math.max(0, startPos + startSize - self.max[dv.dir]);

                            // Calculate the new potential position of the edge from the
                            // original position and the current position of the mouse.
                            const newVal = startPos + (newPos[dv.idx] * dv.mult);

                            // Make sure the edge won't move beyond its limits.
                            if (newVal > startPos) { // Decreasing size (farther from window edge)
                                newSize[dirs[i]] = newVal > dimMin ? dimMin : newVal;
                            }
                            else { // Increasing size (closer to window edge)
                                newSize[dirs[i]] = newVal < dimMax ? dimMax : newVal;
                            }
                        }
                        newSize.width = startDims.parentWidth - (newSize.right + newSize.left);
                        newSize.height = startDims.parentHeight - (newSize.top + newSize.bottom);

                        self._setPos(newSize);
                    })
                    .on("mouseup", e => {
                        w.on("mousemove", null).on("mouseup", null);
                    });

            });
            d3.event.preventDefault();
        }
    }
}

function wintest() {
    const content = '<p style="margin: 10px; padding: 10px; border: 25px solid red; ">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed justo mauris, porttitor sed nibh non, interdum aliquet tellus. Duis eget est lectus. In ultrices finibus semper. Nullam dictum, tortor non placerat convallis, nibh diam sagittis risus, non ultricies diam nisi nec neque. Phasellus dapibus convallis metus. Proin cursus, metus quis ullamcorper suscipit, neque mauris dictum ex, a mattis velit lorem ac erat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Praesent a ligula ut arcu rutrum venenatis. Morbi nec sapien turpis. Nunc tincidunt maximus venenatis. Phasellus facilisis imperdiet velit, nec cursus elit tincidunt pretium. Duis ligula metus, rutrum nec ullamcorper a, pretium eu massa. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam condimentum, urna in congue dignissim, mi risus maximus lectus, interdum cursus neque turpis sed libero. Cras iaculis ornare accumsan. Sed tempor pretium est, eget aliquam purus feugiat ac.</p>';
    myWin = new N2Window();
    myWin.setList({ width: '300px', height: '300px', title: 'This is a very, very, very, very long title indeed', top: '100px', left: '10px' });
    myWin.show();
    myWin.body.html(content);
    myWin.sizeToContent();

    myWin2 = new N2WindowDraggable();
    myWin2.setList({ width: '300px', height: '300px', title: 'Draggable Window', top: '100px', left: '350px' });
    myWin2.show();
    myWin2.body.html(content);
    myWin2.sizeToContent();

    myWin3 = new N2WindowResizable(null, null, { maxWidth: 500, maxHeight: 1000 });
    myWin3.setList({ width: '300px', height: '300px', title: 'Resizable Window', top: '100px', left: '700px' });
    myWin3.showFooter();
    myWin3.show();
    myWin3.body.html(content);
    myWin3.sizeToContent();

}