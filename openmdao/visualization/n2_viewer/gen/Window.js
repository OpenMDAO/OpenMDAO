/**
 * Create a clone of the #window-template defined in index.html and provide
 * management functions such as setting size, position, ribbon color, etc.
 * @typedef Window
 */
class Window {
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

        if (!Window.container) {
            Window.container = d3.select('#n2-windows');
        }

        this._main = this._window.select('.main-window'); // Not referenced very often
        this._header = this._window.select('.window-header');
        this._title = this._header.select('.window-title');
        this._closeButton = this._window.select('.window-close-button');
        this._body = this._window.select('.window-body');
        this._footer = this._window.select('.window-footer');

        this._enabledModal = false;

        const self = this;
        this._closeButton.on('click', e => self.close(e))

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
     * @param {Object} [container = Window.container.node()] HTML element containing window.
     * @returns {Object} Each key represents the position in pixels.
     */
    _getPos(container = Window.container.node()) {
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
            x: childPos.x - parentPos.x,
            y: childPos.y - parentPos.y,
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
     * @returns {Window} Reference to this.
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
     * @returns {Window} Reference to this.
     */
    bringToFront(force = false, inc = 1) {
        if (force || this.window.style('z-index') < Window.zIndex) {
            Window.zIndex += inc;
            this.window.style('z-index', Window.zIndex);
        }

        return this;
    }

    /**
     * Make the window visible.
     * @returns {Window} Reference to this.
     */
    show() {
        this.hidden = false;
        return this;
    }

    /**
     * Make the window invisible.
     * @returns {Window} Reference to this.
     */
    hide() {
        this.hidden = true;
        return this;
    }

    /**
     * If window is visible, hide it; if it's hidden, show it.
     * @returns {Window} Reference to this.
     */
    toggle() {
        if (this.hidden) this.show();
        else this.hide();
        return this;
    }

    /**
     * Set the title if specified or return the current one.
     * @param {String} [newTitle = null] The optional new title.
     * @returns Reference to this if new title set; otherwise String with the current title.
     */
    title(newTitle = null) {
        if (newTitle !== null) {
            this._title.html(newTitle);
            return this;
        }

        return this._title.text();
    }

    /**
     * Change the styling of the window to a preset theme and/or return it.
     * @param {String} [newTheme = null] The name of the theme to change to. No change if null.
     * @returns If newTheme is null, a string with the current theme name;
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
     * @returns {Window} Reference to this.
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
     * @returns {Window} Reference to this.
     */
    setList(options) {
        for (const optName in options) {
            this.set(optName, options[optName]);
        }

        return this;
    }

    /** Delete the window element from the document and remove the event handler. */
    close(e) {
        this.closeButton.on('click', null);
        this.modal(false);
        this.window.remove();
    }

    /**
     * Make the close button invisibile.
     * @returns {Window} Reference to this.
     */
    hideCloseButton() {
        this.closeButton.classed('window-inactive', true);
        return this;
    }

    /**
     * Make the close button visibile.
     * @returns {Window} Reference to this.
     */
    showCloseButton() {
        this.closeButton.classed('window-inactive', false);
        return this;
    }

    /** Set the text in the footer ribbon.
     * @param {String} [footerText = null] If not empty, set the text in the footer.
     * @returns {Window} Reference to this.
     */
    footerText(newText = null) {
        this.footer.select('span').text(newText);
        return this;
    }

    /** Determine whether the footer is visible in this theme. */
    hasFooter() {
        return visible(this.footer.node());
    }

    /**
     * Change the color of both the header and footer
     * @param {String} color An HTML-compatible color value
     * @returns {Window} Reference to this.
     */
    ribbonColor(color) {
        this.header.style('background-color', color);
        this.footer.style('background-color', color);

        return this;
    }

    /**
     * Move the window to the specified coordinates.
     * @param {Number} x If positive, set the left property to this and adjust
     *   the right. If negative, set the right property to this and adjust left.
     * @param {Number} y If positive, set the top property to this and adjust
     *   the bottom. If negative, set the bottom property to this and adjust top.
     * @returns {Window} Reference to this.
     */
    move(x, y) {
        let pos = this._getPos(d3.select('body').node());
        if (x >= 0 ) {
            pos.left = x;
            pos.right = pos.parentWidth - pos.width - pos.left;
        }
        else {
            pos.right = -x;
            pos.left = pos.parentWidth - pos.width - pos.right;
        }

        if (y >= 0) {
            pos.top = y;
            pos.bottom = pos.parentHeight - pos.height - pos.top;
        }
        else {
            pos.bottom = -y;
            pos.top = pos.parentHeight - pos.height - pos.bottom;
        }

        this._setPos(pos);

        return this;
    }

    /**
     * Relocate the window to a position near the mouse
     * @param {Object} event The triggering event containing the position.
     * @param {Number} [offset = 15] Distance from mouse to place window.
     * @returns {Window} Reference to this.
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
     * @returns {Window} Reference to this.
     */
    sizeToContent(extraWidth = 0, extraHeight = 2, considerTitle = false) {
        const bodyWidth = this.body.node().scrollWidth;
        const titleWidth = this._title.node().scrollWidth + 60;
        const contentWidth = ((!considerTitle || bodyWidth > titleWidth)? bodyWidth : titleWidth) + extraWidth;
        const contentHeight = this.body.node().scrollHeight;
        const headerHeight = this.header.node().offsetHeight
        const footerHeight = this.hasFooter() ? this.footer.node().offsetHeight : 0;

        const totalHeight = contentHeight + headerHeight + footerHeight + extraHeight;

        const newSize = {
            width: contentWidth + 'px',
            height: totalHeight + 'px'
        };
        this.setList(newSize);

        return this;
    }

    /**
     * Make visible the div that separates this window from everything else
     * @param {Boolean} [enable = null] Turn on modal mode if true, off if false.
     * @param {String} [style = null] Change the default appearance of translucent black.
     * @returns Modal setting if enable is null, otherwise current modal state.
     */
    modal(enable = null, style = null) {
        const modalDiv = d3.select('.n2-windows-modal-bg');
        if (enable === null) { return this._enabledModal; }

        if (enable) {
            this.bringToFront(true, 2);
            modalDiv.attr('style', style)
                .style('z-index', Window.zIndex - 1)
                
        }
        else {
            modalDiv.attr('style', null);
        }

        modalDiv.classed('window-inactive', !enable);
        this._enabledModal = enable;
        return this;
    }

    /**
     * Take ownership of a predefined DOM object, usually a div, by appending
     * its children to the window body. This allows the definition of more
     * complex window contents to be located in index.html for example.
     * @param {String} id HTML id of the object to copy.
     */
    absorbBody(id) {
      const newParent = this.body.node();
      const oldParent = d3.select(id).node();

        while (oldParent.childNodes.length > 0) {
            newParent.appendChild(oldParent.childNodes[0]);
        }

        oldParent.remove();
    }

    /**
     * Copy the structure of a predefined DOM object, usually a div, by cloning
     * its children to the window body. This allows the definition of more
     * complex window contents to be located in index.html for example.
     * @param {String} id HTML id of the object to copy.
     */
     copyBody(id) {
        const dstParent = this.body.node();
        const srcParent = d3.select(id).node();
  
          for (const child of srcParent.childNodes) {
              dstParent.appendChild(child.cloneNode(true));
          }
      }
}


/**
 * Keep increasing z-index of the focused window to keep it on top.
 * Max z-index is 2147483647. It will be unusual here for it to climb
 * above 100, and even extreme cases (e.g. a diagram that's been in use
 * for weeks with lots of windows) shouldn't get above a few thousand.
 */
Window.zIndex = 1000;
Window.container = null;
