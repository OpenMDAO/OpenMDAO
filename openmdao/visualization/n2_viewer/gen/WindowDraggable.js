// <<hpp_insert gen/Window.js>>

/**
 * Extends Window by allowing the window to be dragged with a mousedown on the header/title.
 * @typedef WindowDraggable
 */
class WindowDraggable extends Window {
    /** Execute the base class constructor and set up drag event handler */
    constructor(newId = null, cloneId = null) {
        super(newId, cloneId);
        this.boundaryMargin = 10;
        this._setupDrag();
    }

    /** Remove the mousedown event handler and call the superclass close() */
    close(e) {
        this.header.on('mousedown', null);
        super.close(e);
    }

    _doDragHeader(e) { this._doDrag(e, '.window-header'); }
    _doDragFooter(e) { this._doDrag(e, '.window-footer'); }

    /** Prevent the window from being dragged off the left or right of the browser. */
    _applyHorizontalBounds(testX) {
        return (testX < this.boundaryMargin ?
            this.boundaryMargin : 
                    (testX > window.innerWidth - this.boundaryMargin?
                        window.innerWidth - this.boundaryMargin : testX));
    }

    /** Prevent the window from being dragged off the top or bottom of the browser. */
    _applyVerticalBounds(testY) {
        return (testY < this.boundaryMargin ?
            this.boundaryMargin : 
                    (testY > window.innerHeight - this.boundaryMargin?
                        window.innerHeight - this.boundaryMargin : testY));
    }

    /** 
     * Perform the dragging operation on either header or footer. The start of the event
     * also brings the window to the front.
     */
    _doDrag(e, ribbonClass) {
        const self = this;
        const dragDiv = self.window;

        self.bringToFront();
        dragDiv.style('cursor', 'grabbing')
            .select(ribbonClass).style('cursor', 'grabbing');

        const dragStart = [e.pageX, e.pageY];
        let newTrans = [0, 0];

        const w = d3.select(window)
            .on("mousemove", e => {
                const x = self._applyHorizontalBounds(e.pageX),
                    y = self._applyVerticalBounds(e.pageY);
                newTrans = [x - dragStart[0], y - dragStart[1]];
                dragDiv.style('transform', `translate(${newTrans[0]}px, ${newTrans[1]}px)`)
            })
            .on("mouseup", () => {
                // Convert the translate to style position
                self._setPos(self._getPos());

                dragDiv.style('cursor', 'auto')
                    .style('transform', null)
                    .select(ribbonClass)
                    .style('cursor', 'grab');

                // Remove event listeners
                w.on("mousemove", null).on("mouseup", null);
            });

        e.preventDefault();
    }

    /** Listen for the mousedown event to begin dragging the window. */
    _setupDrag() {
        const self = this;

        this.header
            .classed('window-draggable-ribbon', true)
            .on('mousedown', self._doDragHeader.bind(self));

        this.footer
            .classed('window-draggable-ribbon', true)
            .on('mousedown', self._doDragFooter.bind(self));
    }
}
