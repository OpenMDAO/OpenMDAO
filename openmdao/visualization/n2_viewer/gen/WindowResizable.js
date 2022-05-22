// <<hpp_insert gen/WindowDraggable.js>>

/**
 * Extends WindowDraggable by setting up 8 divs around the perimeter of the window
 * that change the cursor with mouseover, and allow resizing with mousedown.
 * @typedef WindowResizable
 */
class WindowResizable extends WindowDraggable {
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
     * Set the cursor for all elements to the specified value, for the purpose
     * of resizing the window and not having the cursor change every time another
     * element is hovered.
     * @param {String} cursor The new value of the cursor.
     */
    _lockCursor(cursor) {
        this.header.style('cursor', cursor);
        this.window.style('cursor', cursor);
        this.window.select('.window-close-button').style('cursor', cursor);
        this.resizerDiv.selectAll('div').style('cursor', cursor);        

        return this;
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
        this.resizerDiv = this.window.select('.main-window')
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
            const resizer = this.resizerDiv.append('div')
                // Class style settings determine where each div is positioned
                .attr('class', `rsz-${name} rsz-${resizerClassNames[name]}`);

            const dirs = name.split('-'); // From class name, figure out which directions to handle
            resizer.on('mousedown', e => {
                const startDims = self._getPos();
                self.bringToFront();

                const cursor = resizer.style('cursor');
                self.modal(true, `background-color: none; opacity: 0; cursor: ${cursor};`)
                    ._lockCursor(cursor);

                const dragStart = [e.pageX, e.pageY];
                let newPos = [0, 0]; // Delta values of the current mouse position vs. start position
                let newSize = {}; // Object to store newly computed positions in

                const w = d3.select(window)
                    .on("mousemove", e => {
                        e.stopPropagation();
                        e.preventDefault();

                        newPos = [e.pageX - dragStart[0], e.pageY - dragStart[1]];
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
                    .on("mouseup", () => {
                        w.on("mousemove", null).on("mouseup", null);
                        self.modal(false);
                        self._lockCursor(null);
                    });

            });
        }
    }
}

function wintest() {
    const content = '<p style="margin: 10px; padding: 10px; border: 25px solid red; ">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed justo mauris, porttitor sed nibh non, interdum aliquet tellus. Duis eget est lectus. In ultrices finibus semper. Nullam dictum, tortor non placerat convallis, nibh diam sagittis risus, non ultricies diam nisi nec neque. Phasellus dapibus convallis metus. Proin cursus, metus quis ullamcorper suscipit, neque mauris dictum ex, a mattis velit lorem ac erat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Praesent a ligula ut arcu rutrum venenatis. Morbi nec sapien turpis. Nunc tincidunt maximus venenatis. Phasellus facilisis imperdiet velit, nec cursus elit tincidunt pretium. Duis ligula metus, rutrum nec ullamcorper a, pretium eu massa. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam condimentum, urna in congue dignissim, mi risus maximus lectus, interdum cursus neque turpis sed libero. Cras iaculis ornare accumsan. Sed tempor pretium est, eget aliquam purus feugiat ac.</p>';
    myWin = new Window();
    myWin.setList({ width: '300px', height: '300px', title: 'This is a very, very, very, very long title indeed', top: '100px', left: '10px' });
    myWin.show();
    myWin.body.html(content);
    myWin.sizeToContent();

    myWin2 = new WindowDraggable();
    myWin2.setList({ width: '300px', height: '300px', title: 'Draggable Window', top: '100px', left: '350px' });
    myWin2.show();
    myWin2.body.html(content);
    myWin2.sizeToContent();

    myWin3 = new WindowResizable(null, null, { maxWidth: 500, maxHeight: 1000 });
    myWin3.theme('value-info');
    myWin3.setList({ width: '300px', height: '300px', title: 'Resizable Window', top: '100px', left: '700px' });
    myWin3.show();
    myWin3.body.html(content);
    myWin3.sizeToContent();

}
