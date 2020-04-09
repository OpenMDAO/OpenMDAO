/**
 * Manage all components of the application. The model data, the CSS styles, the
 * user interface, the layout of the N2 matrix, and the N2 matrix grid itself are
 * all member objects.
 * @typedef Toolbar
 */

function subsToolbar() {
    let windowHeight = window.innerHeight;
    let windowWidth = window.innerWidth;


    // const modelHeightSlider = document.getElementById("model-slider");
    // modelHeightSlider.value = windowHeight * .95;
    let toolbar = document.getElementById("toolbarLoc");

    /************ Make the legend draggable ************/
    const legend = document.getElementById('legend-div');
    dragElement(legend);

    function dragElement(elmnt) {
        var pos1 = 0,
            pos2 = 0,
            pos3 = 0,
            pos4 = 0;
        if (document.getElementById(elmnt.id + 'header')) {
            // if present, the header is where you move the DIV from:
            document.getElementById(elmnt.id + 'header').onmousedown = dragMouseDown;
        } else {
            // otherwise, move the DIV from anywhere inside the DIV:
            elmnt.onmousedown = dragMouseDown;
        }

        function dragMouseDown(e) {
            e = e || window.event;
            e.preventDefault();
            // get the mouse cursor position at startup:
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            // call a function whenever the cursor moves:
            document.onmousemove = elementDrag;
        }

        function elementDrag(e) {
            e = e || window.event;
            e.preventDefault();
            // calculate the new cursor position:
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;

            // set the element's new position:
            let topOffset = elmnt.offsetTop - pos2;
            if (topOffset < 0) {
                topOffset = 0;
            } else if (topOffset > windowHeight - 40) {
                topOffset = windowHeight - 40;
            }

            let leftOffset = elmnt.offsetLeft - pos1;
            if (leftOffset < 0) {
                leftOffset = 0;
            } else if (leftOffset > windowWidth - 300) {
                leftOffset = windowWidth - 300;
            }

            elmnt.style.top = topOffset + 'px';
            elmnt.style.left = leftOffset + 'px';
        }

        function closeDragElement() {
            // stop moving when mouse button is released:
            document.onmouseup = null;
            document.onmousemove = null;
        }
    }

    if (_EMBEDDED) { // Hide toolbar/legend if embedded
        toolbar.style.left = '-75px';
        hideToolbarButton.style.left = '-30px';
        hideToolbarIcon.style.transform = 'rotate(-180deg)';
        matrix.style.marginLeft = '-75px';

        d3.select("#legend-div").style('display', 'none');
    }
}