/**
 * Manage all components of the application. The model data, the CSS styles, the
 * user interface, the layout of the N2 matrix, and the N2 matrix grid itself are
 * all member objects.
 * @typedef Toolbar
 */

window.onload = function() {
    let windowHeight = window.innerHeight;
    let windowWidth = window.innerWidth;


    const modelHeightSlider = document.getElementById("model-slider");
    modelHeightSlider.value = windowHeight * .95;

    /** If a tab bar button has multiple nested button expand the view out  **/
    const expandableViews = document.getElementsByClassName('expandable');
    let toolbar = document.getElementById("toolbarLoc");
    for (let i = 0; i < expandableViews.length; i++) {
        let expandableView = expandableViews[i];
        const container = expandableView.querySelector('div');

        expandableView.addEventListener('mouseover', e => {
            toolbar.style.zIndex = '5';
            setMaxWidth(container, '200px');
        });

        expandableView.addEventListener('mouseout', e => {
            setMaxWidth(container, '0');
            toolbar.style.zIndex = '1';
        });
    }

    function setMaxWidth(node, size) {
        node.style.maxWidth = size;
    }

    /************ Handle button clicks and setting the active button ************/
    const expandableButtonsContainer = document.getElementsByClassName(
        'toolbar-group-expandable'
    );

    // Loop through all toolbar groups which have nested buttons
    for (let i = 0; i < expandableButtonsContainer.length; i++) {
        const container = expandableButtonsContainer[i];

        // Get all nested elements which are both buttons and text
        const expandableButtons = expandableButtonsContainer[i].childNodes;

        // Loop through the child nodes to only select the buttons which are every odd index
        for (let x = 1; x < expandableButtons.length; x += 2) {
            const button = expandableButtons.item(x);

            // Only add this functioniality to buttons not to input fields which are the sliders
            if (button.tagName !== 'INPUT') {
                button.addEventListener('click', () => {
                    const currentVisibleIcon =
                        container.previousElementSibling.previousElementSibling;
                    // Set the visible icon in the toolbar to the clicked icon
                    currentVisibleIcon.className = button.className;
                    currentVisibleIcon.id = button.id;

                    // Change the functionality of the new visible icon
                    currentVisibleIcon.onclick = button.onclick;
                });
            }
        }
    }

    /************ Toggle Hiding the Toolbar ************/
    const hideToolbarButton = document.getElementById('hide-toolbar');
    const hideToolbarIcon = hideToolbarButton.childNodes[1];

    const matrix = document.getElementById('d3_content_div');

    hideToolbarButton.addEventListener('click', () => {
        const currentToolbarPosition = toolbar.style.left;

        // Check if the toolbar is already hidden, if it is then show it
        if (currentToolbarPosition === '-75px') {
            hideToolbarIcon.style.transform = 'rotate(0deg)';
            toolbar.style.left = '0px';
            hideToolbarButton.style.left = '45px';
            matrix.style.marginLeft = '0px';
        } else {
            toolbar.style.left = '-75px';
            hideToolbarButton.style.left = '-30px';
            hideToolbarIcon.style.transform = 'rotate(-180deg)';
            matrix.style.marginLeft = '-75px';
        }
    });

    /************ Expand & Contract the Searchbar ************/
    const searchbar = document.getElementById('awesompleteId');
    const searchbarDiv = document.getElementById('searchbar-container');

    searchbarDiv.addEventListener('mouseover', e => {
        searchbar.style.width = '200px';
        toolbar.style.zIndex = '5';
    });

    searchbarDiv.addEventListener('mouseout', e => {
        searchbar.style.width = '0px';
        toolbar.style.zIndex = '1';
    });


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
}