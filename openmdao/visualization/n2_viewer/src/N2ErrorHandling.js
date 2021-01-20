window.addEventListener("error", handleError, true);
function handleError(evt) {

    // Cannot define this color in N2Style since hasn't been loaded yet
    // Cannot call anything in any of the N2 JavaScript code because nothing has been loaded
    document.body.style.background = '#db9f9f';
    document.getElementById("show-error-button-container").style.visibility = 'visible';

    document.getElementById('show-error-button').onclick = function() {
       if (evt.error) { // Chrome sometimes provides this
         alert("An error has occured while displaying the N2 diagram. This error should be reported as an " +
               "issue to the OpenMDAO developers.\n\n" +
               "Error: " + evt.message + "\n\n" +
               "Stack trace: \n" + evt.error.stack + "\n" );
       } else {
         alert("An error has occured while displaying the N2 diagram. This error should be reported as an issue " +
               "to the OpenMDAO developers. \n\n" +
               "For more details about the error, please look in the browser's Console using the menu item:\n\n" +
               "    Develop -> Show JavaScript Console");
       }
    }
}
