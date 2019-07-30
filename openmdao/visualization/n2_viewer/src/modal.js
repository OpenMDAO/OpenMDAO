///////////////////////////
//Modal Help Dialog Stuff
///////////////////////////
var parentDiv = document.getElementById("ptN2ContentDivId");
var modal = parentDiv.querySelector("#myModal");

// When the user clicks the button, open the modal
function DisplayModal() {
    modal.style.display = "block";
}

// When the user clicks on <span> (x), close the modal
parentDiv.querySelector("#idSpanModalClose").onclick = function () {
    modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function (event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}