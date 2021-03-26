var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function analyze() { 
  console.log("Hei Elise <3")
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("analyze-button").innerHTML = "Analyzing...";


  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze_cnn`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      // var response = {"result":[{'class': 'PNEUMONIA', 'output': 0.9, 'prob': 0.93}, {'class': 'NORMAL', 'output': 0.1, 'prob': 0.07}]}
      var result = response["result"]
      el("result-label").innerHTML = `Result = ${result}`;
      var modal = document.getElementById("myModal");
      modal.style.display = "block";
      el("state").innerHTML = `${result[0]}`;
      // state.innerHTML = response["result"][0].class
      el("percent").innerHTML = result[1];
    }
    el("analyze-button").innerHTML = "Analyze";
  };


  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}
// Top navigation bar
$(function(){
  var str = '#len'; //increment by 1 up to 1-nelemnts
  $(document).ready(function(){
    var i, stop;
    i = 1;
    stop = 4; //num elements
    setInterval(function(){
      if (i > stop){
        return;
      }
      $('#len'+(i++)).toggleClass('bounce');
    }, 500)
  });
});

// Modal for results

// When the user clicks the button, open the modal 
// External modal-button for testing locally
// function openModal(){
  // var response = JSON.parse(e.target.responseText);
  // var response = {"result":[{'class': 'PNEUMONIA', 'output': 0.9, 'prob': 0.93}, {'class': 'NORMAL', 'output': 0.1, 'prob': 0.07}]}
  // var response = {"result":[{'class': 'NORMAL', 'output': 1.0, 'prob': 1.0}, {'class': 'PNEUMONIA', 'output': 0.0, 'prob': 0.0}]}
  // el("result-label").innerHTML = `Result = ${response["result"]}`;
  // var modal = document.getElementById("myModal");
  // modal.style.display = "block";
  // el("state").innerHTML = response["result"][0].class
  // state.innerHTML = response["result"][0].class
  // el("percent").innerHTML = response["result"][0].prob * 100
// }


// When the user clicks on <span> (x), close the modal
function closeModal(){
  var modal = document.getElementById("myModal");
  modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it

window.onclick = function(event) {
  var modal = document.getElementById("myModal");
  if (event.target === modal) {
    modal.style.display = "none";
  }
}