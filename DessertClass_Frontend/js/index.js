

function readURL(input) {
  
    var reader = new FileReader();

	var $SCRIPT_ROOT = "http://127.0.0.1:8648/predict";

	reader.readAsDataURL(input.files[0]);

	reader.onload = function (e) {
		$("#imageResult").attr("src", e.target.result);
	};
	reader.onerror = function (e) {
		console.log('Error', e);
	};
	
	reader.onloadend = function (e) {
		var file = e.target;
		var result = file.result;
		
		$.ajax({
			type: "POST",
			url: $SCRIPT_ROOT,
			data: JSON.stringify(result),

			success: function (data) {
				var response = data.response;
				console.log(response);
			},
			error: function (_req, err) {
				console.log(err);
			},
			contentType: "application/json",
			dataType: "json",
		});
	
	};
}

$(function () {
  $("#upload").on("change", function () {
    readURL(input);
  });
});


/*  ==========================================
    SHOW UPLOADED IMAGE NAME
* ========================================== */
var input = document.getElementById( 'upload' );
var infoArea = document.getElementById( 'upload-label' );

input.addEventListener( 'change', showFileName );
function showFileName( event ) {
  var input = event.srcElement;
  var fileName = input.files[0].name;
  infoArea.textContent = 'File name: ' + fileName;
}

