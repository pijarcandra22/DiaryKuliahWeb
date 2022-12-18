$(document).ready(function(){
  $('.topic-child').dblclick(function() {
    index = $(this).index()
    positif = $(".topic-child:eq("+index+") .positif").hasClass("d-none d-md-block");
    negatif = $(".topic-child:eq("+index+") .negatif").hasClass("d-none d-md-block");
    netral = $(".topic-child:eq("+index+") .netral").hasClass("d-none d-md-block");
    alert("Data "+positif+negatif+netral)

    if(!positif){
      $(".topic-child:eq("+index+") .positif").addClass("d-none d-md-block");
      $(".topic-child:eq("+index+") .negatif").removeClass("d-none d-md-block");
    }else if(!negatif){
      $(".topic-child:eq("+index+") .negatif").addClass("d-none d-md-block");
      $(".topic-child:eq("+index+") .netral").removeClass("d-none d-md-block");
    }else if(!netral){
      $(".topic-child:eq("+index+") .netral").addClass("d-none d-md-block");
      $(".topic-child:eq("+index+") .positif").removeClass("d-none d-md-block");
    }
  });
});

function AlertDblclick(id){
  alert(id)
  id.style.display = "none"
}