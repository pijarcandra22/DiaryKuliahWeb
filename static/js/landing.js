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
  target = "#"+id+"-tweet"
  $(target+" .dbclick-alert").css({'display':'none'})
}

function TopicOpen(column,id,max,no){
  id  = parseInt(id)
  max = parseInt(max)
  target = "#"+column+"-tweet"
  target2 = "#"+column+"-"+no+"-identity"
  $('html, body').animate({
    scrollTop: $("#"+column+"-top").offset().top
  }, 10);

  $("#topic_place").find(target);

  $(target).removeClass('d-none')
  $(target).removeClass('d-none')
  $(target+" .topic-tittle").html("Topik #"+no)
  $(target+" .topic-key").html($(target2+" .topic-persentase").html())
  for(i=0;i<max;i++){
    if(i==id){
      $(target+" .positif-"+i+","+target+" .netral-"+i+","+target+" .negatif-"+i).css({'display':'block'})
    }else{
      $(target+" .positif-"+i+","+target+" .netral-"+i+","+target+" .negatif-"+i).css({'display':'none'})
    }
  }
}

function preprocessing(topik){
  $("#preprocessing-"+topik).css({'background-color':'#B01E1E','color':'#fff'})
  $("#preprocessing-"+topik).html(
    '<span class="spinner-grow spinner-grow-sm" role="status" aria-hidden="true"></span>'+
    " Preprocessing"
  )
  $.ajax({
    url: '/preprocessing-topic/'+topik,
    type: 'GET',
    success: function(response){
      alert("Preprocessing Sukses")
      window.location.reload();
    },
    error: function(error){
      alert("Terjadi masalah dalam proses preprocessing, harap diulangi lagi")
      window.location.reload();
    }
  });
}

function emotion(topik){
  $("#emotion-"+topik).css({'background-color':'#B01E1E','color':'#fff'})
  $("#emotion-"+topik).html(
    '<span class="spinner-grow spinner-grow-sm" role="status" aria-hidden="true"></span>'+
    " Analisis Sentimen"
  )
  $.ajax({
    url: '/emotion-topic/'+topik,
    type: 'GET',
    success: function(response){
      alert("Analisis Sentimen Sukses")
      window.location.reload();
    },
    error: function(error){
      alert("Terjadi masalah dalam proses analisis sentimen, harap diulangi lagi")
      window.location.reload();
    }
  });
}

function klasterisasi(topik){
  $("#klasterisasi-"+topik).css({'background-color':'#B01E1E','color':'#fff'})
  $("#klasterisasi-"+topik).html(
    '<span class="spinner-grow spinner-grow-sm" role="status" aria-hidden="true"></span>'+
    " Kategorisasi"
  )
  $.ajax({
    url: '/klasterisasi-topic/'+topik,
    type: 'GET',
    success: function(response){
      alert("Klasterisasi Sukses")
      window.location.reload();
    },
    error: function(error){
      alert("Terjadi masalah dalam proses klasterisasi, harap diulangi lagi")
      window.location.reload();
    }
  });
}

function upSeeTopic(topik,pre,emo,mod){
  whatMustOpen = 0
  if(pre!="True"){whatMustOpen+=1}
  if(emo!="True"){whatMustOpen+=1}
  if(mod!="True"){whatMustOpen+=1}
  $('#topic_place').html("")
  var divElem = $('<div/>').load('/topik/'+topik+"/"+whatMustOpen).addClass("topic-child");
  divElem.appendTo('#topic_place');
  $.ajax({
    url: '/upAccess-topic/'+topik,
    type: 'GET',
    success: function(response){},
    error: function(error){}
  });
}