function parse_data(selected, target, all, labels){
    //window.alert("I am here!") 
    var legendTemplate= "<ul class=\"<%=name.toLowerCase()%>-legend\"><% for (var i=0; i<datasets.length; i++){%><li><span style=\"background-color:<%=datasets[i].fillColor%>\"></span><%if(datasets[i].label){%><%=datasets[i].label%><%}%></li><%}%></ul>"
    var datasets=[{
             label: "Selected",
             fillColor: "rgba(220,220,220,0.5)",
             strokeColor: "rgba(220,220,220,0.8)",
             highlightFill: "rgba(220,220,220,0.75)",
             highlightStroke: "rgba(220,220,220,1)",
             legendTemplate: legendTemplate,
             data: selected},
             {label: "Target",
             fillColor: "rgba(151,187,205,0.5)",
             strokeColor: "rgba(151,187,205,0.8)",
             highlightFill: "rgba(151,187,205,0.75)",
             highlightStroke: "rgba(151,187,205,1)",
             legendTemplate: legendTemplate,
             data: target}, 
             {label: "All",
             fillColor: "rgba(255,174,81,0.5)",
             strokeColor: "rgba(255,174,81,0.8)",
             highlightFill: "rgba(255,174,81,0.75)",
             highlightStroke: "rgba(255,174,81,1)",
             legendTemplate: legendTemplate,
             data: all}]

     var data = {
         labels: labels,
         datasets: datasets
         };
      return data
}

function make_barChart(data) {
     var myBarChart = new Chart(document.getElementById("BarChartLoc").getContext("2d")).Bar(data);
 return myBarChart}


function updateBarChart(myBarChart, selected, target){
    var bla = 0;
    for (i=0; i < selected.length; i++){
        myBarChart.datasets[0].bars[i].value=selected[i];
        }
    for (i=0; i < target.length; i++){
        myBarChart.datasets[1].bars[i].value=target[i];        
        }
    return myBarChart;
}


