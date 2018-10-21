const MULTIPLIER = 9;
const POINT_MULTIPLIER = MULTIPLIER -1;
const POINT_X_OFFSET = 25;
const POINT_Y_OFFSET = 30;

const X_MIN = -62.5 * MULTIPLIER;
const X_MAX = 53.6 * MULTIPLIER;
const Y_MIN = -67.5 * MULTIPLIER;
const Y_MAX = 14.325 * MULTIPLIER;

const VIEW_POINT_DIVIDER_X = 2;
const VIEW_POINT_DIVIDER_Y = 2;

var width = Math.abs(X_MIN) + Math.abs(X_MAX);
var height = Math.abs(Y_MIN) + Math.abs(Y_MAX);

var svg = d3.select("#svgcontainer")
   .append("svg")
   .attr("width", width)
   .attr("height", height)
   .attr("viewBox", `${-width/VIEW_POINT_DIVIDER_X} ${-height/VIEW_POINT_DIVIDER_Y} ${width} ${height}`)
   .style("border", "1px solid black")
   .style("stroke-width", 2);

showAreas();
loadItems();

function showAreas() {
  AREAS.forEach(area => {
    try {
        let shape = JSON.parse(area.shape);
        let points = "";
        shape.coordinates.forEach(points => {
  
        points = points.map(point => [(point[0]+(POINT_X_OFFSET))*POINT_MULTIPLIER , (point[1]+(POINT_Y_OFFSET))*POINT_MULTIPLIER]);
  
        svg.append("polygon")
          .attr("points", points)
          .style("fill", "none")
          .style("stroke", "black")
          .style("strokeWidth", "10px");
      });
    } catch(e) {
      console.warn("JSON is not valid:", e)
    }
  });
}

function showItems(items) {
  items = JSON.parse(items);
  let points = [];
  items.forEach(item => {
    points.push([  (parseInt(item.x, 10) + POINT_X_OFFSET)*POINT_MULTIPLIER, (parseInt(item.y, 10) + POINT_Y_OFFSET)*POINT_MULTIPLIER])
  });

  console.log("item.x+(100000))*POINT_MULTIPLIER", parseInt(items[0].x, 10));
  console.log("item.x+(100000))*POINT_MULTIPLIER", parseInt(items[0].y, 10));

  svg.append("polygon")
      .attr("points", points)
      .style("fill", "none")
      .style("stroke", "red")
      .style("strokeWidth", "10px");
}

function loadItems() {
  const xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = function() {
    if (this.readyState == 4 && this.status == 200) {
     console.log("response...");
     // console.log(this.responseText);
     showItems(this.responseText);
    }
  };
  xhttp.open("GET", "http://localhost:8080/api/items", true);
  xhttp.send();
}

/*svg.append("line")
   .attr("x1", -width/VIEW_POINT_DIVIDER_X)
   .attr("y1", 0)
   .attr("x2", width/VIEW_POINT_DIVIDER_X)
   .attr("y2", 0)  
   .style("stroke", "rgb(255,0,0)");


svg.append("line")
.attr("x1", 0)
.attr("y1", -height/VIEW_POINT_DIVIDER_Y)
.attr("x2", 0)
.attr("y2", height/VIEW_POINT_DIVIDER_Y)  
.style("stroke", "rgb(255,0,0)");*/