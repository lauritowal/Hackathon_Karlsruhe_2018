const MULTIPLIER = 9;
const POINT_MULTIPLIER = MULTIPLIER -1;
const POINT_X_OFFSET = 25;
const POINT_Y_OFFSET = 30;

const X_MIN = -62.5 * MULTIPLIER;
const X_MAX = 53.6 * MULTIPLIER;
const Y_MIN = -67.5 * MULTIPLIER;
const Y_MAX = 14.325 * MULTIPLIER;

const VIEW_POINT_DIVIDER_X = 2;//1.5;
const VIEW_POINT_DIVIDER_Y = 2;//1.25;

var width = Math.abs(X_MIN) + Math.abs(X_MAX);
var height = Math.abs(Y_MIN) + Math.abs(Y_MAX);

var svg = d3.select("#svgcontainer")
   .append("svg")
   .attr("width", width)
   .attr("height", height)
   .attr("viewBox", `${-width/VIEW_POINT_DIVIDER_X} ${-height/VIEW_POINT_DIVIDER_Y} ${width} ${height}`)
   .style("border", "1px solid black")
   .style("stroke-width", 2);


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