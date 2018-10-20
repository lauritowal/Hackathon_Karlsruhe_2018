const MULTIPLIER = 100;

const X_MIN = -62.5 * MULTIPLIER;
const X_MAX = 53.6 * MULTIPLIER;
const Y_MIN = -67.5 * MULTIPLIER;
const Y_MAX = 14.325 * MULTIPLIER;

let margin = {top: 20, right: 20, bottom: 30, left: 50},
    width = 600 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

let svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")")

let x = d3.scaleLinear().range([X_MIN, X_MAX]);
let y = d3.scaleLinear().range([Y_MIN, Y_MAX]);

x.domain([X_MIN, X_MAX]);
y.domain([Y_MIN, Y_MAX]);

var poly = [{"x":10, "y":50},
        {"x":20,"y":20},
        {"x":50,"y":10},
        {"x":30,"y":30}];


  svg.selectAll("polygon")
    .data([poly])
  .enter().append("polygon")
    .attr("points",function(d) {
        return d.map(function(d) {
            return [x(d.x),y(d.y)].join(",");
        }).join(" ");
    });

  svg.append("circle")
    .attr("r", 4)
    .attr("cx", x(point.x))
    .attr("cy", y(point.y))

    // add the X Axis
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // add the Y Axis
  svg.append("g")
    .call(d3.axisLeft(y));