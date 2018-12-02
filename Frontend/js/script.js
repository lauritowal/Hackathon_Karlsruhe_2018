const MULTIPLIER = 9;
const POINT_MULTIPLIER = MULTIPLIER - 1;
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
  .attr("viewBox", `${-width / VIEW_POINT_DIVIDER_X} ${-height / VIEW_POINT_DIVIDER_Y} ${width} ${height}`)
  .style("border", "1px solid black")
  .style("stroke-width", 2);

showAreas();
// loadItems("2018-07-01", "2018-07-05");

function onLoadButtonClick(event) {
  let fromDate = document.getElementById("fromDate").value;
  let toDate = document.getElementById("toDate").value;

  svg.selectAll("circle").remove();
  d3.select("#path").remove();

  loadItems(fromDate, toDate);
}

function onPredictionButtonClick(event) {
  let fromDate = document.getElementById("fromDate").value;
  let toDate = document.getElementById("toDate").value;

  d3.select("#path").remove();

  loadPredictedItems(fromDate, toDate);
}

function loadPredictedItems(fromDate, toDate) {
  const xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      showItems(this.responseText, fromDate, toDate, "green");
    }
  };
  xhttp.open("GET", `http://localhost:8080/api/predictions`, true);
  xhttp.send();
}

function getPointsArray(x, y) {
  return [(x + (POINT_X_OFFSET)) * POINT_MULTIPLIER, (y + (POINT_Y_OFFSET)) * POINT_MULTIPLIER]
}

function showAreas() {
  AREAS.forEach(area => {
    try {
      let shape = JSON.parse(area.shape);
      let points = "";
      shape.coordinates.forEach(points => {

        points = points.map(point => getPointsArray(point[0], point[1]));

        svg.append("polygon")
          .attr("points", point_s)
          .style("fill", "none")
          .style("stroke", "black")
          .style("strokeWidth", "10px");
      });
    } catch (e) {
      console.warn("JSON is not valid:", e)
    }
  });
}

const waitFor = (ms) => new Promise(r => setTimeout(r, ms))
async function addDot(point, color = red, address, z) {
  setTimeout(() => {
    svg.append("circle")
      .attr("cx", point[0])
      .attr("cy", point[1])
      .attr("r", 5)
      .on("mouseout", function () {
        this.style.fill = color;
        this.style.opacity = 0.15;
      })
      .on("mouseover", function () {
        this.style.fill = "blue";
        this.style.opacity = 1;
        document.getElementById('info').innerHTML = `
          <h5>Spot Info</h5>
          <b>Address</b> ${address}<br>
          <b>Process (demo)</b> FI3MEGH<br>
          <b>X</b> ${point[0]}
          <b>Y</b> ${point[1]}
          <b>Z</b> ${(z + (POINT_X_OFFSET)) * POINT_MULTIPLIER}
        `;
      })
      .style("fill", color)
      .style("opacity", 0.15)
      .style("z-index", 2);
  }, 30)
}

async function showItems(items, fromDate, toDate, color = "red") {
  fromDate = new Date(fromDate);
  toDate = new Date(toDate);

  items = JSON.parse(items);
  items = items.filter(item => new Date(item.timestamp) >= fromDate && new Date(item.timestamp) <= toDate);

  items.map(async (item) => {
    await addDot(
      getPointsArray(parseInt(item.x, 10), parseInt(item.y, 10)),
      color,
      item.address,
      item.z
    )
  });

  let points = items.map(item => getPointsArray(parseInt(item.x, 10), parseInt(item.y, 10)));
  setTimeout(() => {
    svg.append("polygon")
      .attr("id", "path")
      .attr("points", points)
      .style("fill", "none")
      .style("stroke", color)
      .style("strokeWidth", "10px")
  }, 1);
}

function loadItems(fromDate, toDate) {
  const xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      showItems(this.responseText, fromDate, toDate);
    }
  };
  xhttp.open("GET", `http://localhost:8080/api/items`, true);
  xhttp.send();
}
