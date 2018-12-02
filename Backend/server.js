// server.js
var express = require('express');
var app = express();
var bodyParser = require('body-parser');
var data = require('./data.json');
var csv = require("csvtojson");
var cors = require('cors')

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cors());

var port = process.env.PORT || 8080;
var router = express.Router();

var itemsJson = "";
var predictionsJson = "";

csv()
    .fromFile("data.csv")
    .then(function (jsonArrayObj) {
        console.log(jsonArrayObj);
        itemsJson = jsonArrayObj;
    });

csv()
    .fromFile("predicted-data.csv")
    .then(function (jsonArrayObj) {
        console.log(jsonArrayObj);
        predictionsJson = jsonArrayObj;
    });

router.use(function (req, res, next) {
    console.log('Something is happening.');
    next();
});

router.get('/', function (req, res) {
    res.json({ message: 'hooray! welcome to our api!' });
});

router.route('/items')
    // GET http://localhost:8080/api/items
    .get(
        function (req, res) {
            res.json(itemsJson);
        }
    );


router.route('/predictions')
    // GET http://localhost:8080/api/items
    .get(
        function (req, res) {
            res.json(predictionsJson);
        }
    );

app.use('/api', router);

// all of our routes will be prefixed with /api
app.listen(port);
console.log('Magic happens on port ' + port);

