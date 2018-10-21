// server.js
var express    = require('express');        
var app        = express();                
var bodyParser = require('body-parser');
var data = require('./data.json');
var csv = require("csvtojson");

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

var port = process.env.PORT || 8080;
var router = express.Router();

var itemsJson = "test";

csv()
.fromFile("data.csv")
.then(function(jsonArrayObj){ 
    console.log(jsonArrayObj); 
    itemsJson = jsonArrayObj;
});

router.use(function(req, res, next) {
    console.log('Something is happening.');
    next(); 
});

router.get('/', function(req, res) {
    res.json({ message: 'hooray! welcome to our api!' });   
});

router.route('/items')
    // GET http://localhost:8080/api/items
    .get(
        function(req, res) {
            res.json(itemsJson);
        }
    );

app.use('/api', router);

// all of our routes will be prefixed with /api
app.listen(port);
console.log('Magic happens on port ' + port);

