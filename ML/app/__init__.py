from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/prediction", methods=["GET"])
def get_prediction(fromDate, toDate):

    result = {"x": 8, "y": 9, "z": 10}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)