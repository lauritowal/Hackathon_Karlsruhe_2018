from flask import Flask, request, jsonify
app = Flask(__name__)
from forecast import forecast


@app.route("/prediction", methods=["GET"])
def get_prediction():

    #format toDate
    #result = forecast("../rnn/24_checkpoint.keras", "../rnn/data.csv",
    #                  "2019-02-10 10:10:10")

    return jsonify(
        forecast("../rnn/24_checkpoint.keras", "../rnn/data.csv",
                 "2019-01-01 10:10:10"))


if __name__ == '__main__':
    app.run(debug=True)