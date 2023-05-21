from flask import Flask, render_template, request, url_for
from predictor import get_prediction

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    prediction = get_prediction(
        request.form["age"],
        request.form["experience"],
        request.form["income"],
        request.form["family"],
        request.form["ccAvg"],
        request.form["education"],
        request.form["mortgage"],
        request.form["personalLoan"],
        request.form["securitiesAccount"],
        request.form["cdAccount"],
        request.form["online"],
    )
    return render_template("predict.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
