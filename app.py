import os
from flask import Flask, render_template, request, jsonify
from src.GNNClassfier.pipeline.predict_pipeline import PredictionPipeline


app = Flask(__name__)

# Initialize the pipeline (update the model path as needed)
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/training/model.pt")

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        if os.path.exists(MODEL_PATH):
            pipeline = PredictionPipeline(MODEL_PATH)
    return pipeline


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "smiles" not in data:
        return jsonify({"error": "Missing SMILES input"}), 400

    smiles = data["smiles"].strip()
    if not smiles:
        return jsonify({"error": "SMILES string cannot be empty"}), 400

    pl = get_pipeline()
    if pl is None:
        return jsonify({
            "error": "Model not loaded. Please ensure model.pt exists at the configured path."
        }), 503

    result = pl.predict(smiles)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
