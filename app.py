from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load model and scaler
model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            alcohol = float(request.form["alcohol"])
            malic_acid = float(request.form["malic_acid"])
            total_phenols = float(request.form["total_phenols"])
            flavanoids = float(request.form["flavanoids"])
            color_intensity = float(request.form["color_intensity"])
            proline = float(request.form["proline"])

            input_data = np.array([[alcohol, malic_acid, total_phenols,
                                    flavanoids, color_intensity, proline]])

            input_scaled = scaler.transform(input_data)
            result = model.predict(input_scaled)[0]

            prediction = f"Cultivar {result + 1}"

        except Exception as e:
            print(f"Error: {e}") # This tells you EXACTLY what went wrong
            prediction = "An internal error occurred."
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
