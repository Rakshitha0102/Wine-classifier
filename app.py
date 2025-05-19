from flask import Flask, request, render_template_string
import joblib
from sklearn.datasets import load_wine

app = Flask(__name__)
model = joblib.load('model.pkl')
data = load_wine()

class_names = {
    0: "Class 0 - Barbera",
    1: "Class 1 - Barolo",
    2: "Class 2 - Grignolino"
}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Wine Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        
    body {
        background: 
          linear-gradient(135deg, rgba(248,249,250,0.85), rgba(223,233,243,0.85)),
          url('/static/wine.jpg') no-repeat center center fixed;
        background-size: cover;
        min-height: 100vh;
        display: flex;
        justify-content: center;
        align-items: flex-start;
        padding: 40px 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    

        .container {
            max-width: 700px;
            background: white;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        h1 {
            margin-bottom: 35px;
            font-weight: 700;
            color: #343a40;
            text-align: center;
        }
        label {
            font-weight: 600;
            color: #495057;
        }
        .btn-primary {
            background-color: #6f42c1;
            border-color: #6f42c1;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        .btn-primary:hover {
            background-color: #5a32a3;
            border-color: #5a32a3;
        }
        .result-card {
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            background: #e9f7ef;
            color: #155724;
            box-shadow: 0 4px 12px rgba(0, 128, 0, 0.15);
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 1.2rem;
            font-weight: 600;
        }
        .result-card i {
            font-size: 2rem;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Wine Classifier</h1>
    <form method="POST" action="/">
        <div class="row justify-content-center">
            {% for i, feature in enumerate(feature_names) %}
            <div class="col-md-4 col-sm-6 mb-3">
                <label for="feature{{ i }}">{{ feature }}</label>
                <input type="number" step="any" class="form-control text-center" id="feature{{ i }}" name="feature{{ i }}" required>
            </div>
            {% endfor %}
        </div>
        <button type="submit" class="btn btn-primary w-100 mt-3">Predict Wine Class</button>
    </form>

    {% if prediction is not none %}
    <div class="result-card mt-4">
        <i class="fa-solid fa-wine-glass"></i>
        <div>Predicted Wine Class: <strong>{{ class_name }}</strong> (ID: {{ prediction }})</div>
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    class_name = None
    if request.method == 'POST':
        try:
            features = [float(request.form[f'feature{i}']) for i in range(len(data.feature_names))]
            prediction = model.predict([features])[0]
            class_name = class_names.get(prediction, "Unknown Class")
        except Exception as e:
            return f"Error: {str(e)}"
    return render_template_string(
        HTML, 
        feature_names=data.feature_names, 
        prediction=prediction, 
        class_name=class_name, 
        enumerate=enumerate  # pass enumerate explicitly to Jinja
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
