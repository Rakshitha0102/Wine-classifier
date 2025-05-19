from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load wine dataset
data = load_wine()
X, y = data.data, data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to file
joblib.dump(model, 'model.pkl')
print("Model trained and saved as model.pkl")
