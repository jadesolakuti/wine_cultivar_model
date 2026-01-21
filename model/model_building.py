# =========================
# Wine Cultivar Model Build
# =========================

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# 2. Feature selection (6 features)
selected_features = [
    'alcohol',
    'malic_acid',
    'total_phenols',
    'flavanoids',
    'color_intensity',
    'proline'
]

X = X[selected_features]

# 3. Handle missing values (dataset has none, but be correct)
X.fillna(X.mean(), inplace=True)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Model selection and training
model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)

# 7. Evaluation
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 8. Save model and scaler
joblib.dump(model, "model/wine_cultivar_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
