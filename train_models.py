import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load training and testing data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Features
features = ["Mar-May", "June", "June_change"]

X_train = train[features]
y_train = train["Flood_Risk"]

X_test = test[features]
y_test = test["Flood_Risk"]

# Create models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC()
}

results = {}

# Train and evaluate each model
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results[name] = [accuracy, precision, recall, f1]

    print("\n", name)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))


# Save the models
joblib.dump(models, "all_models.pkl")

print("\nAll models trained and saved successfully!")
print("Best model saved successfully")

# Save the best model separately
joblib.dump(
    models["Random Forest"],
    "random_forest_model.pkl"
)

print("Best model saved successfully!")