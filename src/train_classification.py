import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import get_preprocessor


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("data/traffic_data_1000_rows.csv")


# ==============================
# 2. Prepare Date
# ==============================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day_of_Week"] = df["Date"].dt.dayofweek

df = df.drop("Date", axis=1)


# ==============================
# 3. Select X and Y
# ==============================

# Vehicle_Count is NOT used because
# Traffic_Level was created from Vehicle_Count.

X = df.drop(["Traffic_Level", "Vehicle_Count"], axis=1)

y = df["Traffic_Level"]


# ==============================
# 4. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 5. Create Preprocessor
# ==============================

preprocessor = get_preprocessor()


# ==============================
# 6. Create Random Forest Model
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==============================
# 7. Create Pipeline
# ==============================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==============================
# 8. Train Model
# ==============================

pipeline.fit(X_train, y_train)


# ==============================
# 9. Make Predictions
# ==============================

y_pred = pipeline.predict(X_test)


# ==============================
# 10. Evaluate Model
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("RANDOM FOREST CLASSIFICATION")
print("==============================")

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==============================
# 11. Save Model
# ==============================

joblib.dump(
    pipeline,
    "models/traffic_classifier.pkl"
)

print("\nModel saved successfully!")
print("Location: models/traffic_classifier.pkl")